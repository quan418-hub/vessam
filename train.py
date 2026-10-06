# -*- coding: utf-8 -*-
"""
VesSAM 训练主脚本 v2
====================
修复 shuffle_multimodal + no_mask_embed bug 后，加入：
- 同步数据增强（翻转/旋转 + 颜色抖动）
- 差分学习率（新增层大 LR，SAM 预训练层小 LR）
- loss 历史记录 + 自动保存最优权重

运行：
    python train.py --epochs 200 --batch-size 2 --lr-new 5e-4 --lr-sam 1e-5
"""
import os
import argparse
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_vessam import build_vessam_model
from Dataloader import VesselDataset
from utils.loss import FocalDiceloss_IoULoss
from utils.train_val_test_related import prompt_and_decoder


# ---------------- 基础变换 ----------------
class TrainTransform:
    MEAN = [123.675, 116.28, 103.53]
    STD = [58.395, 57.12, 57.375]

    def __init__(self, size=512):
        self.size = size

    def __call__(self, x):
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if x.ndim == 2:
            b = (x > 127).astype(np.float32)
            return torch.from_numpy(b).unsqueeze(0)
        t = torch.from_numpy(x).float()
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        return (t.permute(2, 0, 1) - mean) / std


# ---------------- 同步数据增强 ----------------
def augment_batch(batch, size=512):
    if random.random() < 0.5:
        return batch

    B = batch["image"].shape[0]
    new_image = batch["image"].clone()
    new_mask = batch["mask"].clone()
    new_skeleton = batch["skeleton"].clone()
    new_branch = batch["branch_points"].clone() if "branch_points" in batch else None
    new_mid = batch["mid_points"].clone() if "mid_points" in batch else None

    for b in range(B):
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        k = random.choice([0, 1, 2, 3])

        img = new_image[b]
        msk = new_mask[b]
        skl = new_skeleton[b]

        if flip_h:
            img = torch.flip(img, [2])
            msk = torch.flip(msk, [2])
            skl = torch.flip(skl, [2])
        if flip_v:
            img = torch.flip(img, [1])
            msk = torch.flip(msk, [1])
            skl = torch.flip(skl, [1])

        if k > 0:
            img = torch.rot90(img, k, [1, 2])
            msk = torch.rot90(msk, k, [1, 2])
            skl = torch.rot90(skl, k, [1, 2])

        new_image[b] = img
        new_mask[b] = msk
        new_skeleton[b] = skl

        if new_branch is not None:
            pts = new_branch[b].numpy()
            pts = _transform_points(pts, flip_h, flip_v, k, size)
            new_branch[b] = torch.from_numpy(pts).float()
        if new_mid is not None:
            pts = new_mid[b].numpy()
            pts = _transform_points(pts, flip_h, flip_v, k, size)
            new_mid[b] = torch.from_numpy(pts).float()

    batch["image"] = new_image
    batch["mask"] = new_mask
    batch["skeleton"] = new_skeleton
    if new_branch is not None:
        batch["branch_points"] = new_branch
    if new_mid is not None:
        batch["mid_points"] = new_mid
    return batch


def _transform_points(pts, flip_h, flip_v, k, size):
    """同步翻转+旋转点坐标 (k=0:无, 1:90, 2:180, 3:270)"""
    out = pts.copy().astype(np.float32)
    if flip_h:
        out[:, 0] = size - out[:, 0]
    if flip_v:
        out[:, 1] = size - out[:, 1]
    if k == 1:
        out = np.stack([size - out[:, 1], out[:, 0]], axis=1)
    elif k == 2:
        out = np.stack([size - out[:, 0], size - out[:, 1]], axis=1)
    elif k == 3:
        out = np.stack([out[:, 1], size - out[:, 0]], axis=1)
    return out


# ---------------- 差分学习率 ----------------
def get_param_groups(model, lr_new, lr_sam):
    """
    新增层（prompt_encoder 扩展层 + graph + cross_attention + no_mask_embed）用大 LR
    SAM 原生层（mask_decoder + prompt_encoder 原始点嵌入）用小 LR
    """
    new_modules = [
        "prompt_encoder.skeleton_downscaling",
        "prompt_encoder.mask_downscaling",
        "prompt_encoder.no_mask_embed",
        "prompt_encoder.cross_attention",
        "prompt_encoder.graph_embedding",
        "prompt_encoder.graph_proj",
        "prompt_encoder.final_conv",
        "prompt_encoder.three_cross_transformer",
        "prompt_encoder.branch_point_embed",
        "prompt_encoder.mid_point_embed",
    ]

    new_params = []
    sam_params = []
    new_named = []
    sam_named = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = any(n in name for n in new_modules)
        if is_new:
            new_params.append(param)
            new_named.append(name)
        else:
            sam_params.append(param)
            sam_named.append(name)

    print(f"[差分LR] 新增层 lr={lr_new}: {len(new_params)} 组, {sum(p.numel() for p in new_params)/1e6:.2f}M")
    print(f"[差分LR] SAM层 lr={lr_sam}: {len(sam_params)} 组, {sum(p.numel() for p in sam_params)/1e6:.2f}M")
    print(f"  新增层示例: {new_named[:5]}")
    print(f"  SAM层示例: {sam_named[:5]}")

    return [
        {"params": new_params, "lr": lr_new},
        {"params": sam_params, "lr": lr_sam},
    ]


# ---------------- 训练 ----------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = build_vessam_model(image_size=args.image_size,
                               checkpoint=args.sam_checkpoint,
                               adapter_train=args.encoder_adapter).to(device)
    model.train()

    if args.freeze_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
        print("[冻结] 图像编码器 image_encoder")

    transform = TrainTransform(args.image_size)
    dataset = VesselDataset(args.dataset_name, args.data_dir,
                            split="all", transform=transform,
                            max_branch_points=64, max_mid_points=32)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        drop_last=False)
    print(f"样本数: {len(dataset)}，批数: {len(loader)}")

    criterion = FocalDiceloss_IoULoss(weight=args.loss_weight)
    param_groups = get_param_groups(model, args.lr_new, args.lr_sam)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    work_dir = os.path.join(BASE, "work_dir")
    os.makedirs(work_dir, exist_ok=True)

    history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for bi, batch in enumerate(loader):
            batch = augment_batch(batch, args.image_size)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            image_embeddings = model.image_encoder(batch["image"])
            pred, low_res, iou = prompt_and_decoder(args, batch, model, image_embeddings,
                                                    shuffle_multimodal=True)
            loss = criterion(pred, batch["mask"], iou)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg = np.mean(epoch_losses)
        history.append(avg)

        if avg < best_loss:
            best_loss = avg
            best_path = os.path.join(BASE, "temproot", "vessam_best.pth")
            torch.save(model.state_dict(), best_path)

        if epoch % args.save_every == 0 or epoch == args.epochs or epoch == 1:
            last10 = history[-10:]
            print(f"[Epoch {epoch:3d}/{args.epochs}] loss={avg:.4f}  "
                  f"best={best_loss:.4f}  "
                  f"lr_new={optimizer.param_groups[0]['lr']:.2e}  "
                  f"lr_sam={optimizer.param_groups[1]['lr']:.2e}")

    final = os.path.join(BASE, "temproot", "vessam_final.pth")
    torch.save(model.state_dict(), final)
    print(f"\n训练完成!")
    print(f"  最终权重: {final}")
    print(f"  最优权重: {os.path.join(BASE, 'temproot', 'vessam_best.pth')}")
    print(f"  Loss 曲线: {[round(l, 4) for l in history[:10]]}...{[round(l, 4) for l in history[-5:]]}")
    print(f"  Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr-new", type=float, default=5e-4,
                    help="新增层学习率 (skeleton_downscaling/cross_attention等)")
    ap.add_argument("--lr-sam", type=float, default=1e-5,
                    help="SAM预训练层学习率 (mask_decoder等)")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-multimask-outputs", type=int, default=3)
    ap.add_argument("--loss-weight", type=float, default=20.0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--freeze-encoder", type=lambda s: s.lower() in ("1","true","yes"), default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--dataset-name", type=str, default="DRIVE")
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--sam_checkpoint", type=str, default="temproot/sam_vit_b_01ec64.pth")
    ap.add_argument("--encoder_adapter", type=lambda s: s.lower() in ("1","true","yes"), default=False)
    args = ap.parse_args()
    train(args)
