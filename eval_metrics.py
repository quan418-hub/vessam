# -*- coding: utf-8 -*-
"""
VesSAM 评估脚本：在所有 DRIVE 样本上跑推理，计算 Dice 和 IoU 指标。

用法（服务器上，用训练好的权重）：
    python eval_metrics.py --weight temproot/vessam_final.pth
    python eval_metrics.py --weight temproot/vessam_final.pth --max-idx 5   # 只看前5张

输出：终端打印每张的 Dice/IoU，以及全数据集平均。同时保存 ./work_dir/metrics_report.txt
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_vessam import build_vessam_model
from Dataloader import VesselDataset
from utils.train_val_test_related import prompt_and_decoder


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


def dice_coef(pred, gt):
    """Dice = 2*|P∩G| / (|P|+|G|)，pred/gt 均为二值 0/1"""
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * inter) / (denom + 1e-6)


def iou_score(pred, gt):
    """IoU = |P∩G| / |P∪G|"""
    inter = (pred * gt).sum()
    union = ((pred + gt) > 0).float().sum()
    return inter / (union + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=str, default="temproot/vessam_final.pth")
    ap.add_argument("--dataset-name", type=str, default="DRIVE")
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--max-idx", type=int, default=999999, help="最多评估前几张")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-multimask-outputs", type=int, default=3)
    ap.add_argument("--sam-checkpoint", type=str, default="temproot/sam_vit_b_01ec64.pth")
    ap.add_argument("--encoder-adapter", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = build_vessam_model(image_size=args.image_size,
                               checkpoint=args.sam_checkpoint).to(device)
    model.eval()

    if os.path.exists(args.weight):
        sd = torch.load(args.weight, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        model_sd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model_sd.update(sd)
        model.load_state_dict(model_sd)
        print(f"[权重] 已加载训练权重: {args.weight} ({len(sd)} 项)")
    else:
        print(f"[警告] 找不到 {args.weight}！结果无效。")

    ds = VesselDataset(args.dataset_name, args.data_dir, split="all",
                       transform=TrainTransform(args.image_size))
    n = min(len(ds), args.max_idx)
    print(f"将评估 {n} 张样本")

    dices, ious = [], []
    lines = []
    with torch.no_grad():
        for idx in range(n):
            batch = ds[idx]
            batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            emb = model.image_encoder(batch["image"])
            infer_batch = {k: v for k, v in batch.items() if k != "mask"}
            preds, low_res, _ = prompt_and_decoder(args, infer_batch, model, emb,
                                                   shuffle_multimodal=False)
            preds = torch.sigmoid(preds)
            binary = (preds > 0.5).float()[0, 0]          # (512,512)
            gt = batch["mask"][0, 0]                      # (512,512)

            d = dice_coef(binary, gt).item()
            iou_val = iou_score(binary, gt).item()
            dices.append(d)
            ious.append(iou_val)
            line = f"sample {idx:>2d}: Dice={d:.4f}  IoU={iou_val:.4f}"
            print(line)
            lines.append(line)

    md = float(np.mean(dices))
    mi = float(np.mean(ious))
    print("=" * 40)
    print(f"平均 Dice = {md:.4f}")
    print(f"平均 IoU  = {mi:.4f}")
    lines += ["", "=" * 40, f"平均 Dice = {md:.4f}", f"平均 IoU  = {mi:.4f}"]

    rep = os.path.join(BASE, "work_dir", "metrics_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[保存] 指标报告: {rep}")


if __name__ == "__main__":
    main()
