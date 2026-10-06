# -*- coding: utf-8 -*-
"""
VesSAM 推理 demo：加载训练好的权重，用 DRIVE 数据集的一个样本生成血管分割图。

用法：
    python test_infer.py                 # 默认推理第 0 张，加载 temproot/vessam_final.pth
    python test_infer.py --idx 3         # 推理第 3 张
    python test_infer.py --weight work_dir/vessam_epoch50.pt   # 换权重

输出：
    work_dir/result{idx}.png  （原图 / 真值 / 骨架分支点 / 预测 / 差异 / 叠加 六宫格）
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from build_vessam import build_vessam_model
from Dataloader import VesselDataset
from utils.train_val_test_related import prompt_and_decoder, visualize_results


# ---------------- 与训练一致的数据预处理（resize 到 512 + 归一化/二值化） ----------------
class TrainTransform:
    MEAN = [123.675, 116.28, 103.53]
    STD = [58.395, 57.12, 57.375]

    def __init__(self, size=512):
        self.size = size

    def __call__(self, x):
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if x.ndim == 2:  # mask / skeleton -> 二值 0/1
            b = (x > 127).astype(np.float32)
            return torch.from_numpy(b).unsqueeze(0)
        # image: RGB (H,W,3) -> 归一化 (3,H,W)
        t = torch.from_numpy(x).float()
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        return (t.permute(2, 0, 1) - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=str, default="temproot/vessam_final.pth",
                    help="训练好的权重路径")
    ap.add_argument("--dataset-name", type=str, default="DRIVE")
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--idx", type=int, default=0, help="推理第几张样本（0 开始）")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--num-multimask-outputs", type=int, default=3)
    ap.add_argument("--sam-checkpoint", type=str, default="temproot/sam_vit_b_01ec64.pth")
    ap.add_argument("--encoder-adapter", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 构建模型（与训练一致）
    model = build_vessam_model(image_size=args.image_size,
                               checkpoint=args.sam_checkpoint).to(device)
    model.eval()

    # 加载训练好的权重（vessam_final.pth 是整个模型 state_dict）
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
        print(f"[警告] 找不到 {args.weight}，将用 SAM 预训练(未训练)权重推理，效果差！")

    # 加载数据集并取一个样本
    ds = VesselDataset(args.dataset_name, args.data_dir, split="all",
                       transform=TrainTransform(args.image_size))
    print(f"数据集样本数: {len(ds)}，当前推理第 {args.idx} 张")
    batch = ds[args.idx]
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    # 前向推理：真实推理场景不提供 GT mask
    with torch.no_grad():
        image_embeddings = model.image_encoder(batch["image"])
        infer_batch = {k: v for k, v in batch.items() if k != "mask"}
        preds, low_res, iou = prompt_and_decoder(args, infer_batch, model, image_embeddings,
                                                 shuffle_multimodal=False)
        preds = torch.sigmoid(preds)
        binary = (preds > 0.5).float()

    print(f"预测 mask: {tuple(binary.shape)}，iou 预测: {iou.detach().cpu().numpy().flatten()}")

    # 画六宫格对比图，保存到 work_dir/result{idx}.png
    visualize_results(
        image=batch["image"][0],
        mask=batch["mask"][0],
        branch_points=batch["branch_points"][0],
        mid_points=batch["mid_points"][0],
        skeleton=batch["skeleton"][0],
        count=args.idx,
        predicts=binary[0],
        save_path="",
        file_name="result.png",
    )
    out = os.path.join(BASE, "work_dir", f"result{args.idx}.png")
    print(f"[完成] 分割结果图已保存: {out}")


if __name__ == "__main__":
    main()
