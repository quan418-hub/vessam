# -*- coding: utf-8 -*-
"""
VesSAM 文件夹里跑通官方 SAM 权重，对 data_demo 里的血管图做点提示分割，
输出分割效果图。用于先看到血管分割效果。
"""
import os
import numpy as np
from PIL import Image
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

BASE = r"C:\Users\曲乐乐\Desktop\VesSAM"
CHECKPOINT = os.path.join(BASE, "temproot", "sam_vit_b_01ec64.pth")
IMG_NAME = "Retinal1.png"  # 想换图就改这里，如 Aorta1.png / XCAD1.png
IMG_PATH = os.path.join(BASE, "data_demo", IMG_NAME)

# 1. 读图（用PIL避免中文路径问题），转RGB
im = Image.open(IMG_PATH).convert("RGB")
img = np.array(im)
print(f"读入图片: {IMG_NAME}  尺寸: {img.shape[1]}x{img.shape[0]}")

# 2. 加载官方 SAM vit_b 权重
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT).to(device)
sam.eval()
predictor = SamPredictor(sam)
predictor.set_image(img)

# 3. 点提示：在图上放多个候选点（含中心点），提升分割命中
h, w = img.shape[:2]
input_point = np.array([
    [w // 2, h // 2],            # 中心
    [w // 3, h // 3],
    [2 * w // 3, h // 3],
    [w // 3, 2 * h // 3],
    [2 * w // 3, 2 * h // 3],
    [w // 2, h // 3],
    [w // 2, 2 * h // 3],
], dtype=float)
input_label = np.ones(len(input_point), dtype=int)

# 4. 推理，取置信度最高的掩码
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
best = int(np.argmax(scores))
mask = masks[best]
print(f"分割完成，最高置信度: {scores[best]:.3f}")

# 5. 保存纯掩码图
mask_img = (mask * 255).astype(np.uint8)
mask_out = os.path.join(BASE, "result_sam_mask.png")
Image.fromarray(mask_img).save(mask_out)

# 6. 保存叠加图（原图 + 血管标红）
overlay = img.copy()
red = np.zeros_like(img); red[..., 0] = 255
overlay[mask > 0.5] = red[mask > 0.5]
overlay_img = (0.6 * img + 0.4 * overlay).astype(np.uint8)
overlay_out = os.path.join(BASE, "result_sam_overlay.png")
# 用BGR保存，否则颜色会反过来
cv2.imwrite(os.path.join(BASE, "result_sam_overlay_bgr.png"),
            cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
Image.fromarray(overlay_img).save(overlay_out)

print("已保存:")
print("  " + mask_out)
print("  " + overlay_out)
print("完成！")
