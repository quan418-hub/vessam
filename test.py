import torch
import cv2
import numpy as np
import sys
sys.path.append(".")

# 已修正：导入对应文件里的 Ves_SAM 类
from Models.Ves_SAM import Ves_SAM

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型，类名同步改为 Ves_SAM
model = Ves_SAM(
    image_encoder_type="vit_b",
    checkpoint_path="temproot/sam_vit_b_01ec64.pth"
).to(device)
model.eval()

# ========== 改成你 data_demo 文件夹里实际的图片文件名 ==========
img_path = "data_demo/sample.png"
# ================================================================

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# 点提示（图片中心点）
point_coords = torch.tensor([[w//2, h//2]], device=device)
point_labels = torch.tensor([1], device=device)

# 推理
with torch.no_grad():
    masks, iou_predictions = model.predict(
        image=img_rgb,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )

# 取置信度最高的结果
best_idx = torch.argmax(iou_predictions)
mask = masks[best_idx].cpu().numpy().astype(np.uint8) * 255
confidence = iou_predictions[best_idx].item()

# 保存纯掩码图
cv2.imwrite("result_mask.png", mask)

# 保存叠加效果图（血管标红）
overlay = img.copy()
overlay[mask > 127] = [0, 0, 255]
cv2.addWeighted(overlay, 0.5, img, 0.5, 0, overlay)
cv2.imwrite("result_overlay.png", overlay)

print("✅ 运行完成！")
print(f"分割置信度：{round(confidence, 4)}")
print("结果已保存：result_mask.png、result_overlay.png")