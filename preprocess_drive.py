# -*- coding: utf-8 -*-
"""
VesSAM 数据预处理脚本 v3
========================
修复分支点检测阈值（从 4 改回 3，骨架分叉点天然 3 邻居）
+ 增强中点提取（分支点间隔采样确保均匀覆盖 + 保留更多段）
+ 剪毛刺更温和

运行：python preprocess_drive.py
"""
import os
import json
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from skimage.measure import label

BASE = os.path.dirname(os.path.abspath(__file__))
DRIVE_DIR = os.path.join(BASE, "dataset", "DRIVE")
OUT_DIR = os.path.join(BASE, "data", "DRIVE")
IMG_SIZE = 512

# ===== 参数 =====
SPUR_LEN = 5            # 短于这个长度的骨架分支视为毛刺（原8，改5温和些）
BRANCH_MIN_NB = 3       # 分支点邻居阈值（改回3，骨架分叉点天然3邻居）
BRANCH_MIN_DIST = 8     # 两个分支点距离小于这个值就合并（原5，避免重复）
MIN_SEG_LEN = 15        # 长线段最小长度（原30→15，保留更多血管段）
MID_MIN_DIST_BRANCH = 8  # 中点离分支点至少要这么远


def ensure(path):
    os.makedirs(path, exist_ok=True)


def load_image(path):
    im = Image.open(path)
    if im.mode in ('I;16', 'I', 'F'):
        arr = np.array(im.convert('I'), dtype=np.float32)
        mx, mn = arr.max(), arr.min()
        arr = (arr - mn) / (mx - mn + 1e-9) * 255
        arr = arr.astype(np.uint8)
    else:
        arr = np.array(im.convert('RGB'))
    return arr


def load_binary(path):
    arr = np.array(Image.open(path).convert('L'))
    return (arr > 127).astype(np.uint8) * 255


def resize(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def extract_skeleton(mask_bin):
    return skeletonize(mask_bin > 0)


def prune_skeleton(skel, min_spur_len=SPUR_LEN):
    """剪掉骨架上的短毛刺"""
    s = skel.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    for _ in range(10):
        nc = convolve(s, kernel, mode='constant', cval=0)
        endpoints = (s == 1) & (nc == 1)
        if not endpoints.any():
            break

        ys, xs = np.where(endpoints)
        removed_any = False
        for y, x in zip(ys, xs):
            path = [(y, x)]
            cy, cx = y, x
            prev = None
            spur = True
            for _ in range(min_spur_len + 5):
                nxt = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < s.shape[0] and 0 <= nx < s.shape[1]
                                and s[ny, nx] == 1 and (ny, nx) != prev):
                            nxt.append((ny, nx))
                if len(nxt) == 0:
                    break
                if len(nxt) >= 2:
                    spur = False
                    break
                prev = (cy, cx)
                cy, cx = nxt[0]
                path.append((cy, cx))

            if spur and len(path) < min_spur_len:
                for py, px in path:
                    s[py, px] = 0
                removed_any = True

        if not removed_any:
            break
    return s


def detect_branch_points(skel, min_nb=BRANCH_MIN_NB, min_dist=BRANCH_MIN_DIST):
    """检测分支点：邻居数>=min_nb，且邻近点聚类去重"""
    s = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nc = convolve(s, kernel, mode='constant', cval=0)
    raw = np.argwhere((s == 1) & (nc >= min_nb))
    if len(raw) == 0:
        return raw

    kept = []
    used = set()
    raw_list = [tuple(p) for p in raw]
    for i, pt in enumerate(raw_list):
        if i in used:
            continue
        cluster = [pt]
        used.add(i)
        for j in range(i + 1, len(raw_list)):
            if j in used:
                continue
            d = np.hypot(raw_list[j][0] - pt[0], raw_list[j][1] - pt[1])
            if d < min_dist:
                cluster.append(raw_list[j])
                used.add(j)
        cy = int(np.mean([p[0] for p in cluster]))
        cx = int(np.mean([p[1] for p in cluster]))
        kept.append((cy, cx))
    return np.array(kept)


def extract_mid_points(skel, branch_pts, min_seg=MIN_SEG_LEN, min_d=MID_MIN_DIST_BRANCH):
    """
    改进版：每个血管段提取中点 + 长段提取额外中间点（保证均匀覆盖）
    """
    s = skel.copy()
    for y, x in branch_pts:
        s[y, x] = 0

    lab, n = label(s, connectivity=2, return_num=True)
    mids = []
    for i in range(1, n + 1):
        coords = np.argwhere(lab == i)
        if len(coords) < min_seg:
            continue

        # 计算沿路径的点（从上到下 / 从左到右排序）
        # 用简单的排序：先按 y，再按 x
        if coords[:, 0].max() - coords[:, 0].min() > coords[:, 1].max() - coords[:, 1].min():
            order = np.argsort(coords[:, 0])
        else:
            order = np.argsort(coords[:, 1])
        ordered = coords[order]

        # 段太长的话，提取多个中间点
        n_extra = min(len(ordered) // (min_seg + 5), 3)  # 最多3个额外点
        for k in range(n_extra + 1):
            idx = int(len(ordered) * (k + 1) / (n_extra + 2))
            my, mx = int(ordered[idx][0]), int(ordered[idx][1])
            too_close = False
            for by, bx in branch_pts:
                if np.hypot(my - by, mx - bx) < min_d:
                    too_close = True
                    break
            if not too_close:
                mids.append((my, mx))
    return mids


def main():
    train_img = os.path.join(DRIVE_DIR, "training", "images")
    train_man = os.path.join(DRIVE_DIR, "training", "1st_manual")

    d_img = os.path.join(OUT_DIR, "images")
    d_mask = os.path.join(OUT_DIR, "masks")
    d_skel = os.path.join(OUT_DIR, "skeletons")
    d_branch = os.path.join(OUT_DIR, "branch_points")
    d_mid = os.path.join(OUT_DIR, "mid_points")
    for d in [d_img, d_mask, d_skel, d_branch, d_mid]:
        ensure(d)

    files = sorted(os.listdir(train_img))
    info = {}
    done = 0
    for f in files:
        if not f.lower().endswith(('.tif', '.png', '.jpg')):
            continue
        stem = os.path.splitext(f)[0]
        num = stem.split('_')[0]
        man_name = f"{num}_manual1.gif"
        man_path = os.path.join(train_man, man_name)
        if not os.path.exists(man_path):
            print(f"[跳过] 缺少对应标注: {man_name}")
            continue

        img = resize(load_image(os.path.join(train_img, f)))
        mask = resize(load_binary(man_path))
        mask_bin = (mask > 127).astype(np.uint8)

        skel = extract_skeleton(mask_bin)
        skel = prune_skeleton(skel)
        branch_pts = detect_branch_points(skel)
        mid_pts = extract_mid_points(skel, branch_pts)

        branch_xy = [[int(pt[1]), int(pt[0])] for pt in branch_pts]
        mid_xy = [[int(pt[1]), int(pt[0])] for pt in mid_pts]

        Image.fromarray(img).save(os.path.join(d_img, f"{stem}.png"))
        Image.fromarray((mask_bin * 255).astype(np.uint8)).save(os.path.join(d_mask, f"{stem}.png"))
        Image.fromarray((skel.astype(np.uint8) * 255)).save(os.path.join(d_skel, f"{stem}.png"))
        with open(os.path.join(d_branch, f"{stem}.json"), 'w') as fh:
            json.dump(branch_xy, fh)
        with open(os.path.join(d_mid, f"{stem}.json"), 'w') as fh:
            json.dump(mid_xy, fh)

        info[stem] = {
            "image_path": f"data/DRIVE/images/{stem}.png",
            "mask_path": f"data/DRIVE/masks/{stem}.png",
            "skeleton_path": f"data/DRIVE/skeletons/{stem}.png",
            "branch_points_path": f"data/DRIVE/branch_points/{stem}.json",
            "mid_points_path": f"data/DRIVE/mid_points/{stem}.json",
        }
        done += 1
        print(f"[OK] {stem}: 分支点 {len(branch_xy)}, 中点 {len(mid_xy)}")

    with open(os.path.join(OUT_DIR, "dataset_info.json"), 'w') as fh:
        json.dump(info, fh, indent=2)

    print(f"\n完成！样本数: {done}")
    print(f"数据目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
