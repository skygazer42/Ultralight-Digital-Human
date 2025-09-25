# mouth_roi.py
from __future__ import annotations
import numpy as np

def _clip_square(cx, cy, side, W, H):
    xmin = int(max(0, cx - side/2))
    ymin = int(max(0, cy - side/2))
    xmax = int(min(W, xmin + side))
    ymax = int(min(H, ymin + side))
    # 保证严格正方
    side = min(xmax - xmin, ymax - ymin)
    xmax = xmin + side
    ymax = ymin + side
    return xmin, ymin, xmax, ymax

def compute_mouth_roi(
    lms: np.ndarray,
    img_shape,
    # ---- mouth(FA/PFLD) 的原有参数 ----
    fa_enlarge: float = 2.5,
    fa_min_face_ratio: float = 0.42,
    fa_bias: float = 0.02,
    pfld_expand: float = 0.10,
    # ---- 新增：lowerface 用的参数（仅 FA 生效）----
    roi: str = "mouth",           # 'mouth' | 'lowerface' | 'face'
    ear_inset_ratio: float = 0.07,  # 左右向里收一点，避免真耳朵
    top_ratio: float = 0.15,        # 上边界 = 鼻基部往上 top_ratio*脸高
    bottom_ratio: float = 0.03,     # 下边界 = 下巴再往下 bottom_ratio*脸高
    lf_enlarge: float = 1.2,       # lowerface 框整体再放大
):
    if lms is None:
        raise ValueError("`lms` must be provided")
    lms = np.asarray(lms)
    if lms.ndim != 2 or lms.shape[1] < 2:
        raise ValueError(f"Unexpected landmark shape: {lms.shape}")

    H, W = img_shape[:2]
    n = lms.shape[0]

    # ---------- FA 68 点：根据 roi 选择 ----------
    if 68 <= n <= 70:
        # 基础脸宽/脸高估计
        face_w = abs(lms[16, 0] - lms[0, 0])
        brow_y = np.min(lms[17:27, 1])      # 眉毛上沿
        chin_y = lms[8, 1]                  # 下巴
        face_h = max(1.0, chin_y - brow_y)

        if roi == "lowerface":
            # 左右：以内切“耳前区”为目标，用0/16并内缩一点
            x_left  = lms[0, 0]  + ear_inset_ratio * face_w
            x_right = lms[16, 0] - ear_inset_ratio * face_w
            # 上：鼻基部（点33）再往上；下：下巴再往下
            nose_base_y = lms[33, 1]
            y_top = nose_base_y - top_ratio * face_h
            y_bot = chin_y     + bottom_ratio * face_h

            # 做成正方形，并整体放大 lf_enlarge
            cx = (x_left + x_right) * 0.5
            cy = (y_top + y_bot) * 0.5
            side = max(x_right - x_left, y_bot - y_top) * lf_enlarge
            return _clip_square(cx, cy, side, W, H)

        if roi == "face":
            # 整脸可视：用 0..16 的宽 + 眉到下巴的高
            cx = (lms[0,0] + lms[16,0]) * 0.5
            cy = (brow_y + chin_y) * 0.5
            side = max(face_w, face_h) * 1.05  # 稍微放大一点
            return _clip_square(cx, cy, side, W, H)

        # 默认 mouth：原有逻辑，围嘴 48..67
        idx = np.arange(48, 68)
        xs, ys = lms[idx, 0], lms[idx, 1]
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        side = max(max(w, h) * fa_enlarge, face_w * fa_min_face_ratio)
        cx = xs.mean() - fa_bias * side
        cy = ys.mean() - fa_bias * side
        return _clip_square(cx, cy, side, W, H)

    # ---------- PFLD 等：沿用旧逻辑 ----------
    xmin = int(lms[1][0]); ymin = int(lms[52][1]); xmax = int(lms[31][0])
    width = xmax - xmin
    expand = int(width * pfld_expand)
    xmin = max(0, xmin - expand); xmax = min(W, xmax + expand); ymin = max(0, ymin - expand)
    width = xmax - xmin; ymax = min(H, ymin + width)
    return xmin, ymin, xmax, ymax
