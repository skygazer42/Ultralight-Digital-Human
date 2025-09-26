# mouth_roi.py
from __future__ import annotations
import numpy as np

# ============== 小工具 ==============

def _clip_square(cx: float, cy: float, side: float, W: int, H: int):
    """把 (cx,cy,side) 限制在图像内，并强制为正方形（返回 int 坐标）"""
    xmin = int(max(0, cx - side / 2))
    ymin = int(max(0, cy - side / 2))
    xmax = int(min(W, xmin + side))
    ymax = int(min(H, ymin + side))
    side = min(xmax - xmin, ymax - ymin)  # 严格正方
    xmax = xmin + side
    ymax = ymin + side
    return xmin, ymin, xmax, ymax

# ============== 轻量稳框器 ==============

# 每路视频独立状态：key -> (cx, cy, side)
_STAB_STATE: dict[str, tuple[float, float, float]] = {}

def reset_stabilizer(key: str = "default"):
    """切到新视频时可调用，清空上一段的稳框状态"""
    _STAB_STATE.pop(key, None)

def _stabilize_bbox(
    cx: float,
    cy: float,
    side: float,
    ref_scale: float,
    key: str = "default",
    ema: float = 0.55,            # 0.5~0.6：2~3 帧即可稳定
    max_move_ratio: float = 0.03, # 单帧中心最大位移（相对 ref_scale）
    max_scale_ratio: float = 0.06 # 单帧尺度最大变化（相对上一帧）
):
    """EMA + 位移/尺度限幅，抑制抖动与漂移"""
    prev = _STAB_STATE.get(key)
    if prev is None:
        _STAB_STATE[key] = (cx, cy, side)
        return cx, cy, side

    px, py, ps = prev
    # 1) EMA
    sx = ema * cx + (1.0 - ema) * px
    sy = ema * cy + (1.0 - ema) * py
    ss = ema * side + (1.0 - ema) * ps
    # 2) 位移限幅（像素）
    max_move = max(1.0, max_move_ratio * ref_scale)
    sx = px + np.clip(sx - px, -max_move, max_move)
    sy = py + np.clip(sy - py, -max_move, max_move)
    # 3) 尺度限幅
    scale = np.clip(ss / ps, 1.0 - max_scale_ratio, 1.0 + max_scale_ratio)
    ss = ps * scale

    _STAB_STATE[key] = (sx, sy, ss)
    return sx, sy, ss

# ============== 主函数 ==============

def compute_mouth_roi(
    lms: np.ndarray,
    img_shape,
    # 旧 mouth 参数（保留兼容）
    fa_enlarge: float = 1.6,
    fa_min_face_ratio: float = 0.42,
    fa_bias: float = 0.02,
    pfld_expand: float = 0.10,
    # 新：更安全的“下半脸”裁剪（**耳朵外切**、**上沿在眼睛下**）
    roi: str = "safe_lowerface",   # 'mouth' | 'lowerface' | 'safe_lowerface' | 'face'
    ear_outset_ratio: float = 0.06, # 左右向外再扩一点(相对脸宽)，把真耳朵也包进来
    eye_margin_ratio: float = 0.02, # 顶部 = 眼睛下沿 + 余量(相对脸高)
    chin_extra_ratio: float = 0.06, # 底部 = 下巴 + 余量(相对脸高)
    safe_enlarge: float = 1.00,     # 额外整体放大（一般 1.0 即可）
    # 稳定
    stabilize: bool = True,
    stab_key: str = "default",
    ema: float = 0.55,
    max_move_ratio: float = 0.03,
    max_scale_ratio: float = 0.06,
):
    """
    依据关键点生成 ROI（正方形）。
    - FA(68点)：
        * 'safe_lowerface'：上沿在“眼睛下沿+余量”，左右**外切到耳朵**并外扩，底部自动给到下巴以下，适合避免碰眼眉且给足嘴/下巴空间。
        * 'lowerface' / 'mouth' / 'face'：兼容旧逻辑。
    - PFLD：沿用历史 3 点规则，可选稳定。
    """
    if lms is None:
        raise ValueError("`lms` must be provided")
    lms = np.asarray(lms)
    if lms.ndim != 2 or lms.shape[1] < 2:
        raise ValueError(f"Unexpected landmark shape: {lms.shape}")

    H, W = img_shape[:2]
    n = lms.shape[0]

    # ------- FA 68 点 -------
    if 68 <= n <= 70:
        # 基本脸尺度
        x0, x16 = float(lms[0, 0]), float(lms[16, 0])
        face_w = abs(x16 - x0)
        brow_y = float(np.min(lms[17:27, 1]))   # 眉上沿
        chin_y = float(lms[8, 1])               # 下巴
        face_h = max(1.0, chin_y - brow_y)

        if roi == "safe_lowerface":
            # 1) 左右 **外切** 到耳朵并外扩
            x_left  = x0  - ear_outset_ratio * face_w
            x_right = x16 + ear_outset_ratio * face_w

            # 2) 顶部 = “两眼下沿” + 余量（确保不碰眼睛/眉毛）
            eye_idx = np.r_[36:48]
            eye_y = lms[eye_idx, 1]
            eye_lower = float(np.percentile(eye_y, 75))  # 稳健下沿
            y_top = eye_lower + eye_margin_ratio * face_h

            # 3) 底部 = 下巴 + 余量（你说“下面不用管”，这里保证能覆盖到下巴以下）
            y_bot = chin_y + chin_extra_ratio * face_h

            # 4) 取正方形：宽 = x_right-x_left，高 = y_bot-y_top
            width_need  = x_right - x_left
            height_need = y_bot - y_top
            side = max(width_need, height_need) * safe_enlarge

            # 让上沿尽量贴近 y_top：先以上沿为锚点再求中心
            cx = (x_left + x_right) * 0.5
            cy = y_top + side * 0.5

            if stabilize:
                ref = max(face_w, face_h)
                cx, cy, side = _stabilize_bbox(
                    cx, cy, side, ref, key=stab_key,
                    ema=ema, max_move_ratio=max_move_ratio, max_scale_ratio=max_scale_ratio
                )
            return _clip_square(cx, cy, side, W, H)

        if roi == "face":
            cx = (x0 + x16) * 0.5
            cy = (brow_y + chin_y) * 0.5
            side = max(face_w, face_h) * 1.05
            if stabilize:
                ref = max(face_w, face_h)
                cx, cy, side = _stabilize_bbox(
                    cx, cy, side, ref, key=stab_key,
                    ema=ema, max_move_ratio=max_move_ratio, max_scale_ratio=max_scale_ratio
                )
            return _clip_square(cx, cy, side, W, H)

        if roi == "lowerface":
            # 旧版（可能略碰眼），保留兼容
            x_left  = x0  + 0.03 * face_w
            x_right = x16 - 0.03 * face_w
            nose_base_y = float(lms[33, 1])
            y_top = nose_base_y - 0.10 * face_h
            y_bot = chin_y + 0.10 * face_h
            cx = (x_left + x_right) * 0.5
            cy = (y_top + y_bot) * 0.5
            side = max(x_right - x_left, y_bot - y_top) * 1.10
            if stabilize:
                ref = max(face_w, face_h)
                cx, cy, side = _stabilize_bbox(
                    cx, cy, side, ref, key=stab_key,
                    ema=ema, max_move_ratio=max_move_ratio, max_scale_ratio=max_scale_ratio
                )
            return _clip_square(cx, cy, side, W, H)

        # 纯 mouth（48..67），保留
        idx = np.arange(48, 68)
        xs, ys = lms[idx, 0], lms[idx, 1]
        w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        side = max(max(w, h) * fa_enlarge, face_w * fa_min_face_ratio)
        cx = float(xs.mean()) - fa_bias * side
        cy = float(ys.mean()) - fa_bias * side
        if stabilize:
            ref = max(face_w, face_h)
            cx, cy, side = _stabilize_bbox(
                cx, cy, side, ref, key=stab_key,
                ema=ema, max_move_ratio=max_move_ratio, max_scale_ratio=max_scale_ratio
            )
        return _clip_square(cx, cy, side, W, H)

    # ------- PFLD 等老布局 -------
    xmin = int(lms[1][0]); ymin = int(lms[52][1]); xmax = int(lms[31][0])
    width = xmax - xmin
    expand = int(width * pfld_expand)
    xmin = max(0, xmin - expand); xmax = min(W, xmax + expand); ymin = max(0, ymin - expand)
    width = xmax - xmin; ymax = min(H, ymin + width)

    if stabilize:
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        side = float(max(1, xmax - xmin))
        cx, cy, side = _stabilize_bbox(
            cx, cy, side, ref_scale=side, key=stab_key,
            ema=ema, max_move_ratio=max_move_ratio, max_scale_ratio=max_scale_ratio
        )
        xmin, ymin, xmax, ymax = _clip_square(cx, cy, side, W, H)

    return xmin, ymin, xmax, ymax
