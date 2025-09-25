# fa_landmark.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from detect_face import SCRFD  # 你的仓库里已有

class Landmark:
    """SCRFD ROI + face_alignment landmarks，签名保持: detect(img_path)->(pts,x1,y1)"""

    def __init__(
        self,
        use_cuda: bool = True,
        use_3d: bool = False,
        detector_path: Path | None = None,
        detector_conf: float = 0.1,   # 调整为 0.1，提升召回
        detector_nms: float = 0.4,
        margin: float = 0.2,
    ) -> None:
        try:
            import face_alignment
            from face_alignment import LandmarksType
        except Exception as exc:
            raise ImportError("请先: pip install face-alignment") from exc

        device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        self.fa = face_alignment.FaceAlignment(
            LandmarksType.THREE_D if use_3d else LandmarksType.TWO_D,
            flip_input=False,
            device=device,
        )

        # 默认先找 data_utils/ 下的 onnx；再尝试 resources/ 等备用位置
        if detector_path is None:
            root = Path(__file__).resolve().parent
            candidates = [
                root / "scrfd_2.5g_kps.onnx",
                root.parent / "resources" / "scrfd_2.5g_kps.onnx",
                root.parent / "resources" / "onnx" / "scrfd_2.5g_kps.onnx",
            ]
            for cand in candidates:
                if cand.exists():
                    detector_path = cand
                    break
        if detector_path is None or not Path(detector_path).exists():
            raise FileNotFoundError("未找到 scrfd_2.5g_kps.onnx，请检查路径")

        self.detector = SCRFD(str(detector_path), confThreshold=detector_conf, nmsThreshold=detector_nms)
        self.margin = float(margin)

    @staticmethod
    def _largest_face(indices, bboxes) -> int | None:
        if indices is None or len(indices) == 0:
            return None
        best_idx, best_area = None, -1.0
        for idx in indices:
            i = int(idx[0] if isinstance(idx, (tuple, list, np.ndarray)) else idx)
            x, y, w, h = bboxes[i]
            area = float(w * h)
            if area > best_area:
                best_area = area
                best_idx = i
        return best_idx

    def _crop_with_margin(self, img: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, int, int]:
        x, y, w, h = box.astype(np.int32)
        x2 = x + w
        y2 = y + h
        pad_x = int(w * self.margin)
        pad_y = int(h * self.margin)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x2 + pad_x)
        y2 = min(img.shape[0], y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return img, 0, 0
        return img[y1:y2, x1:x2], x1, y1

    def detect(self, img_path: str) -> Tuple[np.ndarray, int, int]:
        """返回 (局部 ROI 下的关键点, x_offset, y_offset)。若整帧FA回退则 offset=0,0。"""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"read image failed: {img_path}")

        # 1) 先用 SCRFD 找人脸 (x,y,w,h)
        bboxes, indices, _ = self.detector.detect(img_bgr)
        best_idx = self._largest_face(indices, bboxes)

        # 2) 找不到就整帧跑 FA（尽量避免返回空）
        if best_idx is None:
            preds = self.fa.get_landmarks(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not preds:
                return np.zeros((0, 2), dtype=np.int32), 0, 0
            pts = self._select_largest_pred(preds)
            return pts.astype(np.int32), 0, 0

        # 3) 裁 ROI + margin，再跑 FA
        roi, x1, y1 = self._crop_with_margin(img_bgr, bboxes[best_idx])
        if roi.size == 0:
            return np.zeros((0, 2), dtype=np.int32), 0, 0

        preds = self.fa.get_landmarks(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if not preds:
            return np.zeros((0, 2), dtype=np.int32), 0, 0

        pts = preds[0][:, :2]
        h_roi, w_roi = roi.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w_roi - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_roi - 1)
        return pts.astype(np.int32), int(x1), int(y1)

    @staticmethod
    def _select_largest_pred(preds) -> np.ndarray:
        best = None
        best_area = -1.0
        for arr in preds:
            pts = arr[:, :2]
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
            area = float((x_max - x_min) * (y_max - y_min))
            if area > best_area:
                best_area = area
                best = pts
        return best if best is not None else np.zeros((0, 2), dtype=np.float32)
