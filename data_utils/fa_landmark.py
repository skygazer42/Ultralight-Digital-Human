# fa_landmark.py
# pip install face-alignment
import cv2, os, numpy as np, torch

class Landmark:
    def __init__(self, use_cuda=True, use_3d=False):
        try:
            import face_alignment
            from face_alignment import LandmarksType
        except Exception as e:
            raise ImportError("请先: pip install face-alignment") from e
        device = 'cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu'
        self.fa = face_alignment.FaceAlignment(
            LandmarksType._3D if use_3d else LandmarksType._2D,
            flip_input=False, device=device
        )

    def detect(self, img_path):
        """
        与 get_landmark.Landmark.detect 保持一致的签名:
        return pre_landmark(np.ndarray[N,2]，int), x1(int), y1(int)

        这里返回的 pre_landmark 是图像上的“绝对坐标”，所以 x1=y1=0。
        process.py 里会把每个点写成 x+x1, y+y1（相当于原样写入）。
        """
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"read image failed: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preds = self.fa.get_landmarks(img_rgb)  # list of [68,2] 或 [68,3]
        if not preds:
            # 找不到脸时，返回空；上层可选择跳过或复用上一帧
            return np.zeros((0, 2), dtype=np.int32), 0, 0

        # 多人时选最大脸
        def area(pts2d):
            x1, y1 = pts2d[:,0].min(), pts2d[:,1].min()
            x2, y2 = pts2d[:,0].max(), pts2d[:,1].max()
            return (x2-x1)*(y2-y1)

        pts = max(preds, key=lambda p: area(p[:, :2]))[:, :2]  # [68,2]
        pts = pts.astype(np.int32)
        return pts, 0, 0
