import os
import cv2
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset

from mouth_roi import compute_mouth_roi

class MyDataset(Dataset):
    """
    目录结构：
      <img_dir>/
        full_body_img/0.jpg, 1.jpg, ...
        landmarks/0.lms, 1.lms, ...
        aud_wenet.npy 或 aud_hu.npy
    """

    def __init__(self, img_dir, mode):
        self.mode = mode  # "wenet" or "hubert"
        self.img_path_list, self.lms_path_list = [], []

        full_img_dir = os.path.join(img_dir, "full_body_img")
        lms_dir = os.path.join(img_dir, "landmarks")

        num_items = len(os.listdir(full_img_dir))
        for i in range(num_items):
            img_path = os.path.join(full_img_dir, f"{i}.jpg")
            lms_path = os.path.join(lms_dir, f"{i}.lms")
            if os.path.isfile(img_path) and os.path.isfile(lms_path):
                self.img_path_list.append(img_path)
                self.lms_path_list.append(lms_path)

        if self.mode == "wenet":
            npy_path = os.path.join(img_dir, "aud_wenet.npy")
        elif self.mode == "hubert":
            npy_path = os.path.join(img_dir, "aud_hu.npy")
        else:
            raise ValueError(f"mode 必须为 'wenet' 或 'hubert'，当前为：{self.mode}")

        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"音频特征不存在: {npy_path}")

        self.audio_feats = np.load(npy_path).astype(np.float32)
        if self.audio_feats.ndim < 2:
            self.audio_feats = np.atleast_2d(self.audio_feats).astype(np.float32)

        if len(self.img_path_list) == 0:
            raise RuntimeError("没有可用的图像/关键点对，请检查数据目录。")

    def __len__(self):
        n_audio = int(self.audio_feats.shape[0])
        n_image = int(len(self.img_path_list))
        return max(0, min(n_audio, n_image))

    def get_audio_window(self, features, index, frames):
        """从 index 周围取 frames 帧（左右对称），不够则零填充。返回 torch.Tensor[L, ...]"""
        half = frames // 2
        left = index - half
        right = index + half  # 右端开区间

        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]

        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
        return auds  # [frames, ...]

    @staticmethod
    def _read_lms(lms_path):
        lms_list = []
        with open(lms_path, "r") as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                lms_list.append(np.array(line.split(" "), dtype=np.float32))
        return np.array(lms_list, dtype=np.int32)

    @staticmethod
    def _crop_face_like(img, lms):
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = compute_mouth_roi(lms, img.shape)
        if xmax <= xmin or ymax <= ymin:
            side = min(h, w); ox = (w - side) // 2; oy = (h - side) // 2
            crop = img[oy:oy+side, ox:ox+side]
        else:
            crop = img[ymin:ymax, xmin:xmax]
        crop = cv2.resize(crop, (168, 168), cv2.INTER_AREA)
        return crop[4:164, 4:164].copy()  # (160,160,3)

    def process_img(self, img, lms_path, img_ex, lms_path_ex):
        lms = self._read_lms(lms_path)
        lms_ex = self._read_lms(lms_path_ex)
        img_real = self._crop_face_like(img, lms)
        img_ref  = self._crop_face_like(img_ex, lms_ex)

        img_masked = img_real.copy()
        cv2.rectangle(img_masked, (5, 5), (150, 145), (0, 0, 0), -1)

        img_real_T   = torch.from_numpy(img_real.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_masked_T = torch.from_numpy(img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_ref_T    = torch.from_numpy(img_ref.transpose(2, 0, 1).astype(np.float32) / 255.0)

        img_concat_T = torch.cat([img_ref_T, img_masked_T], dim=0)  # C=6
        return img_concat_T, img_real_T

    def __getitem__(self, idx):
        n = len(self)
        if n == 0:
            raise RuntimeError("空数据集")
        idx = int(idx) % n

        img = cv2.imread(self.img_path_list[idx])
        if img is None:
            raise FileNotFoundError(f"读取图像失败: {self.img_path_list[idx]}")
        lms_path = self.lms_path_list[idx]

        ex_int = random.randint(0, n - 1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        if img_ex is None:
            raise FileNotFoundError(f"读取参考图像失败: {self.img_path_list[ex_int]}")
        lms_path_ex = self.lms_path_list[ex_int]

        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)

        # ====== 音频特征拼装：按 UNet 期望输出形状 ======
        D = int(np.prod(self.audio_feats.shape[1:]))  # 每帧展平后的维度
        if self.mode == "hubert":
            target_shape = (16, 32, 32)      # UNet 的 audio_model.conv1 期望 16 通道
        else:  # wenet
            target_shape = (128, 16, 32)

        target_total = int(np.prod(target_shape))
        frames_needed = max(1, math.ceil(target_total / D))
        auds = self.get_audio_window(self.audio_feats, idx, frames=frames_needed).float()
        flat = auds.reshape(-1)

        if flat.numel() < target_total:
            pad = target_total - flat.numel()
            flat = torch.nn.functional.pad(flat, (0, pad))
        elif flat.numel() > target_total:
            flat = flat[:target_total]

        audio_feat = flat.view(*target_shape)
        # ===============================================

        return img_concat_T, img_real_T, audio_feat
