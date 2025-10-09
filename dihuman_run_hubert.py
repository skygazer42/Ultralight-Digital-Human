# -*- coding: utf-8 -*-
"""Real-time runner that uses HuBERT audio features."""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Tuple

import cv2
from mouth_roi import compute_mouth_roi
import numpy as np
import onnxruntime
import torch
from transformers import HubertModel, Wav2Vec2Processor

from smooth_filter import LandmarkSmoother

logger = logging.getLogger(__name__)


def _load_hubert(device: torch.device) -> Tuple[Wav2Vec2Processor, HubertModel]:
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    model = model.to(device)
    model.eval()
    return processor, model


def _make_even_first_dim(tensor: torch.Tensor) -> torch.Tensor:
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor


def _create_soft_mask(shape, inner_ratio=0.7):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * inner_ratio / 2), int(h * inner_ratio / 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


class DiHumanProcessorHubert:
    """HuBERT-based processor mirroring the Wenet streaming API."""

    def __init__(self, data_path: str, fps: int = 25, use_smooth: bool = False, use_soft_mask: bool = False, use_seamless: bool = False):
        self.data_path = data_path
        self.fps = fps
        self.use_smooth = use_smooth
        self.use_soft_mask = use_soft_mask
        self.use_seamless = use_seamless

        self.full_body_img_dir = os.path.join(data_path, "img_inference")
        self.lms_dir = os.path.join(data_path, "lms_inference")

        self.full_body_img_list: List[np.ndarray] = []
        self.bbox_list: List[List[int]] = []
        self.landmarks_list: List[np.ndarray] = []
        for i in range(max(0, len(os.listdir(self.lms_dir)) - 1)):
            full_body_img = cv2.imread(os.path.join(self.full_body_img_dir, f"{i}.jpg"))
            if full_body_img is None:
                continue
            self.full_body_img_list.append(full_body_img)
            lms_path = os.path.join(self.lms_dir, f"{i}.lms")
            lms_list = []
            with open(lms_path, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    if not line.strip():
                        continue
                    lms_list.append(np.array(line.split(" "), dtype=np.float32))
            lms = np.array(lms_list, dtype=np.float32)
            self.landmarks_list.append(lms)
            xmin, ymin, xmax, ymax = compute_mouth_roi(lms, full_body_img.shape)
            self.bbox_list.append([xmin, ymin, xmax, ymax])

        if not self.full_body_img_list or not self.bbox_list:
            raise RuntimeError("Missing inference images or landmarks under data_path.")

        self.num_points = int(self.landmarks_list[0].shape[0])
        self.smoother = (
            LandmarkSmoother(num_points=self.num_points, fps=self.fps)
            if self.use_smooth else None
        )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_unet_session = onnxruntime.InferenceSession(
            os.path.join(data_path, "unet.onnx"), providers=providers
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor, self.hubert_model = _load_hubert(self.device)

        self.frames_per_window = 8
        self.chunk_samples = 320 * self.frames_per_window  # 20ms stride * frames
        self.overlap_samples = 80  # kernel(400) - stride(320)
        self.max_feature_queue = 32

        self.sample_buffer = np.zeros((0,), dtype=np.float32)
        self.feature_window = np.zeros((0, 2, 1024), dtype=np.float32)

        self.audio_play_list: List[int] = [0] * 13440

        self.index = 0
        self.step = 1
        self.counter = 0
        self.empty_audio_counter = 56
        self.is_processing = False
        self.return_img: Optional[np.ndarray] = None
        self.silence = True

    def reset(self) -> None:
        self.sample_buffer = np.zeros((0,), dtype=np.float32)
        self.feature_window = np.zeros((0, 2, 1024), dtype=np.float32)
        self.audio_play_list = [0] * 13440
        self.counter = 0
        self.is_processing = True
        if self.use_smooth:
            self.smoother = LandmarkSmoother(num_points=self.num_points, fps=self.fps)

    def _compute_hubert_features(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size == 0:
            return np.zeros((0, 2, 1024), dtype=np.float32)
        inputs = self.processor(
            waveform, return_tensors="pt", sampling_rate=16000, padding=False
        ).input_values.to(self.device)
        with torch.no_grad():
            hidden_states = self.hubert_model(inputs).last_hidden_state[0]  # [T,1024]
        hidden_states = _make_even_first_dim(hidden_states)
        if hidden_states.numel() == 0:
            return np.zeros((0, 2, 1024), dtype=np.float32)
        features = hidden_states.reshape(-1, 2, 1024).cpu().numpy().astype(np.float32)
        return features

    def _update_hubert_features(self) -> None:
        if self.sample_buffer.shape[0] < self.chunk_samples:
            return
        waveform = self.sample_buffer / 32768.0
        features = self._compute_hubert_features(waveform)
        if features.size:
            if self.feature_window.size:
                self.feature_window = np.concatenate([self.feature_window, features], axis=0)
            else:
                self.feature_window = features
            if self.feature_window.shape[0] > self.max_feature_queue:
                self.feature_window = self.feature_window[-self.max_feature_queue :]
        # keep small overlap to preserve context
        if self.sample_buffer.shape[0] > self.overlap_samples:
            self.sample_buffer = self.sample_buffer[-self.overlap_samples :]
        else:
            self.sample_buffer = np.zeros((0,), dtype=np.float32)

    def _compute_bbox(self, frame_idx: int, img: np.ndarray) -> List[int]:
        frame_idx = max(0, min(frame_idx, len(self.landmarks_list) - 1))
        landmarks = self.landmarks_list[frame_idx]
        if self.use_smooth and self.smoother is not None:
            landmarks = self.smoother.smooth(landmarks.astype(np.float32))
        xmin, ymin, xmax, ymax = compute_mouth_roi(landmarks, img.shape)
        bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
        self.bbox_list[frame_idx] = bbox
        return bbox

    def process(self, audio_frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, int]:
        audio_frame = audio_frame.astype(np.int16)
        if not np.any(audio_frame):
            if not self.silence:
                self.empty_audio_counter += 1
            if self.empty_audio_counter >= 100:
                self.silence = True
        else:
            self.empty_audio_counter = 0
            self.silence = False

        if self.silence:
            self.sample_buffer = np.zeros((0,), dtype=np.float32)
            self.feature_window = np.zeros((0, 2, 1024), dtype=np.float32)
            self.is_processing = False
            if self.counter == 0:
                return_img = self.full_body_img_list[self.index].copy()
                self.index += self.step
                if self.index >= len(self.bbox_list) - 1:
                    self.step = -1
                elif self.index <= 0:
                    self.step = 1
                check_img = 1
                self.counter += 1
            else:
                return_img = None
                check_img = 0
                self.counter += 1
                if self.counter == 5:
                    self.counter = 0
            playing_audio = np.zeros((160,), dtype=np.int16)
            return return_img, playing_audio, check_img

        if not self.is_processing:
            self.reset()

        if audio_frame.shape[0] < 160:
            audio_frame = np.pad(audio_frame, (0, 160 - audio_frame.shape[0]))

        self.audio_play_list.extend(audio_frame.tolist())
        self.sample_buffer = np.concatenate([
            self.sample_buffer,
            audio_frame.astype(np.float32),
        ])
        self._update_hubert_features()

        return_img = None
        check_img = 0

        if self.feature_window.shape[0] >= self.frames_per_window:
            img = self.full_body_img_list[self.index].copy()
            bbox = self._compute_bbox(self.index, img)
            self.index += self.step
            if self.index >= len(self.bbox_list) - 1:
                self.step = -1
            elif self.index <= 0:
                self.step = 1

            xmin, ymin, xmax, ymax = bbox
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w = crop_img.shape[:2]
            crop_img = cv2.resize(crop_img, (168, 168))
            crop_img_ori = crop_img.copy()
            roi_original = img[ymin:ymax, xmin:xmax].copy()

            img_real_ex = crop_img[4:164, 4:164].copy()
            img_real_ex_ori = img_real_ex.copy()
            img_masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 150, 145), (0, 0, 0), -1)

            img_masked = img_masked.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0

            img_onnx_in = np.concatenate((img_real_ex[None], img_masked[None]), axis=1)

            window = self.feature_window[: self.frames_per_window]
            audio_feat = window.reshape(1, 16, 32, 32)

            ort_inputs = {
                self.ort_unet_session.get_inputs()[0].name: img_onnx_in,
                self.ort_unet_session.get_inputs()[1].name: audio_feat,
            }
            ort_outs = self.ort_unet_session.run(None, ort_inputs)

            pred = ort_outs[0][0]
            pred = pred.transpose(1, 2, 0) * 255.0
            pred = pred.astype(np.uint8)

            if self.use_soft_mask:
                soft_mask = _create_soft_mask(pred.shape)
                pred = (pred.astype(np.float32) * soft_mask[..., None] +
                        img_real_ex_ori.astype(np.float32) * (1 - soft_mask[..., None])).astype(np.uint8)

            crop_img_ori[4:164, 4:164] = pred
            crop_img_ori = cv2.resize(crop_img_ori, (w, h))

            if self.use_seamless:
                try:
                    mask = np.ones((h, w), dtype=np.uint8) * 255
                    center = (xmin + w // 2, ymin + h // 2)
                    img = cv2.seamlessClone(crop_img_ori, img, mask, center, cv2.NORMAL_CLONE)
                except Exception:
                    if self.use_soft_mask:
                        crop_mask = _create_soft_mask((h, w), inner_ratio=0.9)[..., None]
                        blend = (crop_img_ori.astype(np.float32) * crop_mask +
                                 roi_original.astype(np.float32) * (1 - crop_mask))
                        img[ymin:ymax, xmin:xmax] = blend.astype(np.uint8)
                    else:
                        img[ymin:ymax, xmin:xmax] = crop_img_ori
            else:
                if self.use_soft_mask:
                    crop_mask = _create_soft_mask((h, w), inner_ratio=0.9)[..., None]
                    blend = (crop_img_ori.astype(np.float32) * crop_mask +
                             roi_original.astype(np.float32) * (1 - crop_mask))
                    img[ymin:ymax, xmin:xmax] = blend.astype(np.uint8)
                else:
                    img[ymin:ymax, xmin:xmax] = crop_img_ori

            self.feature_window = self.feature_window[1:]

            return_img = img.copy()
            check_img = 1

        if self.audio_play_list:
            playing_audio = np.array(self.audio_play_list[:160], dtype=np.int16)
            self.audio_play_list = self.audio_play_list[160:]
        else:
            playing_audio = np.zeros((160,), dtype=np.int16)

        return return_img, playing_audio, check_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    processor = DiHumanProcessorHubert(args.data_path)
    logger.info("HuBERT processor ready. This script is intended for server integration.")
