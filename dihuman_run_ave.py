# -*- coding: utf-8 -*-
"""Real-time runner for the 328 UNet with AVE audio encoder."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime
import torch

from mouth_roi import compute_mouth_roi
from smooth_filter import LandmarkSmoother
from data_utils.ave import audio as ave_audio
from data_utils.ave.models.audioEnc import AudioEncoder

logger = logging.getLogger(__name__)

SYNCNET_MEL_STEP_SIZE = 16
HOP_SIZE = 200  # samples (12.5 ms @ 16 kHz)
WIN_SIZE = 800  # samples (50 ms @ 16 kHz)
SAMPLE_RATE = 16000
MEL_WINDOW_SAMPLES = WIN_SIZE + HOP_SIZE * (SYNCNET_MEL_STEP_SIZE - 1)
FPS = 25
FUTURE_CONTEXT = 8  # frames on each side used to build UNet audio tensor


def _create_soft_mask(shape, inner_ratio=0.7):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * inner_ratio / 2), int(h * inner_ratio / 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


class DiHumanProcessorAve:
    """Streaming processor using the AVE audio encoder and 328 UNet."""

    def __init__(self, data_path: str, fps: int = FPS, use_smooth: bool = False, use_soft_mask: bool = False, use_seamless: bool = False):
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
            lms_list: List[np.ndarray] = []
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = os.path.join("model", "checkpoints", "audio_visual_encoder.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Audio encoder checkpoint not found: {ckpt_path}")

        self.audio_encoder = AudioEncoder().to(self.device).eval()
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = {f"audio_encoder.{k}": v for k, v in ckpt.items()}
        self.audio_encoder.load_state_dict(state_dict)

        preferred_model = os.path.join(data_path, "unet_ave.onnx")
        fallback_model = os.path.join(data_path, "unet.onnx")
        unet_path = preferred_model if os.path.isfile(preferred_model) else fallback_model
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(
                f"Could not locate ONNX model. Expected '{preferred_model}' or '{fallback_model}'."
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_unet_session = onnxruntime.InferenceSession(unet_path, providers=providers)

        self.fps = FPS

        self.sample_buffer = np.zeros(0, dtype=np.float32)
        self.audio_features: List[np.ndarray] = []
        self.feature_template: Optional[np.ndarray] = None
        self.frame_cursor = 0
        self.next_video_index = 0

        self.audio_play_list: List[int] = [0] * 13440
        self.index = 0
        self.step = 1
        self.counter = 0
        self.empty_audio_counter = 56
        self.is_processing = False
        self.silence = True

    def reset(self) -> None:
        self.sample_buffer = np.zeros(0, dtype=np.float32)
        self.audio_features = []
        self.feature_template = None
        self.frame_cursor = 0
        self.next_video_index = 0
        self.audio_play_list = [0] * 13440
        self.counter = 0
        self.is_processing = True
        if self.use_smooth:
            self.smoother = LandmarkSmoother(num_points=self.num_points, fps=self.fps)

    def _append_feature(self, feature: np.ndarray) -> None:
        feature = feature.astype(np.float32)
        if self.feature_template is None:
            self.feature_template = np.zeros_like(feature)
        self.audio_features.append(feature)

    def _build_audio_tensor(self, idx: int) -> np.ndarray:
        if self.feature_template is None:
            template = np.zeros(512, dtype=np.float32)
        else:
            template = self.feature_template

        left = idx - FUTURE_CONTEXT
        right = idx + FUTURE_CONTEXT
        stack = []
        for pos in range(left, right):
            if 0 <= pos < len(self.audio_features):
                stack.append(self.audio_features[pos])
            else:
                stack.append(template)
        stacked = np.stack(stack, axis=0)  # (16, 512)
        return stacked.reshape(1, 32, 16, 16).astype(np.float32)

    def _render_frame(self, audio_tensor: np.ndarray) -> np.ndarray:
        img = self.full_body_img_list[self.index].copy()
        bbox = self._compute_bbox(self.index, img)

        self.index += self.step
        if self.index >= len(self.bbox_list) - 1:
            self.step = -1
        elif self.index <= 0:
            self.step = 1

        xmin, ymin, xmax, ymax = bbox
        crop_img = img[ymin:ymax, xmin:xmax]
        if crop_img.size == 0:
            return img

        crop_h, crop_w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()
        roi_original = img[ymin:ymax, xmin:xmax].copy()
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 310, 305), (0, 0, 0), -1)

        img_real_ex = (img_real_ex.transpose(2, 0, 1) / 255.0).astype(np.float32)
        img_masked = (img_masked.transpose(2, 0, 1) / 255.0).astype(np.float32)
        img_onnx_in = np.concatenate((img_real_ex, img_masked), axis=0)[None]

        ort_inputs = {
            self.ort_unet_session.get_inputs()[0].name: img_onnx_in,
            self.ort_unet_session.get_inputs()[1].name: audio_tensor,
        }
        ort_outs = self.ort_unet_session.run(None, ort_inputs)
        pred = ort_outs[0][0].transpose(1, 2, 0) * 255.0
        pred = pred.astype(np.uint8)

        if self.use_soft_mask:
            soft_mask = _create_soft_mask(pred.shape)
            pred = (pred.astype(np.float32) * soft_mask[..., None] +
                    img_real_ex_ori.astype(np.float32) * (1 - soft_mask[..., None])).astype(np.uint8)

        crop_img_ori[4:324, 4:324] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

        if self.use_seamless:
            try:
                mask = np.ones((crop_h, crop_w), dtype=np.uint8) * 255
                center = (xmin + crop_w // 2, ymin + crop_h // 2)
                img = cv2.seamlessClone(crop_img_ori, img, mask, center, cv2.NORMAL_CLONE)
            except Exception:
                if self.use_soft_mask:
                    crop_mask = _create_soft_mask((crop_h, crop_w), inner_ratio=0.9)[..., None]
                    blend = (crop_img_ori.astype(np.float32) * crop_mask +
                             roi_original.astype(np.float32) * (1 - crop_mask))
                    img[ymin:ymax, xmin:xmax] = blend.astype(np.uint8)
                else:
                    img[ymin:ymax, xmin:xmax] = crop_img_ori
        else:
            if self.use_soft_mask:
                crop_mask = _create_soft_mask((crop_h, crop_w), inner_ratio=0.9)[..., None]
                blend = (crop_img_ori.astype(np.float32) * crop_mask +
                         roi_original.astype(np.float32) * (1 - crop_mask))
                img[ymin:ymax, xmin:xmax] = blend.astype(np.uint8)
            else:
                img[ymin:ymax, xmin:xmax] = crop_img_ori
        return img

    def _update_audio_features(self) -> None:
        while True:
            start_idx = int(80 * (self.frame_cursor / self.fps))
            start_sample = start_idx * HOP_SIZE
            end_sample = start_sample + MEL_WINDOW_SAMPLES

            if end_sample > self.sample_buffer.shape[0]:
                break

            segment = self.sample_buffer[start_sample:end_sample]
            if segment.shape[0] < MEL_WINDOW_SAMPLES:
                segment = np.pad(
                    segment,
                    (0, MEL_WINDOW_SAMPLES - segment.shape[0]),
                    mode="constant",
                )

            mel = ave_audio.melspectrogram(segment).T  # (frames, 80)
            if mel.shape[0] < SYNCNET_MEL_STEP_SIZE:
                pad = SYNCNET_MEL_STEP_SIZE - mel.shape[0]
                mel = np.pad(mel, ((0, pad), (0, 0)), mode="edge")
            else:
                mel = mel[:SYNCNET_MEL_STEP_SIZE]

            mel_tensor = torch.from_numpy(mel.T).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.audio_encoder(mel_tensor).squeeze(0).cpu().numpy()

            self._append_feature(feature)
            self.frame_cursor += 1

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
            self.sample_buffer = np.zeros(0, dtype=np.float32)
            self.audio_features = []
            self.feature_template = None
            self.frame_cursor = 0
            self.next_video_index = 0
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

        float_frame = audio_frame.astype(np.float32) / 32768.0
        self.sample_buffer = np.concatenate((self.sample_buffer, float_frame))
        self._update_audio_features()

        return_img: Optional[np.ndarray] = None
        check_img = 0

        if len(self.audio_features) > FUTURE_CONTEXT:
            available_last_index = len(self.audio_features) - 1
            if self.next_video_index + FUTURE_CONTEXT <= available_last_index:
                audio_tensor = self._build_audio_tensor(self.next_video_index)
                return_img = self._render_frame(audio_tensor)
                check_img = 1
                self.next_video_index += 1
                self.counter = 1
            else:
                if self.counter != 0:
                    self.counter += 1
                    if self.counter == 5:
                        self.counter = 0

        if self.audio_play_list:
            playing_audio = np.array(self.audio_play_list[:160], dtype=np.int16)
            self.audio_play_list = self.audio_play_list[160:]
        else:
            playing_audio = np.zeros((160,), dtype=np.int16)

        return return_img, playing_audio, check_img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AVE real-time processor (debug mode)")
    parser.add_argument("--data_path", type=str, required=True, help="Path containing img/landmark resources and ONNX.")
    args = parser.parse_args()

    processor = DiHumanProcessorAve(args.data_path)
    logger.info("AVE processor ready for streaming integration.")
