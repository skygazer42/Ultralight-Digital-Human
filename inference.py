import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
from smooth_filter import LandmarkSmoother
import time

parser = argparse.ArgumentParser(description='Inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="hubert")
parser.add_argument('--dataset', type=str, default="")
parser.add_argument('--audio_feat', type=str, default="")
parser.add_argument('--save_path', type=str, default="")
parser.add_argument('--checkpoint', type=str, default="")
# 新增优化选项
parser.add_argument('--use_smooth', action='store_true', help="使用关键点平滑")
parser.add_argument('--use_soft_mask', action='store_true', help="使用软边缘融合")
parser.add_argument('--use_seamless', action='store_true', help="使用seamlessClone")
args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
dataset_dir = args.dataset
audio_feat_path = args.audio_feat
mode = args.asr

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_soft_mask(shape, inner_ratio=0.7):
    """创建软边缘掩码"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # 创建椭圆形掩码
    center = (w // 2, h // 2)
    axes = (int(w * inner_ratio / 2), int(h * inner_ratio / 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # 高斯模糊创建软边缘
    kernel_size = 21
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return mask


def get_audio_features(features, index):
    left = index - 4
    right = index + 4
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
    return auds


# 加载数据
audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir + "0.jpg")
H, W = exm_img.shape[:2]  # 使用大写避免变量冲突

# 初始化关键点平滑器
fps = 25 if mode == "hubert" else 20
smoother = LandmarkSmoother(num_points=68, fps=fps) if args.use_smooth else None

# 视频写入器
if mode == "hubert":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (W, H))
else:
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (W, H))

step_stride = 0
img_idx = 0

# 加载模型
net = Model(6, mode).to(device)
net.load_state_dict(torch.load(checkpoint, map_location=device))
net.eval()

print(f"Processing {audio_feats.shape[0]} frames...")
for i in tqdm(range(audio_feats.shape[0])):
    if img_idx > len_img - 1:
        step_stride = -1
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride

    img_path = img_dir + str(img_idx) + '.jpg'
    lms_path = lms_dir + str(img_idx) + '.lms'

    img = cv2.imread(img_path)

    # 读取关键点
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.float32)

    # 应用关键点平滑
    if smoother is not None:
        lms = smoother.smooth(lms)
    else:
        lms = lms.astype(np.int32)

    # 计算裁剪区域（稍微扩大避免边缘问题）
    xmin = lms[1][0]
    ymin = lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin

    # 扩大裁剪区域10%
    expand = int(width * 0.1)
    xmin = max(0, xmin - expand)
    xmax = min(img.shape[1], xmax + expand)
    ymin = max(0, ymin - expand)
    width = xmax - xmin
    ymax = min(img.shape[0], ymin + width)

    # 裁剪
    crop_img = img[ymin:ymax, xmin:xmax]
    crop_h, crop_w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()

    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 150, 145), (0, 0, 0), -1)

    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0).to(device)
    img_masked_T = torch.from_numpy(img_masked / 255.0).to(device)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]

    # 音频特征
    audio_feat = get_audio_features(audio_feats, i)
    if mode == "hubert":
        audio_feat = audio_feat.reshape(16, 32, 32)
    else:
        audio_feat = audio_feat.reshape(128, 16, 32)
    audio_feat = audio_feat[None].to(device)

    # 推理
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]

    pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
    pred = np.array(pred, dtype=np.uint8)

    # 应用软边缘融合
    if args.use_soft_mask:
        soft_mask = create_soft_mask(pred.shape)
        soft_mask = soft_mask[..., np.newaxis]  # 添加通道维度
        pred = (pred.astype(np.float32) * soft_mask +
                img_real_ex_ori.astype(np.float32) * (1 - soft_mask)).astype(np.uint8)

    crop_img_ori[4:164, 4:164] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (crop_w, crop_h))

    # 使用seamlessClone融合
    if args.use_seamless:
        try:
            mask = np.ones((crop_h, crop_w), dtype=np.uint8) * 255
            center = (xmin + crop_w // 2, ymin + crop_h // 2)
            img = cv2.seamlessClone(crop_img_ori, img, mask, center, cv2.NORMAL_CLONE)
        except:
            # 如果seamlessClone失败，使用普通粘贴
            img[ymin:ymax, xmin:xmax] = crop_img_ori
    else:
        # 普通粘贴（可选软边缘）
        if args.use_soft_mask:
            # 为整个裁剪区域创建软掩码
            crop_mask = create_soft_mask((crop_h, crop_w), inner_ratio=0.9)
            crop_mask = crop_mask[..., np.newaxis]
            blend = (crop_img_ori.astype(np.float32) * crop_mask +
                     img[ymin:ymax, xmin:xmax].astype(np.float32) * (1 - crop_mask))
            img[ymin:ymax, xmin:xmax] = blend.astype(np.uint8)
        else:
            img[ymin:ymax, xmin:xmax] = crop_img_ori

    video_writer.write(img)

video_writer.release()
print(f"Video saved to {save_path}")

# 合并音频
wav_path = os.path.join(dataset_dir, 'aud.wav')
if os.path.exists(wav_path):
    output_with_audio = save_path.replace('.mp4', '_with_audio.mp4')
    cmd = f'ffmpeg -i {save_path} -i {wav_path} -c:v copy -c:a aac {output_with_audio} -y'
    os.system(cmd)
    print(f"Video with audio saved to {output_with_audio}")