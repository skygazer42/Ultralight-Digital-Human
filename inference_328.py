import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from smooth_filter import LandmarkSmoother
from unet_328 import Model

try:
    from utils import AudioEncoder, AudDataset, get_audio_features
except ImportError:  # Fallback for environments without top-level utils module
    from data_utils.ave.models.audioEnc import AudioEncoder
    from data_utils.ave.test_w2l_audio import AudDataset

    def get_audio_features(features, index):
        left = index - 8
        right = index + 8
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Inference (328 variant)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--asr', type=str, default='ave', choices=['hubert', 'wenet', 'ave'])
    parser.add_argument('--name', type=str, default='May')
    parser.add_argument('--audio_path', type=str, default='demo/talk_hb.wav')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--parsing', action='store_true', help='Use semantic parsing masks during compositing.')
    parser.add_argument('--use_smooth', action='store_true', help='Apply landmark smoothing before cropping mouth ROI.')
    parser.add_argument('--smooth_fps', type=int, default=0, help='Optional FPS override for the smoother (0 keeps mode default).')
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_path = os.path.join('./checkpoint', args.name)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint directory not found: {checkpoint_path}')

    ckpt_candidates = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
    if not ckpt_candidates:
        raise FileNotFoundError(f'No .pth checkpoint found under {checkpoint_path}')

    latest_ckpt = sorted(ckpt_candidates, key=lambda x: int(os.path.splitext(x)[0]))[-1]
    checkpoint = os.path.join(checkpoint_path, latest_ckpt)
    print(f'[INFO] Using checkpoint: {checkpoint}')

    save_path = os.path.join(
        './result',
        f"{args.name}_{os.path.splitext(os.path.basename(args.audio_path))[0]}_{os.path.splitext(latest_ckpt)[0]}.mp4",
    )
    dataset_dir = os.path.join('./dataset', args.name)
    audio_path = args.audio_path
    mode = args.asr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==== Audio feature extraction ============================================================
    model = AudioEncoder().to(device).eval()
    audio_enc_ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth', map_location=device)
    model.load_state_dict({f'audio_encoder.{k}': v for k, v in audio_enc_ckpt.items()})

    dataset = AudDataset(audio_path)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    outputs = []
    for mel in data_loader:
        mel = mel.to(device)
        with torch.no_grad():
            out = model(mel)
        outputs.append(out)
    outputs = torch.cat(outputs, dim=0).cpu()
    first_frame, last_frame = outputs[:1], outputs[-1:]
    audio_feats = torch.cat([first_frame, outputs, last_frame], dim=0).numpy()

    # ==== Dataset paths =====================================================================
    img_dir = os.path.join(dataset_dir, 'full_body_img')
    lms_dir = os.path.join(dataset_dir, 'landmarks')
    if not os.path.isdir(img_dir) or not os.path.isdir(lms_dir):
        raise FileNotFoundError(f'Dataset directories not found under {dataset_dir}')

    len_img = len(os.listdir(img_dir)) - 1
    example_img = cv2.imread(os.path.join(img_dir, '0.jpg'))
    if example_img is None:
        raise FileNotFoundError(f'Failed to read example frame from {img_dir}')
    frame_h, frame_w = example_img.shape[:2]

    parsing_dir = os.path.join(dataset_dir, 'parsing') if args.parsing else None

    # ==== Video writer =======================================================================
    temp_save_path = save_path.replace('.mp4', 'temp.mp4')
    fps = args.smooth_fps if args.smooth_fps > 0 else (25 if mode in ('hubert', 'ave') else 20)
    video_writer = cv2.VideoWriter(
        temp_save_path,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fps,
        (frame_w, frame_h),
    )

    # ==== Model loading =====================================================================
    net = Model(6, mode).to(device)
    state = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state)
    net.eval()

    # ==== Landmark smoothing ================================================================
    smoother = None

    step_stride = 0
    img_idx = 0

    for i in tqdm(range(audio_feats.shape[0])):
        if img_idx > len_img - 1:
            step_stride = -1
        if img_idx < 1:
            step_stride = 1
        img_idx += step_stride

        frame_id = img_idx + args.start_frame
        img_path = os.path.join(img_dir, f'{frame_id}.jpg')
        lms_path = os.path.join(lms_dir, f'{frame_id}.lms')

        img = cv2.imread(img_path)
        if img is None:
            continue

        lms_list = []
        with open(lms_path, 'r') as f:
            for line in f.read().splitlines():
                arr = np.array(line.split(' '), dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.float32)

        if args.use_smooth:
            if smoother is None:
                smoother = LandmarkSmoother(num_points=int(lms.shape[0]), fps=fps)
            lms = smoother.smooth(lms)

        lms_int = lms.astype(np.int32)
        xmin = lms_int[1][0]
        ymin = lms_int[52][1]
        xmax = lms_int[31][0]
        width = xmax - xmin
        ymax = ymin + width

        crop_img = img[ymin:ymax, xmin:xmax]
        if crop_img.size == 0:
            continue
        crop_img_par = crop_img.copy()

        if parsing_dir is not None:
            parsing_img = cv2.imread(os.path.join(parsing_dir, f'{frame_id}.png'))
            crop_parsing_img = parsing_img[ymin:ymax, xmin:xmax] if parsing_img is not None else None
        else:
            crop_parsing_img = None

        crop_h, crop_w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()

        img_masked = cv2.rectangle(img_real_ex_ori.copy(), (5, 5), (310, 305), (0, 0, 0), -1)
        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0).to(device)
        img_masked_T = torch.from_numpy(img_masked / 255.0).to(device)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None]

        audio_feat = get_audio_features(audio_feats, i)
        if mode == 'hubert':
            audio_feat = audio_feat.reshape(32, 32, 32)
        elif mode == 'wenet':
            audio_feat = audio_feat.reshape(256, 16, 32)
        else:  # ave
            audio_feat = audio_feat.reshape(32, 16, 16)
        audio_feat = audio_feat[None].to(device)

        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)[0]

        pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        crop_img_ori[4:324, 4:324] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

        if crop_parsing_img is not None:
            parsing_mask = (
                (crop_parsing_img == [0, 0, 255]).all(axis=2) |
                (crop_parsing_img == [255, 255, 255]).all(axis=2)
            )
            crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]

        img[ymin:ymax, xmin:xmax] = crop_img_ori
        video_writer.write(img)

    video_writer.release()

    os.system(f"ffmpeg -i {temp_save_path} -i {audio_path} -c:v libx264 -c:a aac -crf 20 {save_path} -y")
    os.remove(temp_save_path)
    print(f'[INFO] ===== save video to {save_path} =====')


if __name__ == '__main__':
    main()
