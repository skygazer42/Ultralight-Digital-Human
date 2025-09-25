# -*- coding: utf-8 -*-
import os, argparse, numpy as np, torch
from tqdm import trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasetsss import MyDataset
from syncnet import SyncNet_color

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据目录，例如 ./data_utils")
    ap.add_argument("--asr", default="hubert", choices=["hubert","wenet"])
    ap.add_argument("--ckpt", required=True, help="SyncNet ckpt 路径，例如 ./syncnet_ckpt/39.pth")
    ap.add_argument("--gamma", type=float, default=1.0, help="难度指数，>1 更侧重难样本")
    ap.add_argument("--alpha", type=float, default=1.0, help="对覆盖权重（weights.npy）的权重指数")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = MyDataset(args.root, args.asr)
    N = len(ds)

    # 载入 SyncNet
    net = SyncNet_color(args.asr).eval().to(device)
    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state)

    scores = np.zeros(N, dtype=np.float32)

    for i in trange(N, desc="Scoring with SyncNet"):
        img_concat_T, img_real_T, audio_feat = ds[i]  # 真实图像 + 同步音频窗
        img = img_real_T.unsqueeze(0).to(device)      # [1,3,160,160]
        audio = audio_feat.unsqueeze(0)               # [1,C,H,W]
        # Hubert：UNet 用16通道，SyncNet习惯32通道，这里补到 32
        if args.asr == "hubert" and audio.shape[1] == 16:
            audio = torch.cat([audio, audio], dim=1)
        elif args.asr == "wenet" and audio.shape[1] == 128:
            audio = torch.cat([audio, audio], dim=1)
        audio = audio.to(device)
        a, v = net(img, audio)
        sim = torch.nn.functional.cosine_similarity(a, v).mean().clamp(-1,1).item()
        scores[i] = sim

    # 分数 [-1,1] → 相似度 [0,1] → 难度 [0,1]
    s = (scores + 1.0) / 2.0
    d = (1.0 - s) ** args.gamma

    # 读取 A1 的覆盖权重
    base_path = os.path.join(args.root, "weights.npy")
    if os.path.isfile(base_path):
        w_base = np.load(base_path)[:N]
    else:
        w_base = np.ones(N, dtype=np.float32)

    # 融合：困难优先 × 覆盖均衡
    w = d * (w_base ** args.alpha)
    w = w / (w.mean() + 1e-9)

    out = os.path.join(args.root, "weights_hard.npy")
    np.save(out, w.astype(np.float32))
    print(f"[OK] saved {out}, mean={w.mean():.3f}")

if __name__ == "__main__":
    main()
