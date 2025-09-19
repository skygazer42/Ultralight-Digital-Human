# -*- coding: utf-8 -*-
import os, argparse, glob, numpy as np
from sklearn.cluster import MiniBatchKMeans
_HAS_SK = True
def _simple_kmeans(x, k=50, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    c = x[idx].copy()
    for _ in range(iters):
        # L2 最近中心
        d2 = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for j in range(k):
            m = (lab == j)
            if m.any(): c[j] = x[m].mean(0)
    # 赋标签
    d2 = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    lab = d2.argmin(1)
    return lab

def mouth_open_ratio(lms_file):
    pts=[]
    with open(lms_file, "r") as f:
        for line in f.read().splitlines():
            if not line.strip(): continue
            x,y = line.split(" "); pts.append([float(x), float(y)])
    p=np.array(pts, dtype=np.float32)
    # 上/下唇 52,58；脸宽 1~31
    h = abs(p[58,1]-p[52,1]); w = max(1.0, abs(p[31,0]-p[1,0]))
    return float(h/w)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据目录，例如 ./data_utils")
    ap.add_argument("--asr", default="hubert", choices=["hubert","wenet"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--alpha_mouth", type=float, default=0.5)
    ap.add_argument("--alpha_unit", type=float, default=0.5)
    args=ap.parse_args()

    aud_path = os.path.join(args.root, "aud_hu.npy" if args.asr=="hubert" else "aud_wenet.npy")
    aud = np.load(aud_path).astype(np.float32)
    T, D = aud.shape[0], int(np.prod(aud.shape[1:]))
    X = aud.reshape(T, D)

    # HuBERT 单元聚类
    if _HAS_SK:
        km = MiniBatchKMeans(n_clusters=args.k, batch_size=4096, n_init="auto", random_state=0)
        labels = km.fit_predict(X)
    else:
        labels = _simple_kmeans(X, k=args.k, iters=30, seed=0)
    cnt = np.bincount(labels, minlength=args.k).astype(np.float32)
    unit_inv = (1.0 / np.maximum(cnt, 1.0))[labels]      # 稀有单元更大
    unit_inv /= (unit_inv.mean() + 1e-9)

    # 口型开口度稀有度
    lms_dir = os.path.join(args.root, "landmarks")
    lms_files = sorted(glob.glob(os.path.join(lms_dir, "*.lms")),
                       key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
    N = min(T, len(lms_files))
    mouth = np.array([mouth_open_ratio(lms_files[i]) for i in range(N)], dtype=np.float32)
    q25, q50, q75 = np.percentile(mouth, [25,50,75])
    dev = np.abs(mouth - q50) / (q75 - q25 + 1e-6)
    dev = np.clip(dev, 0, 3.0)
    if N < T:
        dev = np.pad(dev, (0, T-N), constant_values=float(dev.mean()))
    dev /= (dev.mean() + 1e-9)

    # 融合并归一
    w = args.alpha_unit * unit_inv + args.alpha_mouth * dev
    w = w / (w.mean() + 1e-9)

    out = os.path.join(args.root, "weights.npy")
    np.save(out, w.astype(np.float32))
    print(f"[OK] saved {out}, shape={w.shape}, mean={w.mean():.3f}")

if __name__=="__main__":
    main()
