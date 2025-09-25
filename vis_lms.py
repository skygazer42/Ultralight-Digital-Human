# tools/vis_lms.py
import os, glob, argparse
import cv2
import numpy as np

from mouth_roi import compute_mouth_roi as shared_compute_mouth_roi

def compute_mouth_roi(lms: np.ndarray, img_shape):
    """Proxy to shared ROI helper for visualization."""
    return shared_compute_mouth_roi(lms, img_shape)



def load_lms(path):
    # 每行: "x y"
    pts = []
    with open(path, 'r') as f:
        for line in f:
            ss = line.strip().split()
            if len(ss) < 2: continue
            x, y = float(ss[0]), float(ss[1])
            pts.append([x, y])
    return np.array(pts, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True, help='e.g. data_utils/full_body_img')
    ap.add_argument('--lms_dir', required=True, help='e.g. data_utils/landmarks')
    ap.add_argument('--out_dir', required=True, help='output debug dir')
    ap.add_argument('--provider', choices=['auto','fa','pfld'], default='auto')
    ap.add_argument('--with_index', action='store_true')
    ap.add_argument('--step', type=int, default=1, help='visualize every N frames')
    ap.add_argument('--limit', type=int, default=0, help='max frames to dump (0=all)')
    ap.add_argument('--make_video', action='store_true')
    ap.add_argument('--fps', type=int, default=20, help='video fps if make_video')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_paths = sorted([p for p in glob.glob(os.path.join(args.img_dir, '*.jpg'))],
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    dumped = 0
    for i, img_p in enumerate(img_paths):
        if i % args.step != 0:
            continue
        base = os.path.splitext(os.path.basename(img_p))[0]
        lms_p = os.path.join(args.lms_dir, base + '.lms')
        if not os.path.exists(lms_p):
            continue
        img = cv2.imread(img_p)
        if img is None:
            continue
        lms = load_lms(lms_p)
        if lms.size == 0:
            continue

        n = lms.shape[0]
        prov = args.provider
        if prov == 'auto':
            prov = 'fa' if 68 <= n <= 70 else 'pfld'

        # 画所有点
        vis = img.copy()
        for j, (x, y) in enumerate(lms.astype(int)):
            cv2.circle(vis, (int(x), int(y)), 1, (255, 200, 0), -1)
            if args.with_index and (j % 5 == 0):
                cv2.putText(vis, str(j), (int(x)+2, int(y)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 画嘴部 ROI 和（若为FA）嘴部点集
        xmin, ymin, xmax, ymax = compute_mouth_roi(lms, vis.shape)
        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        mouth_idx = np.arange(48, 68) if 68 <= n <= 70 else None
        if mouth_idx is not None:
            for j in mouth_idx:
                x, y = int(lms[j,0]), int(lms[j,1])
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        # 角落标注：provider / 点数
        cv2.putText(vis, f'{prov} | {n} pts | frame {i}', (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 255, 30), 2, cv2.LINE_AA)

        out_p = os.path.join(args.out_dir, f'{base}.jpg')
        cv2.imwrite(out_p, vis)
        dumped += 1
        if args.limit and dumped >= args.limit:
            break

    if args.make_video and dumped > 0:
        # 合成视频（按输出目录里的顺序）
        frames = sorted(glob.glob(os.path.join(args.out_dir, '*.jpg')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        h, w = cv2.imread(frames[0]).shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(os.path.join(args.out_dir, 'vis.mp4'), fourcc, args.fps, (w, h))
        for f in frames:
            vw.write(cv2.imread(f))
        vw.release()
        print(f'[OK] video saved: {os.path.join(args.out_dir, "vis.mp4")}')
    print(f'[OK] dumped {dumped} frames to {args.out_dir}')

if __name__ == '__main__':
    main()
