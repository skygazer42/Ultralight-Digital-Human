import os
import cv2
import argparse
import numpy as np
import subprocess
import sys
from pathlib import Path
#数据预处理主脚本。这是数据准备的入口，它会自动调用其他脚本来完成视频抽帧、人脸检测、关键点提取、音频提取和特征提取等一系列操作。

ROOT_DIR = Path(__file__).resolve().parents[1]


def _build_env_with_root():
    env = os.environ.copy()
    root = str(ROOT_DIR)
    current = env.get('PYTHONPATH', '')
    parts = [p for p in current.split(os.pathsep) if p]
    if root not in parts:
        parts.insert(0, root)
    env['PYTHONPATH'] = os.pathsep.join(parts) if parts else root
    return env

#提取音频这个函数的功能非常专一，就是把视频文件中的声音“剥离”出来，存成一个独立的 .wav 音频文件。
def extract_audio(path, out_path, sample_rate=16000):
     #代表 audio rate，设置音频的采样率为16000Hz。为什么是16000？ 因为AI模型在训练时听的音频都是这个标准，所以我们必须把新的音频也统一成这个标准，模型才能“听懂”。
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = [
        'ffmpeg', '-y', '-i', path, '-f', 'wav', '-ac', '1', '-ar', str(sample_rate), out_path
    ]
    subprocess.run(cmd, check=True)
    print(f'[INFO] ===== extracted audio =====')
   #extract_images提取图像帧把视频像播放幻灯片一样，一帧一帧地拆开，并把每一帧都存成一张独立的 .jpg 图片。
def extract_images(path, mode):
    
    
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)
    
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if mode == "hubert" and fps != 25:
        raise ValueError("Using hubert,your video fps should be 25!!!")
    if mode == "wenet" and fps != 20:
        raise ValueError("Using wenet,your video fps should be 20!!!")
        
    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(full_body_dir+"/"+str(counter)+'.jpg', frame)
        counter += 1
  #提取音频特征人能听懂声音，但计算机不行。这个函数的作用就是把 .wav 声音文件，转换成AI模型能“听懂”的数字特征（一长串数字，也就是向量或数组）。
def get_audio_feature(wav_path, mode):
    
    print("extracting audio feature...")
    
    if mode == "wenet":
        subprocess.run([sys.executable, "-m", "data_utils.wenet_infer", wav_path], check=True, env=_build_env_with_root())
    if mode == "hubert":
        subprocess.run([sys.executable, "-m", "data_utils.hubert", "--wav", wav_path], check=True, env=_build_env_with_root())
    #遍历刚刚提取出来的每一张图片，在图片上找到人脸，并精确定位出眼睛、眉毛、鼻子、嘴巴等部位的68个或更多关键点的坐标。这些点就是“landmarks”。它们是驱动数字人表情和口型的核心数据。
# def get_landmark(path, landmarks_dir):
#     print("detecting landmarks...")
#     full_img_dir = path.replace(path.split("/")[-1], "full_body_img")
#
#     from get_landmark import Landmark
#     landmark = Landmark()
#
#     for img_name in os.listdir(full_img_dir):
#         if not img_name.endswith(".jpg"):
#             continue
#         img_path = os.path.join(full_img_dir, img_name)
#         lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
#         pre_landmark, x1, y1 = landmark.detect(img_path)
#         with open(lms_path, "w") as f:
#             for p in pre_landmark:
#                 x, y = p[0]+x1, p[1]+y1
#                 f.write(str(x))
#                 f.write(" ")
#                 f.write(str(y))
#                 f.write("\n")

def get_landmark(path, landmarks_dir, provider='fa', fa_cpu=False):
    print(f"detecting landmarks with provider={provider} ...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")
    if provider == 'pfld':
        from get_landmark import Landmark
        landmark = Landmark()
    elif provider == 'fa':
        from fa_landmark import Landmark
        landmark = Landmark(use_cuda=(not fa_cpu), use_3d=False)
    else:
        raise ValueError("provider must be 'pfld' or 'fa'")

    for img_name in os.listdir(full_img_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        if pre_landmark.size == 0:
            continue
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0] + x1, p[1] + y1
                f.write(f"{x} {y}\n")
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--asr', type=str, default='hubert', help="wenet or hubert")
    opt = parser.parse_args()
    asr_mode = opt.asr

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    # 提取声音wav
    extract_audio(opt.path, wav_path)
    # 转成 img图片
    extract_images(opt.path, asr_mode)
    # 转出landmark骨骼
    get_landmark(opt.path, landmarks_dir)
    # 获取音频特征
    get_audio_feature(wav_path, asr_mode)
    
    