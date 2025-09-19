from transformers import Wav2Vec2Processor, HubertModel #是一个方便的工具，它负责音频的预处理。它的工作是接收原始的音频波形，并将其转换成 HuBERT 模型能够“消化”的标准格式（比如进行归一化等操作）。
import soundfile as sf
import numpy as np
import torch
#Hubert音频特征提取器。使用Hubert模型将 .wav 音频文件转换为数字特征向量 (.npy 文件)，供模型训练和推理使用。
print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert
#这是一个“装饰器”，它告诉PyTorch：“接下来只是进行推断（inference），不需要计算梯度（用于训练）”。这能极大地提升速度并节省显存。
@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    #将一个可能非常长的音频，通过HuBERT模型，转换成特征序列。
    global hubert_model
    hubert_model = hubert_model.to(device)
    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    #wav2vec2_processor 将整个音频 speech 转换成一个巨大的张量 input_values_all，为模型准备好了“口粮”。


    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400 #模型为了计算出这 1 个特征向量，它需要“看”一个长度为 400 个采样点的音频窗口。这个窗口我们称之为“感受野”或“卷积核大小”。
    stride = 320 #它每向前“滑动” 320 个原始音频采样点，就会生成 1 个特征向量（在输出端的时间步长是1）。
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
   # 循环处理每一个分块不能简单地把长音频切成不重叠的小段，因为这样会在切割处产生“接缝”，影响特征质量。正确的做法是切成有重叠的小段，
    # 让模型在处理每一段时都能看到上一段的“尾巴”，从而保证特征的连续性。kernel（窗口大小）和 stride（步长）这两个参数就是用来计算如何进行这种重叠切分的。
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            #为了计算出第1000个特征向量（对应 320*999 到 320*999+400 这个窗口），模型的感受野最右端需要达到 319680+400 = 320080 这个位置。所以第一块面包必须切得比 320000 更长一点，保证最后一块窗口是完整的。
            #第一块送入模型的音频，是从 0 到 320080。
            #第二块送入模型的音频，是从 320000 到 640080。
            #这个重叠区域的数据被处理了两次，但我们只在拼接时使用一次，这就完美解决了“接缝”问题，保证了在 320000 这个切割点上，特征的计算是带有正确上下文的。
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
   #有一个关键检查 if input_values.shape[1] >= kernel:。如果剩下的“面包屑”连 400 个采样点都不到，那模型连一个完整的窗口都凑不出来，自然也无法生成任何特征，所以就直接跳过。 扔掉特征点不足问题
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

def make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor

import soundfile as sf
import numpy as np
import torch
from argparse import ArgumentParser
import librosa

parser = ArgumentParser()
parser.add_argument('--wav', type=str, help='')
args = parser.parse_args()

wav_name = args.wav

#sf.read(wav_name) 使用 soundfile 库读取指定的音频文件。它返回两个值：speech (包含音频波形数据的NumPy数组) 和 sr (原始音频的采样率，例如 44100 Hz)。
speech, sr = sf.read(wav_name)
#使用 librosa 库将音频的采样率强制转换为 16000 Hz。这是必需的，因为 facebook/hubert-large-ls960-ft 这个预训练模型是在 16kHz 采样率的音频数据上进行训练的
speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
print("SR: {} to {}".format(sr, 16000))
# print(speech.shape, speech_16k.shape)

hubert_hidden = get_hubert_from_16k_speech(speech_16k)
#加载了预训练的 HuBERT 模型，并执行了复杂的分块推理逻辑来处理可能很长的音频。
hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
np.save(wav_name.replace('.wav', '_hu.npy'), hubert_hidden.detach().numpy())
print(hubert_hidden.detach().numpy().shape)