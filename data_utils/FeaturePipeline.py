import numpy as np
import yaml
import torch
import torchaudio.compliance.kaldi as kaldi #调用了 torchaudio.compliance.kaldi 里的 fbank 函数。fbank 是 filterbank（滤波器组）的缩写，是计算梅尔频谱图的另一种叫法。

class Feature_Pipeline():

    def __init__(self, engine_config):
        #with open(model_config_path, 'r') as fin:
        #    self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.configs = engine_config
        #梅尔频率箱的数量。你可以把它想象成最终生成的频谱图的“高度”，或者说“垂直分辨率”。它表示我们把声音的频率范围划分成了80个子频段来分析。80是一个非常经典和常用的值。
        self.num_mel_bins = self.configs['data_conf']['fbank_conf']['num_mel_bins'] # 80
        #计算机分析音频时，不是连续分析，而是把它切成一个个非常短的、重叠的“小窗”（frame）来处理。这里就是把音频切成每段25毫秒的小窗。
        self.frame_length = self.configs['data_conf']['fbank_conf']['frame_length'] # 25
        # self.frame_shift = 1
        #帧移，单位是毫秒。它定义了“小窗”滑动的步长。也就是说，分析完一个25毫秒的窗口后，下一个窗口不是从第26毫秒开始，而是从第11毫秒开始，这样帧与帧之间就有15毫秒的重叠。这种重叠可以使得特征更加平滑和稳定。
        self.frame_shift = self.configs['data_conf']['fbank_conf']['frame_shift']   # 10
        #抖动。这是一个高级参数。它会在原始音频上增加一点点微小的随机噪声。这样做可以防止在计算中出现纯零的情况，从而增加特征的鲁棒性
        self.dither = self.configs['data_conf']['fbank_conf']['dither'] # 0.0
        #采样率。再次强调，这个参数定义了每秒钟采集多少个音频数据点，必须和模型训练时用的保持一致
        self.sample_rate = self.configs['engine_sample_rate_hertz'] # 16000
        #self.feature_queue_ = torch.tensor([[0]])
        #流式处理准备的状态变量，用来暂存接收到的音频数据。
        self.first_wav_ = b''
        self.remained_wav_ = b''
        self._waveform = b'' #torch.tensor([[0]])
        self.exist_endpoint = False
# AcceptWaveform 和 ReadFeats，它们主要是为了支持流式音频输入而设计的辅助功能
    def AcceptWaveform(self, audio):  # audio: b''
        first, self.remained_wav_, ExitEndpoint = self.vad.endpoint_detect(audio)
        self._waveform += first
        #if ExitEndpoint:
        #self._waveform = self._waveform + self.first_wav_
        #else:
        #    self._waveform
        #self.remained_wav_ = second_wav
        '''
        feat, feat_length = self._extact_feature(waveform)
        if self.feature_queue_.shape[1]==1:
            self.feature_queue_ = feat
        else:
            self.feature_queue_ = torch.cat((self.feature_queue_, feat), 1)
        '''
        self.exist_endpoint = ExitEndpoint
        #self.mutex.release()
        #print('待计算音频长度：', len(self._waveform))
        return ExitEndpoint #, self.feature_queue_.shape[1]

    def _extract_feature(self, waveform_int16):
        #print(waveform_int16.shape)
        #assert max(waveform_int16.shape) > 512
        #输入的 waveform_int16（一维的波形数据）就被转换成了一个二维的数组 feat，也就是梅尔频谱图。这个数组的形状是 [帧数, 80]。
        feat = kaldi.fbank(waveform_int16,
                          num_mel_bins=self.num_mel_bins,
                          frame_length=self.frame_length,
                          frame_shift=self.frame_shift,
                          dither=self.dither, 
                          energy_floor=0.0,
                          sample_frequency=self.sample_rate
                          )
        #为深度学习模型通常被设计为可以一次性处理**一批（batch）**数据。这行代码就是把我们单个的频谱图，包装成一个“只包含一个样本的批次”。数组的形状从 [帧数, 80] 变成了 [1, 帧数, 80]。
        feat = feat.unsqueeze(0) #.to(device)
        feat_length = torch.IntTensor([feat.size()[1]])
        return feat, feat_length


    def Reset(self):
        self.remained_wav_ = b''
        self._waveform = b''
        #self.feature_queue_ = torch.tensor([[0]])

    def get_waveform_len(self):
        return len(self._waveform)

    def ReadFeats(self):
        if len(self._waveform) < 512:
            return None
        waveform = np.frombuffer(self._waveform, dtype=np.int16)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        feat, feat_length = self._extract_feature(waveform)
        #self._waveform = b''
        if self.exist_endpoint:
            self._waveform = self.remained_wav_
        else: self._waveform = b''
        return feat #, self.exist_endpoint

