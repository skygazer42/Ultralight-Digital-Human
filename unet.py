import torch
from torch import nn
from torch.nn import functional as F

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        mid = inp * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class DoubleConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride, use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1, use_res_connect=True, expand_ratio=2),
        )

    def forward(self, x): return self.double_conv(x)

class InConvDw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=1, use_res_connect=False, expand_ratio=2)
        )

    def forward(self, x): return self.inconv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = DoubleConvDW(in_channels, out_channels, stride=2)

    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x1, x2], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x): return self.conv(x)


class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sig(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class SpatialAttnLite(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.sig(self.conv(y))
        return x * y

class AudioConvWenet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[2], ch[3], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(ch[3]);
        self.relu = nn.ReLU()
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    def forward(self, x):
        x = self.conv1(x);
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x);
        x = self.conv7(x)
        return x

class AudioConvHubert(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(16, ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(ch[3]);
        self.relu = nn.ReLU()
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
    def forward(self, x):
        x = self.conv1(x);
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x);
        x = self.conv7(x)
        return x


class AudioFiLM(nn.Module):
    """从音频向量 a 生成通道级 scale/bias：y = x * (1 + gamma) + beta"""

    def __init__(self, audio_dim: int, out_channels: int, hidden: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * out_channels)
        )

    def forward(self, x, a_vec):
        C = x.shape[1]  # 实际特征通道
        assert C == self.out_channels, f"AudioFiLM expects {self.out_channels} channels, got {C}"
        gb = self.fc(a_vec)  # (B, 2C)
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class AudioGate(nn.Module):
    """音频向量 → 跳连通道门控（sigmoid）"""

    def __init__(self, audio_dim: int, skip_channels: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, skip_channels), nn.Sigmoid()
        )

    def forward(self, skip, a_vec):
        g = self.fc(a_vec).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return skip * g


class Model(nn.Module):
    def __init__(self, n_channels=6, mode='wenet'):
        super().__init__()
        self.n_channels = n_channels
        ch = [32, 64, 128, 256, 512]

        # 音频分支 & 全局向量
        self.audio_model = AudioConvHubert() if mode == 'hubert' else AudioConvWenet()
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 图像编码器
        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        # 瓶颈融合（与你原版保持一致）
        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1)  # -> 256
        )

        # 解码
        self.up1 = Up(ch[4], ch[3] // 2)  # 256+256 -> 128
        self.up2 = Up(ch[3], ch[2] // 2)  # 128+128 -> 64
        self.up3 = Up(ch[2], ch[1] // 2)  # 64 + 64 -> 32
        self.up4 = Up(ch[1], ch[0])  # 32 + 32 -> 32
        self.outc = OutConv(ch[0], 3)

        # ---- 注意力：低分辨率与解码端 ----
        self.eca_bottleneck = ECA(k_size=3)
        self.spa_bottleneck = SpatialAttnLite(k=7)
        self.eca_up1 = ECA(3);
        self.eca_up2 = ECA(3);
        self.eca_up3 = ECA(3);
        self.eca_up4 = ECA(3)
        self.spa_up1 = SpatialAttnLite(7);
        self.spa_up2 = SpatialAttnLite(7)
        self.spa_up3 = SpatialAttnLite(7);
        self.spa_up4 = SpatialAttnLite(7)

        # ---- 多尺度 Audio-FiLM（通道数与各层输出严格对齐）+ 跳连门控 ----
        audio_dim = ch[4]
        self.film_bottleneck = AudioFiLM(audio_dim, ch[3])  # 256
        self.film_up1 = AudioFiLM(audio_dim, ch[3] // 2)  # 128  ← 修复
        self.film_up2 = AudioFiLM(audio_dim, ch[2] // 2)  # 64
        self.film_up3 = AudioFiLM(audio_dim, ch[1] // 2)  # 32   ← 修复
        self.film_up4 = AudioFiLM(audio_dim, ch[0])  # 32

        self.gate_x4 = AudioGate(audio_dim, ch[3])  # x4(256)
        self.gate_x3 = AudioGate(audio_dim, ch[2])  # x3(128)
        self.gate_x2 = AudioGate(audio_dim, ch[1])  # x2(64)
        self.gate_x1 = AudioGate(audio_dim, ch[0])  # x1(32)

    def forward(self, x, audio_feat):
        # 图像编码
        x1 = self.inc(x)  # 32
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 128
        x4 = self.down3(x3)  # 256
        x5 = self.down4(x4)  # 512

        # 音频编码 + 向量
        a_map = self.audio_model(audio_feat)  # (B,512,h,w)
        a_vec = self.gap(a_map).flatten(1)  # (B,512)

        # 跳连门控
        x4 = self.gate_x4(x4, a_vec)
        x3 = self.gate_x3(x3, a_vec)
        x2 = self.gate_x2(x2, a_vec)
        x1 = self.gate_x1(x1, a_vec)

        # 瓶颈融合 + 注意力 + FiLM
        x5 = torch.cat([x5, a_map], dim=1)  # (B,1024,h,w)
        x5 = self.fuse_conv(x5)  # (B,256,h,w)
        x5 = self.eca_bottleneck(x5);
        x5 = self.spa_bottleneck(x5)
        x5 = self.film_bottleneck(x5, a_vec)

        # 解码 + 每层 FiLM + 注意力
        x = self.up1(x5, x4);
        x = self.film_up1(x, a_vec);
        x = self.eca_up1(x);
        x = self.spa_up1(x)
        x = self.up2(x, x3);
        x = self.film_up2(x, a_vec);
        x = self.eca_up2(x);
        x = self.spa_up2(x)
        x = self.up3(x, x2);
        x = self.film_up3(x, a_vec);
        x = self.eca_up3(x);
        x = self.spa_up3(x)
        x = self.up4(x, x1);
        x = self.film_up4(x, a_vec);
        x = self.eca_up4(x);
        x = self.spa_up4(x)

        out = torch.sigmoid(self.outc(x))
        return out
