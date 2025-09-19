import torch
from torch import nn
from torch.nn import functional as F


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super().__init__()
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        mid = max(1, int(inp * expand_ratio))
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
        y = self.conv(x)
        return x + y if self.use_res_connect and x.shape == y.shape else y


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
    def __init__(self, in_channels, out_channels, align_corners=False, mode='nearest'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners if mode == 'bilinear' else None)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        if diffX or diffY:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x1, x2], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x): return self.conv(x)


class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)  # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B,1,C)
        y = self.sig(y.transpose(-1, -2).unsqueeze(-1))  # (B,C,1,1)
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
    def __init__(self, ch):
        super().__init__()
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
    def __init__(self, ch):
        super().__init__()
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
    def __init__(self, audio_dim: int, out_channels: int, hidden: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * out_channels)
        )

    def forward(self, x, a_vec):
        gb = self.fc(a_vec)  # (B, 2*out_channels)
        C = x.shape[1]

        assert gb.shape[1] == 2 * C, \
            f"AudioFiLM expects 2*C={2 * C}, got {gb.shape[1]}. " \
            f"Check out_channels={self.out_channels} vs feature C={C}."
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class AudioGate(nn.Module):
    def __init__(self, audio_dim: int, skip_channels: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(audio_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, skip_channels), nn.Sigmoid()
        )

    def forward(self, skip, a_vec):
        g = self.fc(a_vec).unsqueeze(-1).unsqueeze(-1)
        return skip * g


class TinyCrossAttn(nn.Module):
    def __init__(self, c_vis: int, c_aud: int, d: int = 64):
        super().__init__()
        self.q = nn.Conv2d(c_vis, d, 1, bias=False)
        self.k = nn.Conv2d(c_aud, d, 1, bias=False)
        self.v = nn.Conv2d(c_aud, d, 1, bias=False)
        self.out = nn.Conv2d(d, c_vis, 1, bias=False)
        self.scale = d ** -0.5

    def forward(self, x_vis, x_aud):
        B, _, H, W = x_vis.shape
        q = self.q(x_vis).flatten(2).transpose(1, 2)  # (B, HW, d)
        k = self.k(x_aud).flatten(2)  # (B, d, HW)
        v = self.v(x_aud).flatten(2).transpose(1, 2)  # (B, HW, d)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)  # (B, HW, HW)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(B, -1, H, W)
        return self.out(out) + x_vis


class Model(nn.Module):
    def __init__(self, n_channels=6, mode='wenet',
                 use_cross_attn: bool = False,
                 upsample_mode: str = 'nearest',
                 width_mult: float = 1.0):
        super().__init__()
        self.n_channels = n_channels
        self.use_cross_attn = use_cross_attn

        # 通道系数
        alpha = float(width_mult)
        ch = [max(8, int(32 * alpha)), max(16, int(64 * alpha)), max(24, int(128 * alpha)),
              max(24, int(256 * alpha)), max(32, int(512 * alpha))]

        # 音频分支 & 全局向量
        self.audio_model = AudioConvHubert(ch) if mode == 'hubert' else AudioConvWenet(ch)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 图像编码器
        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),  # 1024a -> 512a
            DoubleConvDW(ch[4], ch[3], stride=1)  # 512a  -> 256a
        )

        # 瓶颈注意力
        if use_cross_attn:
            self.tiny_attn = TinyCrossAttn(c_vis=ch[3], c_aud=ch[4], d=min(64, ch[3]))

        # 解码
        self.up1 = Up(ch[4], ch[3] // 2, mode=upsample_mode)  # (256a + 256a) -> 128a
        self.up2 = Up(ch[3], ch[2] // 2, mode=upsample_mode)  # (128a + 128a) -> 64a
        self.up3 = Up(ch[2], ch[1] // 2, mode=upsample_mode)  # (64a  + 64a)  -> 32a
        self.up4 = Up(ch[1], ch[0], mode=upsample_mode)  # (32a  + 32a)  -> 32a
        self.outc = OutConv(ch[0], 3)

        # 轻注意力：瓶颈 + 最后两层
        self.eca_bottleneck = ECA(k_size=3)
        self.spa_bottleneck = SpatialAttnLite(k=7)
        self.eca_up3 = ECA(3);
        self.spa_up3 = SpatialAttnLite(7)
        self.eca_up4 = ECA(3);
        self.spa_up4 = SpatialAttnLite(7)

        # 多尺度 FiLM + 跳连门控（通道严格对齐）
        audio_dim = ch[4]
        self.film_bottleneck = AudioFiLM(audio_dim, ch[3])  # 256a
        self.film_up1 = AudioFiLM(audio_dim, ch[3] // 2)  # 128a
        self.film_up2 = AudioFiLM(audio_dim, ch[2] // 2)  # 64a  ← 修复点
        self.film_up3 = AudioFiLM(audio_dim, ch[1] // 2)  # 32a
        self.film_up4 = AudioFiLM(audio_dim, ch[0])  # 32a

        self.gate_x4 = AudioGate(audio_dim, ch[3])
        self.gate_x3 = AudioGate(audio_dim, ch[2])
        self.gate_x2 = AudioGate(audio_dim, ch[1])
        self.gate_x1 = AudioGate(audio_dim, ch[0])

    def forward(self, x, audio_feat):
        # 图像编码
        x1 = self.inc(x)  # 32a
        x2 = self.down1(x1)  # 64a
        x3 = self.down2(x2)  # 128a
        x4 = self.down3(x3)  # 256a
        x5 = self.down4(x4)  # 512a

        # 音频编码 + 向量
        a_map = self.audio_model(audio_feat)  # (B,512a,h,w)
        a_vec = self.gap(a_map).flatten(1)  # (B,512a)

        # 跳连做音频门控
        x4 = self.gate_x4(x4, a_vec)
        x3 = self.gate_x3(x3, a_vec)
        x2 = self.gate_x2(x2, a_vec)
        x1 = self.gate_x1(x1, a_vec)

        # 瓶颈融合 + 注意力 + FiLM
        x5 = torch.cat([x5, a_map], dim=1)  # 512a + 512a
        x5 = self.fuse_conv(x5)  # -> 256a
        if hasattr(self, "tiny_attn"):
            x5 = self.tiny_attn(x5, a_map)
        x5 = self.eca_bottleneck(x5);
        x5 = self.spa_bottleneck(x5)
        x5 = self.film_bottleneck(x5, a_vec)

        # 解码（后两层注意力）+ 每层 FiLM
        x = self.up1(x5, x4);
        x = self.film_up1(x, a_vec)
        x = self.up2(x, x3);
        x = self.film_up2(x, a_vec)
        x = self.up3(x, x2);
        x = self.film_up3(x, a_vec);
        x = self.eca_up3(x);
        x = self.spa_up3(x)
        x = self.up4(x, x1);
        x = self.film_up4(x, a_vec);
        x = self.eca_up4(x);
        x = self.spa_up4(x)

        return torch.sigmoid(self.outc(x))


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    with torch.no_grad():
        w = conv.weight
        if conv.bias is None:
            bias = torch.zeros(w.size(0), device=w.device)
        else:
            bias = conv.bias
        gamma = bn.weight;
        beta = bn.bias
        mean = bn.running_mean;
        var = bn.running_var;
        eps = bn.eps
        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta + (bias - mean) * gamma / std
        fused = nn.Conv2d(w.size(1), w.size(0), kernel_size=w.shape[2:4],
                          stride=conv.stride, padding=conv.padding,
                          dilation=conv.dilation, groups=conv.groups, bias=True).to(w.device)
        fused.weight.data.copy_(w_fused)
        fused.bias.data.copy_(b_fused)
        return fused


def fuse_model_bn_(m: nn.Module):
    # 递归把 Conv2d+BN
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Sequential):
            modules = list(child.children())
            new_modules = []
            skip = False
            for i in range(len(modules)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(modules) and isinstance(modules[i], nn.Conv2d) and isinstance(modules[i + 1],
                                                                                             nn.BatchNorm2d):
                    fused = fuse_conv_bn(modules[i], modules[i + 1])
                    new_modules.append(fused)
                    skip = True
                else:
                    new_modules.append(modules[i])
            setattr(m, name, nn.Sequential(*new_modules))
            fuse_model_bn_(getattr(m, name))
        else:
            fuse_model_bn_(child)
