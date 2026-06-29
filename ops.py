import torch
import torch.nn as nn


class _Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=True,
        )
        self.weight.data = torch.normal(
            torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02
        )
        self.bias.data = torch.zeros(out_channels)


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN=False, act=None, stride=1):
        super().__init__()
        m = [_Conv(in_channel, out_channel, kernel_size, stride=stride)]

        if BN:
            m.append(nn.BatchNorm2d(num_features=out_channel))
        if act is not None:
            m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.body = nn.Sequential(
            conv(channels, channels, kernel_size, BN=True, act=act),
            conv(channels, channels, kernel_size, BN=True, act=None),
        )

    def forward(self, x):
        return self.body(x) + x


class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act=nn.ReLU(inplace=True)):
        super().__init__()
        m = [
            conv(channel, channel * scale * scale, kernel_size),
            nn.PixelShuffle(scale),
        ]
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act=nn.LeakyReLU(inplace=True)):
        super().__init__()
        self.body = nn.Sequential(
            conv(in_feats, out_feats, kernel_size, BN=True, act=act),
            conv(out_feats, out_feats, kernel_size, BN=True, act=act, stride=2),
        )

    def forward(self, x):
        return self.body(x)
