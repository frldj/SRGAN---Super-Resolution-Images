import torch.nn as nn
from ops import conv, ResBlock, Upsampler, discrim_block


class Generator(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=16, act=nn.PReLU(), scale=4):
        super().__init__()

        self.head = conv(img_feat, n_feats, kernel_size=9, BN=False, act=act)
        self.body = nn.Sequential(*[ResBlock(n_feats, kernel_size, act=act) for _ in range(num_block)])
        self.body_tail = conv(n_feats, n_feats, kernel_size=3, BN=True, act=None)

        if scale == 4:
            upsample_blocks = [Upsampler(n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(n_feats, kernel_size=3, scale=scale, act=act)]
        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = conv(n_feats, img_feat, kernel_size=3, BN=False, act=nn.Tanh())

    def forward(self, x):
        x = self.head(x)
        skip = x

        x = self.body(x)
        x = self.body_tail(x)
        feat = x + skip

        x = self.tail(feat)
        x = self.last_conv(x)

        return x, feat


class Discriminator(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3,
                 act=nn.LeakyReLU(inplace=True), num_of_block=3, patch_size=96):
        super().__init__()
        self.act = act

        self.head = nn.Sequential(
            conv(img_feat, n_feats, kernel_size=3, BN=False, act=self.act),
            conv(n_feats, n_feats, kernel_size=3, BN=False, act=self.act, stride=2),
        )

        self.body = nn.Sequential(*[
            discrim_block(n_feats * (2 ** i), n_feats * (2 ** (i + 1)), kernel_size, act=self.act)
            for i in range(num_of_block)
        ])

        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))

        self.tail = nn.Sequential(
            nn.Linear(self.linear_size, 1024),
            self.act,
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = x.view(-1, self.linear_size)
        return self.tail(x)
