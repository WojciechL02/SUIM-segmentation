import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from model.block import Block


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)) -> None:
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, enc_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            enc_feature = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feature], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_features, x):
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        return enc_features
