import torch.nn as nn
from torch.nn import functional as F
from model.encoder import Encoder
from model.decoder import Decoder


class UNet(nn.Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16),
                 n_classes=1,
                 retain_dim=True,
                 out_size=(224, 224)) -> None:
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        self.head = nn.Conv2d(dec_channels[-1], n_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        enc_features = self.encoder(x)

        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])

        mask = self.head(dec_features)
        if self.retain_dim:
            mask = F.interpolate(mask, self.out_size)

        return mask
