import torch.nn as nn
from model.block import Block


class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)) -> None:
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [
                Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        block_outputs = []

        for block in self.enc_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)

        return block_outputs
