# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import numpy as np
import torch.nn as nn

from components import ResidualBlock, ConvBlockDown, ConvBlockUp

# %%

"""
Implements the Generator as proposed in the paper.
The Architecture of the Generator is as follows
    The input is first padded using reflection pad
    Then the input passes through 3 convolution blocks
    Then there are 9 Residual Blocks
        The above forms the encoder part of the Generator

    Then there are 3 ConvTranspose Layers that decode the image
"""


class Generator(nn.Module):
    def __init__(self, name, in_features, out_features):
        super(Generator, self).__init__()

        # name
        self.name = name

        # Input Reflection Pad
        self.rfpadU = nn.ReflectionPad2d(3)

        # DownSampling
        self.convD1 = ConvBlockDown(in_features, 64, 7)
        self.convD2 = ConvBlockDown(64, 128, 3)
        self.convD3 = ConvBlockDown(128, 256, 3)

        # Residual Layers
        self.res = []
        for _ in range(9):
            self.res.append(ResidualBlock(256, 256))

        # Upsample Layers
        self.convU3 = ConvBlockUp(256, 128, 3)
        self.convU2 = ConvBlockUp(128, 64, 3)

        # Last Layer
        self.rfpadU = nn.ReflectionPad2d(3)
        self.convU1 = ConvBlockUp(64, out_features, 7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.rfpadU(x)
        x = self.convD1(x)
        x = self.convD2(x)
        x = self.convD3(x)

        for block in self.res:
            x = block(x)

        x = self.convU3(x)
        x = self.convU2(x)
        x = self.rfpadU(x)
        x = self.convU1(x)
        x = self.tanh(x)

        return x

    def __str__(self):
        return self.name

        # %%
"""
 Implements the Discriminator Model as proposed in the paper. 
"""


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
