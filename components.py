
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# Residual Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # left
        lx = self.conv(x)

        #righr
        rx = x

        return lx + rx


# %%
# Conv Block Down
class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlockDown, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


# %%
# Conv Block Up
class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlockUp, self).__init__()

        self.convUp = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.convUp(x)

