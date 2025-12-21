"""
Docstring for resnet.model
- This model is for ResNet trained on ImageNet.
- I realized the ImageNet model is a bit too large for actually training so
_ there is a separate model for CIFAR-10, which should be smaller.
"""

import torch
from torch import nn


class LinearProjection(nn.Module):
    """projects W_x * x"""

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()  # so we take all the features from nn.Module
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=stride,
        )
        # The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions).
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        """forward pass for linear projection for identify mapping"""
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    """
    Building blocks for ResNet-34
    Docstring for ResidualBlock
    Conv -> BN -> Conv -> BN -> + residual
    """

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        if in_channel != out_channel:
            self.skipconnection = LinearProjection(
                in_channel=in_channel, out_channel=out_channel, stride=stride
            )
        else:
            self.skipconnection = nn.Identity()

        self.act2 = nn.ReLU()

    def forward(self, x):
        """
        forward function
        """
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        skipconnection = self.skipconnection(x)

        return self.act2(x + skipconnection)


class ResNet34(nn.Module):
    """
    Docstring for ResNet34
    - num_blocks=[3, 4, 6, 3], num_channels=[64, 128, 256, 512])
    """

    def __init__(self, num_blocks, num_channels):
        super().__init__()

        # ------------- conv1 -------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # techincally, max pooling is part of conv2_x
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = 64

        layers = []
        for i, _ in enumerate(num_blocks):
            # From paper: when the shortcuts go across feature maps of two
            # sizes, they are performed with a stride of 2.
            stride = (
                1 if i == 0 else 2
            )  # for conv2_x, no need to have stride 2 as maxpooling already reduces dim
            layers.append(
                self._make_layers(ResidualBlock, num_channels[i], num_blocks[i], stride)
            )

        self.body = nn.Sequential(*layers)

        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels[-1], 1000)

    def _make_layers(self, block, out_channels, num_blocks, stride):
        layers = []
        # first block connects prev. channel dim to current channel dimm
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # remaining has same channel dims
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Docstring for forward
        """
        x = self.stem(x)
        x = self.body(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
