# Author: Yingru Liu
# Institute: Stony Brook University
# torch modules to be used in deep reinforcement learning.
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, residual=False):
        super(ConvBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )
        self.residual = residual
        return

    def forward(self, input):
        """
        :param input: size is [batch, channels, sequence_length]
        :return:
        """
        hidden = self.network(input)
        return hidden + input if self.residual else hidden



class ResConvNet(nn.Module):
    def __init__(self):
        super(ResConvNet, self).__init__()
        return

    def forward(self, input):
        return

class TextTransformer(nn.Module):
    """
    A roBERTa model to read the sequence of instrument name and extract hidden feature.
    """
    def __init__(self):
        super(TextTransformer, self).__init__()
        return

    def forward(self, input):
        return