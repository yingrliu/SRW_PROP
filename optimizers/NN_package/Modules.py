# Author: Yingru Liu
# Institute: Stony Brook University
# torch modules to be used in deep reinforcement learning.
import torch
import numpy as np
import torch.nn as nn


def GaussianNLL(X, mean, std=1e-2):
    errors = (X - mean) / std
    constant = np.log(2 * np.pi)
    nll = 0.5 * (errors ** 2 + 2 * np.log(std) + constant)
    return nll


class ConvBlock(nn.Module):
    """
    A convolutional block with residual learning (skip connection).
    """
    def __init__(self, in_channels, out_channels, kernel_size, residual=False):
        """

        :param in_channels [Int]: the dimension of features in the input 3D array.
        :param out_channels [Int]: the dimensions of features in the output 3D array.
        :param kernel_size [Int]:  the kernel sizes of convolutional block.
        :param residual [Bool]: whether use residual connection.
        """
        super(ConvBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
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
    """
    A residual convolutional network ( + dense network) that transforms an intensity array into vectorized features.
    """
    def __init__(self, input_size, in_channels, hidden_channels, kernel_sizes, dim_denses, residual=True):
        """

        :param input_size [Tuple(vertical, horizontal)]: the vertical and horizontal dimensions of the intensity distribution.
        :param in_channels [Int]: the dimension of features in the input 3D array.
        :param hidden_channels [List(Int)]: the dimensions of features in each convolutional block.
        :param kernel_sizes [List(Int)]: the kernel size of each convolutional block.
        :param dim_denses [List(Int)]: the dimensions of each hidden layer in the final dense network.
        :param residual [Bool]: whether use residual connection.
        """
        super(ResConvNet, self).__init__()
        assert len(hidden_channels) > 0 and len(hidden_channels) == len(kernel_sizes), \
            "Length hidden_channels should equal kernel_sizes and be larger than 0."
        # build convolutional network.
        channels = [in_channels] + hidden_channels
        self.CNN = nn.Sequential()
        dim_cnn_output = input_size
        for i in range(1, len(channels)):
            self.CNN.add_module('block-{}'.format(i),
                                ConvBlock(channels[i - 1], channels[i], kernel_sizes[i - 1], residual))
            self.CNN.add_module('Pool-{}'.format(i), nn.MaxPool2d(2))
            dim_cnn_output /= 2
            assert dim_cnn_output > 0, "too much block and pooling for the input."
        # build dense network.
        self.DenseNet = get_DenseNet(int(hidden_channels[-1] * dim_cnn_output ** 2), dim_denses)
        return

    def forward(self, input):
        """

        :param input:
        :return:
        """
        return self.DenseNet(self.CNN(input).flatten(start_dim=1))


class NameEntityRNN(nn.Module):
    """
    A RNN model to read the sequence of instrument name and extract hidden feature. (with embedding layer)
    """

    def __init__(self, dimEmbed, dimRNN, entities, UnitType='GRU'):
        super(NameEntityRNN, self).__init__()
        assert UnitType in ['GRU', 'LSTM', 'Transformer'], "Currently only support 'GRU', 'LSTM', 'Transformer'."
        self.word_to_ix = {word: i for i, word in enumerate(entities)}
        self.embeds = nn.Embedding(len(entities), dimEmbed)
        self.Units = get_RNN(dimEmbed, dimRNN, UnitType)
        self.config = {"UnitType": UnitType}
        return

    def forward(self, input):
        word_idx =torch.tensor([self.word_to_ix[item] for item in input], dtype=torch.long).to(next(self.parameters()).device)
        word_embeds = self.embeds(word_idx).unsqueeze(1)
        # todo: compute RNN feature.
        rnn_feature = self.Units(word_embeds)
        return rnn_feature


def get_RNN(dimIn, dimRNN, UnitType="GRU"):
    """
    return RNN cells.
    :param dimIn:
    :param dimRNN:
    :param UnitType:
    :return:
    """
    assert UnitType in ['GRU', 'LSTM', 'Transformer'], "Currently only support 'GRU', 'LSTM', 'Transformer'."
    if UnitType == 'GRU':
        Units = nn.GRU(input_size=dimIn, hidden_size=dimRNN)
    elif UnitType == 'LSTM':
        Units = nn.LSTM(input_size=dimIn, hidden_size=dimRNN)
    else:
        Units = None
    return Units


def get_DenseNet(dimIn, dimHiddens, dimOut=None):
    """
    return a dense network with the given params.
    :param dimIn:
    :param dimHiddens:
    :return:
    """
    Net = nn.Sequential()
    dims = [dimIn] + dimHiddens
    for i in range(1, len(dims)):
        Net.add_module('Dense-{}'.format(i), nn.Linear(dims[i - 1], dims[i]))
        Net.add_module('Tanh-{}'.format(i), nn.Tanh())
    if dimOut:
        Net.add_module('Output', nn.Linear(dims[-1], dimOut))
        Net.add_module('Softplus', nn.Softplus())
    return Net


class OpticNet(nn.Module):
    """
    A multi-modal sequential network that compute the incremental of propagation parameters.
    """

    def __init__(self,
                 dimEmbed,
                 dimNameRNN,
                 dimProp,
                 dimArray,
                 dimCNN_channels,
                 dimCNN_kernels,
                 dimCNN_denses,
                 dimOutput_denses = None,
                 dimPropRNN = None,
                 CNN_residual=True,
                 RNNUnitType='GRU',
                 optics_entities=('Lens', 'CRL', 'Zone_Plate', 'Fiber', 'Aperture', 'Obstacle',
                                  'Sample', 'Planar', 'Circular_Cylinder', 'Elliptical_Cylinder',
                                  'Toroid', 'Crystal', 'Grating', 'Watchpoint', 'Drift')
                 ):
        """

        :param dimEmbed:
        :param dimNameRNN:
        :param dimProp:
        :param dimArray:
        :param dimCNN_channels:
        :param dimCNN_kernels:
        :param dimCNN_denses:
        :param dimOutput_denses:
        :param dimPropRNN:
        :param CNN_residual:
        :param RNNUnitType:
        :param optics_entities:
        """
        super(OpticNet, self).__init__()
        self.optics_entities = optics_entities
        self.IntensityCNN = ResConvNet(input_size=dimArray, in_channels=1, hidden_channels=dimCNN_channels,
                                       kernel_sizes=dimCNN_kernels, dim_denses=dimCNN_denses, residual=CNN_residual)
        self.TextRnn = NameEntityRNN(dimEmbed, dimNameRNN, optics_entities, RNNUnitType)
        dimPropRNN = dimPropRNN if dimPropRNN else dimNameRNN
        dimOutput_denses = dimOutput_denses if dimOutput_denses else dimCNN_denses
        self.PropRNN = get_RNN(dimProp, dimPropRNN, RNNUnitType)
        self.OutputNet = get_DenseNet(dimNameRNN + dimPropRNN + dimCNN_denses[-1], dimOutput_denses, dimProp)
        # record the configuration.
        self.config = {"dimEmbed": dimEmbed, "dimNameRNN": dimNameRNN, "dimProp": dimProp, "dimArray": dimArray,
                       "dimCNN_channels": dimCNN_channels, "dimCNN_kernels": dimCNN_kernels,
                       "dimCNN_denses": dimCNN_denses, "dimOutput_denses": dimOutput_denses,
                       "dimPropRNN": dimPropRNN, "CNN_residual": CNN_residual, "RNNUnitType": RNNUnitType}
        return

    def forward(self, Names, Props, Arrays):
        Arrays = (Arrays - Arrays.mean()) / (Arrays.std() + 1e-4)
        Array_features = self.IntensityCNN(Arrays).mean(0, keepdim=True)
        Names_features = self.TextRnn(Names)[0]
        Props_features = self.PropRNN(Props)[0]
        MultiModal_features = torch.cat([Array_features.unsqueeze(0).repeat(len(Names_features), Props.size()[1], 1),
                                         Names_features.repeat(1, Props.size()[1], 1),
                                         Props_features],
                                        dim=-1)
        delta_Prop = torch.exp(self.OutputNet(MultiModal_features))
        return Props + delta_Prop, delta_Prop



class OpticNet_DDPG(OpticNet):
    """
    A multi-modal sequential network that compute the incremental of propagation parameters. (which components
    required to train by Deep Deterministic Policy Gradient (DDPG))
    """

    def __init__(self,
                 dimEmbed,
                 dimNameRNN,
                 dimProp,
                 dimArray,
                 dimCNN_channels,
                 dimCNN_kernels,
                 dimCNN_denses,
                 dimOutput_denses = None,
                 dimPropRNN = None,
                 CNN_residual=True,
                 RNNUnitType='GRU',
                 optics_entities=('Lens', 'CRL', 'Zone_Plate', 'Fiber', 'Aperture', 'Obstacle',
                                  'Sample', 'Planar', 'Circular_Cylinder', 'Elliptical_Cylinder',
                                  'Toroid', 'Crystal', 'Grating', 'Watchpoint', 'Drift')
                 ):
        """

        :param dimEmbed:
        :param dimNameRNN:
        :param dimProp:
        :param dimArray:
        :param dimCNN_channels:
        :param dimCNN_kernels:
        :param dimCNN_denses:
        :param dimOutput_denses:
        :param dimPropRNN:
        :param CNN_residual:
        :param RNNUnitType:
        :param optics_entities:
        """
        super(OpticNet_DDPG, self).__init__(dimEmbed, dimNameRNN, dimProp, dimArray, dimCNN_channels, dimCNN_kernels,
                 dimCNN_denses, dimOutput_denses, dimPropRNN, CNN_residual, RNNUnitType, optics_entities)
        # extra components for DDPG.
        dimPropRNN = dimPropRNN if dimPropRNN else dimNameRNN
        dimOutput_denses = dimOutput_denses if dimOutput_denses else dimCNN_denses
        self.CriticNet = get_DenseNet(dimNameRNN + dimPropRNN + dimCNN_denses[-1] + dimProp, dimOutput_denses, 1)
        # split the parameter groups.
        self.actor_params = nn.ParameterList()
        self.actor_params.extend(self.OutputNet.parameters())
        self.actor_params.extend(self.IntensityCNN.parameters())
        self.actor_params.extend(self.PropRNN.parameters())
        self.actor_params.extend(self.TextRnn.parameters())
        self.critic_params = nn.ParameterList(self.CriticNet.parameters())
        return

    def forward(self, Names, Props, Arrays, delta_Prop_input=None):
        """

        :param Names:
        :param Props:
        :param Arrays:
        :return:
        """
        Array_features = self.IntensityCNN(Arrays).mean(0, keepdim=True)
        Names_features = self.TextRnn(Names)[0]
        Props_features = self.PropRNN(Props)[0]
        MultiModal_features = torch.cat([Array_features.unsqueeze(0).repeat(len(Names_features), Props.size()[1], 1),
                                         Names_features.repeat(1, Props.size()[1], 1),
                                         Props_features],
                                        dim=-1)
        delta_Prop = self.OutputNet(MultiModal_features)
        if delta_Prop_input is not None:
            critic = torch.exp(self.CriticNet(torch.cat([MultiModal_features, delta_Prop_input], dim=-1))).mean()
            return Props + delta_Prop, delta_Prop, critic
        else:
            return Props + delta_Prop, delta_Prop
