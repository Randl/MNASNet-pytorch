import torch
import torch.nn as nn
from torch.nn import init

debug_global = False
default_activation = nn.ReLU


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBlock(nn.Module):
    def __init__(self,
                 in_,
                 out_,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 activation=default_activation,
                 momentum=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_,
                              out_,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_)
        self.activation = activation(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SepConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 reduce=False,
                 repeat=0):
        super(SepConv, self).__init__()

        padding = kernel_size // 2
        stride = 2 if reduce else 1

        self.sequence = [ConvBlock(in_=in_channels,
                                   out_=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels),
                         ConvBlock(in_=in_channels,
                                   out_=in_channels,
                                   kernel_size=1,
                                   stride=1)] * repeat + \
                        [ConvBlock(in_=in_channels,
                                   out_=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels),
                         ConvBlock(in_=in_channels,
                                   out_=out_channels,
                                   kernel_size=1,
                                   stride=1)]

        self.sequence = nn.Sequential(*self.sequence)

    def forward(self, input):
        output = self.sequence(input)
        if debug_global:
            print(output.shape)
        return output


class MBConv_block(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_factor,
                 kernel_size=3,
                 ):
        super(MBConv_block, self).__init__()

        self.in_channels = in_channels
        padding = kernel_size // 2

        self.sequence = nn.Sequential(ConvBlock(in_=in_channels,
                                                out_=in_channels * channel_factor,
                                                kernel_size=1,
                                                stride=1),
                                      ConvBlock(in_=in_channels * channel_factor,
                                                out_=in_channels * channel_factor,
                                                kernel_size=kernel_size,
                                                stride=1,
                                                padding=padding,
                                                groups=in_channels * channel_factor),
                                      ConvBlock(in_=in_channels * channel_factor,
                                                out_=in_channels,
                                                kernel_size=1,
                                                stride=1))

    def forward(self, input):
        output = input + self.sequence(input)

        if debug_global:
            print(output.shape)
        return output


class MBConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 channel_factor,
                 layers,
                 kernel_size=3,
                 reduce=True,
                 cut_channels_first=True):
        super(MBConv, self).__init__()

        if cut_channels_first:
            block_channels = out_channels
        else:
            block_channels = in_channels

        stride = 2 if reduce else 1

        self.sequence = [ConvBlock(in_=in_channels,
                                   out_=out_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1)] + \
                        [MBConv_block(block_channels,
                                      channel_factor,
                                      kernel_size)] * layers
        if cut_channels_first:
            self.sequence = nn.Sequential(*self.sequence)
        else:
            self.sequence = nn.Sequential(*list(reversed(self.sequence)))

    def forward(self, input):
        output = self.sequence(input)
        return output


class Mnasnet(nn.Module):
    def __init__(self, num_classes=1000, m=1, cut_channels_first=True):
        super(Mnasnet, self).__init__()

        self.features = nn.Sequential(ConvBlock(3, 32 * m, kernel_size=3, stride=2, padding=1),
                                      SepConv(32 * m, 16 * m, kernel_size=3),
                                      MBConv(16 * m, 24 * m, channel_factor=3, layers=3, kernel_size=3, reduce=True,
                                             cut_channels_first=cut_channels_first),
                                      MBConv(24 * m, 40 * m, channel_factor=3, layers=3, kernel_size=5, reduce=True,
                                             cut_channels_first=cut_channels_first),
                                      MBConv(40 * m, 80 * m, channel_factor=6, layers=3, kernel_size=5, reduce=True,
                                             cut_channels_first=cut_channels_first),
                                      MBConv(80 * m, 96 * m, channel_factor=6, layers=2, kernel_size=3, reduce=False,
                                             cut_channels_first=cut_channels_first),
                                      MBConv(96 * m, 192 * m, channel_factor=6, layers=4, kernel_size=5, reduce=True,
                                             cut_channels_first=cut_channels_first),
                                      MBConv(192 * m, 320 * m, channel_factor=6, layers=1, kernel_size=3, reduce=False,
                                             cut_channels_first=cut_channels_first)
                                      )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(320 * m, num_classes)

        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.features(input)
        output = self.fc(self.avgpool(output).view(input.size(0), -1))
        return output


if __name__ == "__main__":
    """Testing
    """
    model1 = Mnasnet()
    print(model1)
    x = torch.randn(2, 3, 224, 224)
    print(model1(x))
