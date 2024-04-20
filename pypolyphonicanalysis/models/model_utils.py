import torch
from torch import nn


def Conv(in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], depthwise_separabile: bool = False, padding: str | int = "same", bias: bool = True) -> nn.Module:
    if not depthwise_separabile or out_channels % in_channels != 0:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size, groups=in_channels, padding=padding, bias=bias)


class SelfAttention(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self._n_channels = n_channels
        self._query = self._conv(n_channels, n_channels // 8)
        self._key = self._conv(n_channels, n_channels // 8)
        self._value = self._conv(n_channels, n_channels // 2)
        self._out = self._conv(n_channels // 2, n_channels)
        self._gamma = nn.Parameter(torch.Tensor([0.0]))

    def _conv(self, n_in: int, n_out: int) -> nn.Module:
        return torch.nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_ftrs = x.shape[2] * x.shape[3]
        f = self._query(x).view(-1, self._n_channels // 8, n_ftrs)
        g = torch.max_pool2d(self._key(x), [2, 2]).view(-1, self._n_channels // 8, n_ftrs // 4)
        h = torch.max_pool2d(self._value(x), [2, 2]).view(-1, self._n_channels // 2, n_ftrs // 4)
        beta = torch.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        o = self._out(torch.bmm(h, beta.transpose(1, 2)).view(-1, self._n_channels // 2, x.shape[2], x.shape[3]))
        output: torch.Tensor = self._gamma * o + x
        return output


class ResidualCNNBlockModule(nn.Module):
    def __init__(self, input_channels: int, channels: int, kernel_sizes: list[int | tuple[int, int]], use_depthwise_separable_convolutions_when_possible: bool) -> None:
        super().__init__()
        depthwise = use_depthwise_separable_convolutions_when_possible
        self._residual_connection_on_first_block = input_channels == channels
        self._convs = nn.ModuleList([Conv(input_channels, channels, 5, depthwise)] + [Conv(channels, channels, kernel_sizes[idx], depthwise) for idx in range(len(kernel_sizes))])
        self._activations = nn.ModuleList([nn.ReLU()] + [nn.ReLU() for _ in range(len(kernel_sizes))])
        self._batchnorms = nn.ModuleList([nn.BatchNorm2d(channels)] + [nn.BatchNorm2d(channels) for _ in range(len(kernel_sizes))])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        for idx, (conv, activation, batchnorm) in enumerate(zip(self._convs, self._activations, self._batchnorms)):
            inp = out
            out = conv(out)
            out = activation(out)
            out = batchnorm(out)
            if idx != 0 or self._residual_connection_on_first_block:
                out += inp
        return out


# UNet primitives and module adapted from:
# https://github.com/milesial/Pytorch-UNet/tree/master


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depthwise: bool) -> None:
        super().__init__()
        self._double_conv = nn.Sequential(
            Conv(in_channels, out_channels, 3, depthwise, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, 3, depthwise, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        double_conv_out: torch.Tensor = self._double_conv(x)
        return double_conv_out


class Down(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, depthwise: bool) -> None:
        super().__init__()
        self._maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, depthwise))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maxpool_conv_out: torch.Tensor = self._maxpool_conv(x)
        return maxpool_conv_out


class Up(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, depthwise: bool) -> None:
        super().__init__()
        self._up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self._conv = DoubleConv(in_channels, out_channels, depthwise)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self._up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        conv_out: torch.Tensor = self._conv(x)
        return conv_out


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depthwise: bool) -> None:
        super(UNet, self).__init__()
        self._inc = DoubleConv(in_channels, 64, depthwise)
        self._down1 = Down(64, 128, depthwise)
        self._down2 = Down(128, 256, depthwise)
        self._down3 = Down(256, 512, depthwise)
        self._down4 = Down(512, 1024, depthwise)
        self._up1 = Up(1024, 512, depthwise)
        self._up2 = Up(512, 256, depthwise)
        self._up3 = Up(256, 128, depthwise)
        self._up4 = Up(128, 64, depthwise)
        self._out = Conv(64, out_channels, 1, depthwise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._inc(x)
        x2 = self._down1(x1)
        x3 = self._down2(x2)
        x4 = self._down3(x3)
        x5 = self._down4(x4)
        x = self._up1(x5, x4)
        x = self._up2(x, x3)
        x = self._up3(x, x2)
        x = self._up4(x, x1)
        logits: torch.Tensor = self._out(x)
        return logits
