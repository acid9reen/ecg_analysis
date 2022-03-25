from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

activation_funcs = Literal["relu", "leaky_relu", "selu", "none"]


# TODO: Does it need to be inplace?
def activation_func(activation: activation_funcs):
    return nn.ModuleDict(
        {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01, inplace=True),
            "selu": nn.SELU(inplace=True),
            "none": nn.Identity(),
        }
    )[activation]


class Conv1dAuto(nn.Conv1d):
    """Auto padding tweak to pytorch Conv1d"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,)


def conv_block(
        input_size: int,
        output_size: int,
        *,
        kernel_size: int,
        activation: activation_funcs = "relu",
        dropout_p: float=0.5,
) -> nn.Sequential:
    block = nn.Sequential(
        Conv1dAuto(
            input_size,
            output_size,
            kernel_size=kernel_size
        ),
        nn.BatchNorm1d(output_size),
        activation_func(activation),
        nn.Dropout(dropout_p),
        nn.MaxPool1d(2),
    )

    return block


def lin_block(input_size: int, output_size: int) -> nn.Sequential:
    block = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.BatchNorm1d(output_size),
    )

    return block


class SimpleConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = conv_block(12, 16, kernel_size=5)
        self.conv_2 = conv_block(16, 32, kernel_size=3)
        self.conv_3 = conv_block(32, 64, kernel_size=3)
        self.conv_4 = conv_block(64, 256, kernel_size=3)

        self.ln_1 = lin_block((1000 // pow(2, 4)) * 256, 128)
        self.ln_2 = lin_block(128, 64)
        self.ln_3 = lin_block(64, 64)
        self.ln_4 = lin_block(64, 44)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave = self.conv_1(wave)
        wave = self.conv_2(wave)
        wave = self.conv_3(wave)
        wave = self.conv_4(wave)

        wave = torch.flatten(wave, 1)

        wave = F.relu(self.ln_1(wave))
        wave = F.relu(self.ln_2(wave))
        wave = F.relu(self.ln_3(wave))
        wave = self.ln_4(wave)

        return torch.sigmoid(wave)
