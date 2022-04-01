from typing import Literal, Optional

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


def skip_connection_block(
        in_channels: int,
        out_channels: int,
        downsample: int,
        kernel_size: int,
) -> nn.Module:
    block = nn.Sequential(
        nn.MaxPool1d(kernel_size, stride=downsample, padding=kernel_size // 2),
        nn.Conv1d(in_channels, out_channels, kernel_size=1),
    )

    return block


def straight_connection_block(
        in_channels: int,
        out_channels: int,
        downsample: int,
        kernel_size: int,
        dropout_prob: float = 0.5,
        activataion: activation_funcs = "relu",
) -> nn.Module:
    block = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        activation_func(activataion),
        nn.Dropout(dropout_prob),
        nn.Conv1d(out_channels, out_channels, kernel_size, downsample, kernel_size // 2)
    )

    return block


def outer_block(
        in_channels: int,
        activation: activation_funcs,
        dropout_prob: float = 0.5
) -> nn.Module:
    block = nn.Sequential(
        nn.BatchNorm1d(in_channels),
        activation_func(activation),
        nn.Dropout(dropout_prob),
    )

    return block


class ResBlk(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: int,
            kernel_size: int,
            dropout_prob: float,
            activation: activation_funcs,
            is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.activate = activation_func(activation)

        blocks: list[Optional[nn.Module]] = [
            None if is_first else outer_block(
                self.in_channels,
                self.activation,
                dropout_prob
            ),
            straight_connection_block(
                self.in_channels,
                self.out_channels,
                self.downsample,
                self.kernel_size,
                self.dropout_prob,
            )
        ]

        self.straight = nn.Sequential(
            *filter(None, blocks)
        )

        self.skip_connection_block = skip_connection_block(
                self.in_channels,
                self.out_channels,
                self.downsample,
                self.kernel_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_connection_block(x) + self.straight(x)


class Encoder(nn.Module):
    def __init__(
            self,
            channels_progression: list[int],
            downsamples: list[int],
            kernels_sizes: list[int],
            dropout_probs: list[float],
            outer_dropout_prob: float = 0.3,
            activation: activation_funcs = "relu",
    ) -> None:
        super().__init__()
        self.activation: activation_funcs = activation
        self.last_out_channels = channels_progression[-1]
        self.outer_dropout_prob = outer_dropout_prob

        in_channels_lst = [channels for channels in channels_progression[:-1]]
        out_channels_lst = [channels for channels in channels_progression[1:]]

        lengths = {
            len(in_channels_lst),
            len(out_channels_lst),
            len(downsamples),
            len(kernels_sizes),
            len(dropout_probs)
        }

        if len(lengths) != 1:
            raise ValueError("Mismatching params sizes!")

        is_first = True
        blocks: list[nn.Module] = []

        for (in_channels, out_channels, downsample, dropout_prob, kernel_size) in zip(
            in_channels_lst, out_channels_lst, downsamples, dropout_probs, kernels_sizes
        ):
            blocks.append(ResBlk(
                in_channels,
                out_channels,
                downsample,
                kernel_size,
                dropout_prob,
                activation,
                is_first,
            ))

            is_first = False

        self.encoder = nn.Sequential(
            *blocks,
            outer_block(
                self.last_out_channels,
                self.activation,
                self.outer_dropout_prob,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x, __ = torch.max(x, 2)  # Global max pooling
        return x


class ResidualConvNet(Encoder):
    def __init__(
            self,
            channels_progression: list[int],
            downsamples: list[int],
            kernels_sizes: list[int],
            dropout_probs: list[float],
            linear_layers_sizes: list[int],
            num_classes: int,
            outer_dropout_prob: float = 0.3,
            activation: activation_funcs = "relu",
    ) -> None:
        super().__init__(
            channels_progression,
            downsamples,
            kernels_sizes,
            dropout_probs,
            outer_dropout_prob,
            activation,
        )
        linear_layers_sizes = [self.last_out_channels] + linear_layers_sizes
        in_sizes = [sizes for sizes in linear_layers_sizes[:-1]]
        out_sizes = [sizes for sizes in linear_layers_sizes[1:]]

        lin_blocks: list[nn.Module] = []

        for in_size, out_size in zip(in_sizes, out_sizes):
            lin_blocks.append(lin_block(in_size, out_size))
            lin_blocks.append(activation_func(self.activation))

        lin_blocks.append(lin_block(out_sizes[-1], num_classes))

        self.decoder = nn.Sequential(
            *lin_blocks,
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.decoder(x)

        return x

