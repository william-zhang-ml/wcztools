"""
Miscellaneous processing blocks.
This module is mainly for ideas I imagine (versus from literature).
"""
from typing import List
import torch
from torch import nn
from .conv import ConvBlock


class SoftSpatialMultiplex(nn.Module):
    """
    Block to multiplex b/w different feature maps at each spatial location.
    The multiplexing is ~soft~ and implemented as a weighted sum.
    The weights are input-dependent (think attention mechanism).
    """
    def __init__(self,
                 n_inputs: int,
                 in_channels: int) -> None:
        """
        Constructor. Set up attention layers to compute weights for sum.

        Args:
            n_inputs:    number of inputs to multiplex
            in_channels: number of neurons/channels per input
        """
        super().__init__()
        self.attn = nn.Sequential(
            ConvBlock(
                in_channels=n_inputs * in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            ConvBlock(
                in_channels=in_channels,
                out_channels=n_inputs,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Softmax()
        )

    def forward(self, inp: List[torch.Tensor]) -> torch.Tensor:
        """
        Multiplex between different feature maps at each spatial location.

        Args:
            inp: input feature maps to multiplex

        Returns: multiplexed feature map
        """
        # pylint: disable=invalid-name
        B, _, H, W = inp[0].shape
        K = len(inp)
        # pylint: enable=invalid-name
        # pylint: disable=no-member
        stacked = torch.stack(inp, dim=1)  # B x K x C x H x W
        # pylint: enable=no-member
        plex_weights = self.attn(stacked.view(B, -1, H, W)).view(B, K, 1, H, W)

        # weigh and sum
        outp = plex_weights * stacked
        outp = outp.mean(dim=1)
        return outp
