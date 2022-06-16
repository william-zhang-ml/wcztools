"""
This code supports feature pyramid networks and similar methods.
This code implements both what's in papers as well as my modified versions.


References:
[1] Feature Pyramid Networks for Object Detection
    https://arxiv.org/pdf/1612.03144.pdf
"""
import torch
from torch import nn
from torch.nn import functional as F


class FeaturePyramidLink(nn.Module):
    """
    Fusion block for top-down lower res. features and higher res. features.

    Feature Pyramid Networks for Object Detection
    https://arxiv.org/pdf/1612.03144.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        """
        Constructor. Set up lateral projection and post-fusion filter.

        Args:
            in_channels:  number of higher resolution channels
            out_channels: desired number of top-down path channels
        """
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.filter = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self,
                lowres: torch.Tensor,
                highres) -> torch.Tensor:
        """
        Fuse features for higher resolution multi-scale feature map.

        Args:
            lowres:  top-down feature map w/abstract lower resolution features,
                     (Cout x H x W)
            highres: lateral feature map w/higher resolution features,
                     (Cin x 2H x 2W)

        Returns: higher resolution multi-scale feature map, (Cout x 2H x 2W)
        """
        upsamped = F.interpolate(lowres, scale_factor=2)
        proj = self.projection(highres)
        fused = upsamped + proj
        return self.filter(fused)
