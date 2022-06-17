""" Loss functions for training neural networks."""
from typing import Tuple
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss


# pylint: disable=too-few-public-methods
class YoloLoss:
    """ Loss function for training Yolo or Yolo-like object detectors.

        You Only Look Once: Unified, Real-Time Object Detection
        https://arxiv.org/pdf/1506.02640.pdf

        Focal Loss for Dense Object Detection
        https://arxiv.org/pdf/1708.02002.pdf
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 use_focal_on_det: bool = True,
                 alpha: float = 0.25,
                 gamma: float = 2,
                 use_cross_on_label: bool = False,
                 reduction: str = 'mean') -> None:
        """
        Constructor. Save loss function hyperparameters.

        Args:
            use_focal_on_det:   toggle loss func applied to detection
                                (focal loss vs binary cross entropy)
            alpha:              1-vs-0 det weight if using focal loss
            gamma:              focal strength if using focal loss
            use_cross_on_label: toggle loss func applied to classification
                                (cross entropy vs binary cross entropy)
            reduction:          how to report loss
                                * mean -> mean across all elements
                                * none -> no reduction
        """
        self.use_focal_on_det = use_focal_on_det
        self.alpha = alpha if use_focal_on_det else None
        self.gamma = gamma if use_focal_on_det else None
        self.use_cross_on_label = use_cross_on_label
        self.reduction = reduction
    # pylint: enable=too-many-arguments

    def __call__(self,
                 pred: Tensor,
                 target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute Yolo object detection loss.

        Args:
            pred:   detector predictions (batch x (5 + numlabel) x H x W)
            target: detector predictions (batch x 6 x H x W)

        Returns: each loss component ... detection, regression, classification
        """
        # batchsize x (5 + numlabel) x HW, batch x 6 x HW
        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target = target.view(target.shape[0], target.shape[1], -1)

        # detection loss
        if self.use_focal_on_det:
            # normal BCE puts "1" weight on 0/1 labels, x2 here to match that
            det_loss = 2 * sigmoid_focal_loss(
                inputs=pred[:, 0],
                targets=target[:, 0],
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction)
        else:
            det_loss = F.binary_cross_entropy_with_logits(
                input=pred[:, 0],
                target=target[:, 0])

        # regression loss
        mask = target[:, :1] == 1  # ignore regression for background
        reg_loss = F.mse_loss(mask * pred[:, 1:5], mask * target[:, 1:5])

        # classification loss
        if self.use_cross_on_label:
            label_loss = F.cross_entropy(
                mask * pred[:, 5:],
                (mask * target[:, 5:]).long().squeeze(),
                reduction=self.reduction)
        else:
            one_hot = F.one_hot(
                target[:, 5].long(),
                num_classes=pred.shape[1] - 5
            ).transpose(1, 2).float()
            label_loss = F.binary_cross_entropy_with_logits(
                mask * pred[:, 5:],
                (mask * one_hot).float(),
                reduction=self.reduction)

        return det_loss, reg_loss, label_loss
# pylint: enable=too-few-public-methods
