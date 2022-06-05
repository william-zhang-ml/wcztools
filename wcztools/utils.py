""" Various data munging functions. """
from typing import Dict, List
import torch
from torchvision.ops import box_area, masks_to_boxes


def draw_box_around(tens: torch.Tensor,
                    queries=List[int]) -> Dict[int, torch.Tensor]:
    """ Compute bounding boxes around specific values in a integer tensor.

        Args:
            tens:    2-dimension tensor
            queries: tensor values of interest

        Returns: (x1, y1, x2, y2) extents for each query
    """
    queries = queries or tens.unique()
    # pylint: disable=no-member
    masks = torch.zeros(len(queries), *tens.shape)
    # pylint: enable=no-member
    for i_query, curr_query in enumerate(queries):
        masks[i_query] = tens == curr_query
        assert masks[i_query].sum() > 0, f'cannot find query {curr_query}'
    boxes = masks_to_boxes(masks)
    return dict(zip(queries, boxes))


def target_area_covered(inputs: torch.Tensor,
                        targets: torch.Tensor) -> torch.Tensor:
    """
    Compute fraction of target box that each input box covers.

    Args:
        inputs:  N x (x1, y1, x2, y2) input box
        targets: M x (x1, y1, x2, y2) target box

    Returns: M x N fraction of the mth target that the nth input covers
    """
    targ_area = box_area(targets)[:, None]  # M x 1

    # intersection area
    # pylint: disable=no-member
    lower = torch.max(targets[:, None, :2], inputs[None, :, :2])
    upper = torch.min(targets[:, None, 2:], inputs[None, :, 2:])
    # pylint: enable=no-member
    width_height = (upper - lower).clamp(0)  # M x N x 2
    inter_area = width_height.prod(dim=-1)   # M x N

    return inter_area / targ_area
