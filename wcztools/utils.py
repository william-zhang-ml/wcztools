""" Various data munging functions. """
from typing import Dict, List
import torch
from torchvision.ops import masks_to_boxes


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
