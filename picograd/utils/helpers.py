from typing import Tuple

import random
import torch
import numpy as np


def fix_seeds(k, with_cuda=True):
    """ Set seeds for all commonly used sources of randomness """

    random.seed(k)
    np.random.seed(k)
    torch.manual_seed(k)
    if with_cuda:
        torch.cuda.manual_seed(k)


def unravel_index(
    indices: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)
