import itertools
from typing import Iterable, List

import numpy as np
import torch
from torch import Tensor

from sentence_benchmark.data import Example


def cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    x = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.nan_to_num(x)


def batch(dataset: List[Example], batch_size: int) -> Iterable:
    args = [iter(dataset)] * batch_size
    for batch in itertools.zip_longest(*args, fillvalue=None):
        yield tuple(filter(lambda x: x is not None, batch))


def masked_sum(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Sum (masked version)

    :param x: Input
    :param mask: Mask Could be broadcastable
    :param dim:
    :return: Result of sum
    """
    return torch.sum(x * mask.float(), dim=dim)


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Mean (masked version)

    :param x: Input
    :param mask: Mask Could be broadcastable
    :param dim:
    :return: Result of mean
    """
    return masked_sum(x, mask, dim) / torch.sum(mask.float(), dim)
