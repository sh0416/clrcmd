import itertools
from typing import Iterable, List

import numpy as np


def cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    x = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.nan_to_num(x)


def batch(dataset: List, batch_size: int) -> Iterable:
    args = [iter(dataset)] * batch_size
    for batch in itertools.zip_longest(*args, fillvalue=None):
        yield tuple(filter(lambda x: x is not None, batch))
