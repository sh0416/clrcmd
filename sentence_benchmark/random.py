from typing import Any, Dict, List

import numpy as np

from sentence_benchmark.data import Input


def prepare(inputs: List[Input]) -> Dict:
    return {}


def batcher(inputs: List[Input], param: Any) -> np.ndarray:
    return np.random.rand(len(inputs))
