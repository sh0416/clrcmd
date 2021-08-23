from typing import Dict, List

import numpy as np

from sentence_benchmark.data import Input


def prepare(inputs: List[Input], param: Dict) -> Dict:
    return param


def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
    return np.random.rand(len(inputs))
