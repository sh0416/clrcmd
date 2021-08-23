"""Sentence bert"""
from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

from sentence_benchmark.data import Input
from sentence_benchmark.utils import cos


def prepare(inputs: List[Input]) -> Any:
    model = SentenceTransformer("nli-roberta-base-v2")
    return {"model": model}


def batcher(inputs: List[Input], param: Any) -> np.ndarray:
    model = param["model"]
    x1 = model.encode(
        [" ".join(x.text1) for x in inputs], 4, show_progress_bar=False
    )
    x2 = model.encode(
        [" ".join(x.text2) for x in inputs], 4, show_progress_bar=False
    )
    x = list(map(lambda x: cos(x[0], x[1]), zip(x1, x2)))
    return np.asarray(x)
