"""Bag of word model"""
import itertools
from collections import Counter
from typing import Dict, List

import numpy as np

from sentence_benchmark.data import Input
from sentence_benchmark.utils import cos


def prepare(inputs: List[Input], param: Dict) -> Dict:
    sentences = itertools.chain.from_iterable(inputs)
    tokens = itertools.chain.from_iterable(map(lambda x: x.split(), sentences))
    words = Counter(tokens)

    # Load word2vec
    fasttext_path = param.get("fasttext_path", ".data/fasttext/crawl-300d-2M.vec")
    word2vec = {}
    with open(fasttext_path) as f:
        for line in f:
            word, vec = line.split(" ", 1)
            if word in words:
                word2vec[word] = np.fromstring(vec, sep=" ")
    param["word2vec"] = word2vec
    return param


def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
    def _compute(x: List[str]) -> np.ndarray:
        x = map(param["word2vec"].get, x.split())
        x = filter(lambda x: x is not None, x)
        x = list(x)
        x = np.mean(np.stack(list(x), axis=0), axis=0)
        assert x.shape[0] == 300
        return x

    x = map(lambda x: (_compute(x[0]), _compute(x[1])), inputs)
    x = map(lambda x: cos(x[0], x[1]), x)
    x = list(x)
    return np.asarray(x)
