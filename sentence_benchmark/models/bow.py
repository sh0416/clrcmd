"""Bag of word model"""
import itertools
from collections import Counter
from typing import Dict, List

import numpy as np

from sentence_benchmark.data import Input
from sentence_benchmark.utils import cos

FASTTEXT_PATH = ".data/fasttext/crawl-300d-2M.vec"


def prepare(inputs: List[Input], param: Dict) -> Dict:
    stream1 = itertools.chain.from_iterable((x.text1 for x in inputs))
    stream2 = itertools.chain.from_iterable((x.text2 for x in inputs))
    words = Counter(itertools.chain(stream1, stream2))

    # Load word2vec
    word2vec = {}
    with open(FASTTEXT_PATH) as f:
        for line in f:
            word, vec = line.split(" ", 1)
            if word in words:
                word2vec[word] = np.fromstring(vec, sep=" ")
    param["word2vec"] = word2vec
    return param


def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
    def _compute(x: List[str]) -> np.ndarray:
        x = map(param["word2vec"].get, x)
        x = filter(lambda x: x is not None, x)
        x = list(x)
        x = np.mean(np.stack(list(x), axis=0), axis=0)
        assert x.shape[0] == 300
        return x

    x = map(lambda x: (_compute(x.text1), _compute(x.text2)), inputs)
    x = map(lambda x: cos(x[0], x[1]), x)
    x = list(x)
    return np.asarray(x)
