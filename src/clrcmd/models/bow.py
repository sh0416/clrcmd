"""Bag of word model"""
import itertools
from collections import Counter
from typing import List, Tuple

import numpy as np

from sentsim.models.base import SemanticTextualSimilarityModel
from sentsim.utils import cos


class BagOfWord(SemanticTextualSimilarityModel):
    def __init__(self, word2vec_path: str, corpus: List[str]):
        # Collect used words
        words = Counter(itertools.chain.from_iterable(map(lambda x: x.split(), corpus)))

        # Load pretrained vector for used word
        word2vec = {}
        with open(word2vec_path) as f:
            for line in f:
                word, vec = line.split(" ", 1)
                if word in words:
                    word2vec[word] = np.fromstring(vec, sep=" ")
        self.word2vec = word2vec

    def predict(self, inputs: List[Tuple[str, str]]) -> np.ndarray:
        def _compute(x: List[str]) -> np.ndarray:
            x = [word for word in map(self.word2vec.get, x.split()) if word is not None]
            return np.mean(np.stack(x, axis=0), axis=0)

        return np.asarray([cos(_compute(x1), _compute(x2)) for x1, x2 in inputs])
