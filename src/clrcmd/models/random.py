from typing import List, Tuple

import numpy as np

from sentsim.models.base import SemanticTextualSimilarityModel


class RandomSimilarityModel(SemanticTextualSimilarityModel):
    def predict(self, inputs: List[Tuple[str, str]]) -> np.ndarray:
        return np.random.rand(len(inputs))
