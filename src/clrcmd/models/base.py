import abc
from typing import List, Tuple

import numpy as np


class SemanticTextualSimilarityModel:
    @abc.abstractmethod
    def predict(self, inputs: List[Tuple[str, str]]) -> np.ndarray:
        pass
