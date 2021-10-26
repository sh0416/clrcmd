"""Sentence bert"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sentsim.models.base import SemanticTextualSimilarityModel


class PytorchSemanticTextualSimilarityModel(SemanticTextualSimilarityModel):
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def predict(self, inputs: List[Tuple[str, str]]) -> np.ndarray:
        sent1 = self.tokenizer(
            [x[0] for x in inputs], padding=True, return_tensors="pt"
        )
        sent2 = self.tokenizer(
            [x[1] for x in inputs], padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        sent1 = {k: v.to(device) for k, v in sent1.items()}
        sent2 = {k: v.to(device) for k, v in sent2.items()}
        with torch.no_grad():
            return self.model(sent1, sent2).cpu().numpy()
