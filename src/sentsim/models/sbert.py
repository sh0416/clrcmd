"""Sentence bert"""
from typing import List, Tuple

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from sentsim.models.base import SemanticTextualSimilarityModel
from sentsim.utils import cos, masked_mean


class SentenceBert(SemanticTextualSimilarityModel):
    def __init__(self, model_name_or_path: str, pooler_type: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.pooler_type = pooler_type

    def predict(self, inputs: List[Tuple[str, str]]) -> np.ndarray:
        sentences = [x[0] for x in inputs] + [x[1] for x in inputs]
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_outputs = self.model(**encoded_inputs)[0]
            if self.pooler_type == "cls":
                sentence_embeddings = model_outputs[:, 0]
            elif self.pooler_type == "avg":
                sentence_embeddings = masked_mean(
                    model_outputs, encoded_inputs["attention_mask"].unsqueeze(-1), dim=1
                )
            else:
                raise AttributeError()
            x1 = sentence_embeddings[: len(inputs)].numpy()
            x2 = sentence_embeddings[len(inputs) :].numpy()
        return np.asarray(list(map(cos, x1, x2)))
