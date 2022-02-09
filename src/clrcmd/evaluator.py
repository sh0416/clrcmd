"""Evaluate model using given dataset"""
import itertools
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats.stats import pearsonr, spearmanr

from sentsim.data.sts import SemanticTextualSimilarityDataset
from sentsim.models.base import SemanticTextualSimilarityModel


def batch(examples: List[Any], batch_size: int) -> Iterable:
    args = [iter(examples)] * batch_size
    for batch in itertools.zip_longest(*args, fillvalue=None):
        yield tuple(filter(lambda x: x is not None, batch))


class SemanticTextualSimilarityEvaluator:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def evaluate(
        self,
        model: SemanticTextualSimilarityModel,
        dataset: SemanticTextualSimilarityDataset,
    ) -> Dict[str, float]:
        results, scores_all, labels_all = {}, [], []
        for source_name, source_examples in dataset.items():
            scores, labels = [], []
            for examples in batch(source_examples, self.batch_size):
                scores.append(model.predict([input_pair for input_pair, _ in examples]))
                labels.append([label for _, label in examples])
            scores, labels = np.concatenate(scores), np.concatenate(labels)
            results[f"{source_name}_pearson"] = pearsonr(scores, labels)[0]
            results[f"{source_name}_spearman"] = spearmanr(scores, labels)[0]
            scores_all.append(scores)
            labels_all.append(labels)
        scores_all, labels_all = np.concatenate(scores_all), np.concatenate(labels_all)
        results[f"all_pearson_all"] = pearsonr(scores_all, labels_all)[0]
        results[f"all_spearman_all"] = spearmanr(scores_all, labels_all)[0]

        pearsons = [results[f"{source}_pearson"] for source in dataset.keys()]
        spearmans = [results[f"{source}_spearman"] for source in dataset.keys()]
        results[f"all_pearson_mean"] = np.average(pearsons)
        results[f"all_spearman_mean"] = np.average(spearmans)
        return results


class InterpretableSemanticTextualSimilarityEvaluator:
    def __init__(self):
        pass

    def evaluate(self, model, dataset):
        pass
