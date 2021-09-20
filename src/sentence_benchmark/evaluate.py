from typing import Any, Callable, Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

from sentence_benchmark.data import Example, Input
from sentence_benchmark.utils import batch


def evaluate_sts(
    dataset: Dict[str, List[Example]],
    param: Dict[str, Any],
    prepare: Callable[[List[Input], Dict], Dict],
    batcher: Callable[[List[Input], Dict], np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    results = {}
    scores_all = []
    labels_all = []
    for name, _dataset in dataset.items():
        scores, labels = [], []
        param = prepare([x.input for x in _dataset], param)
        for examples in batch(_dataset, param["batch_size"]):
            scores.append(batcher([x.input for x in examples], param))
            labels.append([x.score for x in examples])
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        scores_all.append(scores)
        labels_all.append(labels)

        score_pearson = pearsonr(scores, labels)
        score_spearman = spearmanr(scores, labels)
        results[name] = {
            "pearson": score_pearson,
            "spearman": score_spearman,
            "nsamples": len(scores),
        }
    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    pearson_all = pearsonr(scores_all, labels_all)[0]
    spearman_all = spearmanr(scores_all, labels_all)[0]

    weights = [results[k]["nsamples"] for k in results.keys()]
    list_prs = np.array([results[k]["pearson"][0] for k in results.keys()])
    list_spr = np.array([results[k]["spearman"][0] for k in results.keys()])
    pearson_avg = np.average(list_prs)
    spearman_avg = np.average(list_spr)
    pearson_wavg = np.average(list_prs, weights=weights)
    spearman_wavg = np.average(list_spr, weights=weights)

    results["all"] = {
        "pearson": {
            "all": pearson_all,
            "mean": pearson_avg,
            "wmean": pearson_wavg,
        },
        "spearman": {
            "all": spearman_all,
            "mean": spearman_avg,
            "wmean": spearman_wavg,
        },
    }

    return results
