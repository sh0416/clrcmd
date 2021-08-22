from typing import Any, Callable, Dict, List

from scipy.stats import pearsonr, spearmanr
from utils import batch

from sentence_benchmark.data import Example, Input


def evaluate_sts(
    dataset: Dict[str, List[Example]],
    param: Dict[str, Any],
    prepare: Callable[[List[Input]], Any],
    batcher: Callable[[List[Input], Any], np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for name, _dataset in dataset.items():
        scores, labels = [], []
        param = {**param, **prepare([x.input for x in _dataset])}
        for examples in batch(_dataset, 4):
            scores.append(batcher([x.input for x in examples], param))
            labels.append([x.score for x in examples])
        scores, labels = np.concatenate(scores), np.concatenate(labels)

        score_pearson = pearsonr(scores, labels)
        score_spearman = spearmanr(scores, labels)
        results[name] = {
            "pearson": score_pearson,
            "spearman": score_spearman,
            "nsamples": len(scores),
        }
    weights = [results[k]["nsamples"] for k in results.keys()]
    list_prs = np.array([results[k]["pearson"][0] for k in results.keys()])
    list_spr = np.array([results[k]["spearman"][0] for k in results.keys()])

    avg_pearson, avg_spearman = np.average(list_prs), np.average(list_spr)
    wavg_pearson = np.average(list_prs, weights=weights)
    wavg_spearman = np.average(list_spr, weights=weights)
    results["all"] = {
        "pearson": {"mean": avg_pearson, "wmean": wavg_pearson},
        "spearman": {"mean": avg_spearman, "wmean": wavg_spearman},
    }

    return results
