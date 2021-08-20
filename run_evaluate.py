import argparse
import logging
from typing import Any, Callable, Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

from dataset import (
    Example,
    Input,
    load_sources,
    load_sts12,
    load_sts13,
    load_sts14,
    load_sts15,
    load_sts16,
)
from utils import batch

logger = logging.getLogger(__name__)


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


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--method",
    type=str,
    default="random",
    choices=["random", "bow", "sbert"],
    help="method",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="STS12",
    choices=["STS12", "STS13", "STS14", "STS15", "STS16", "custom"],
    help="dataset",
)
parser.add_argument(
    "--data-dir", type=str, default="data/STS/STS12-en-test", help="data dir"
)
parser.add_argument("--sources", type=str, nargs="*", help="sources")


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("** Command Line Arguments **")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Load dataset
    if args.dataset == "STS12":
        dataset = load_sts12(args.data_dir)
    elif args.dataset == "STS13":
        dataset = load_sts13(args.data_dir)
    elif args.dataset == "STS14":
        dataset = load_sts14(args.data_dir)
    elif args.dataset == "STS15":
        dataset = load_sts15(args.data_dir)
    elif args.dataset == "STS16":
        dataset = load_sts16(args.data_dir)
    elif args.dataset == "custom":
        dataset = load_sources(args.data_dir, args.sources)
    else:
        raise argparse.ArgumentError("Invalid --dataset")

    # Load method
    if args.method == "random":

        def prepare(inputs: List[Input]) -> Dict:
            return {}

        def batcher(inputs: List[Input], param: Any) -> np.ndarray:
            return np.random.rand(len(inputs))

    elif args.method == "bow":
        from bow import batcher, prepare
    elif args.method == "sbert":
        from sbert import batcher, prepare
    else:
        raise AttributeError()

    # Evaluate
    result = evaluate_sts(dataset, {}, prepare, batcher)
    logger.info("** Result **")
    for source, source_result in result.items():
        if source == "all":
            continue
        logger.info(f"  source: {source}")
        logger.info(f"    pearson: {source_result['pearson'][0]:.4f}")
        logger.info(f"    spearman: {source_result['spearman'][0]:.4f}")
    logger.info("  all")
    logger.info(f"    pearson  (average): {result['all']['pearson']['mean']:.4f}")
    logger.info(f"    spearman (average): {result['all']['spearman']['mean']:.4f}")
    logger.info(f"    pearson  (waverage): {result['all']['pearson']['wmean']:.4f}")
    logger.info(f"    spearman (waverage): {result['all']['spearman']['wmean']:.4f}")

"""
class STSBenchmarkEval(SICKRelatednessEval):
    def __init__(self, task_path, seed=1111):
        logging.debug("\n\n***** Transfer task : STSBenchmark*****\n\n")
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, "sts-train.csv"))
        dev = self.loadFile(os.path.join(task_path, "sts-dev.csv"))
        test = self.loadFile(os.path.join(task_path, "sts-test.csv"))
        self.sick_data = {"train": train, "dev": dev, "test": test}

    def loadFile(self, fpath):
        sick_data = {"X_A": [], "X_B": [], "y": []}
        with io.open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip().split("\t")
                sick_data["X_A"].append(text[5].split())
                sick_data["X_B"].append(text[6].split())
                sick_data["y"].append(text[4])

        sick_data["y"] = [float(s) for s in sick_data["y"]]
        return sick_data
"""
