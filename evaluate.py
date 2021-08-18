import csv
import argparse
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils import batch


class Input(NamedTuple):
    text1: List[str]
    text2: List[str]


class Example(NamedTuple):
    input: Input
    score: float


def tokenize(row: Dict[str, str]) -> Dict[str, List[str]]:
    return {"text1": row["text1"].split(), "text2": row["text2"].split()}


def load_data_sts(filepaths: Tuple[str, str]) -> List[Example]:
    filepath_input, filepath_label = filepaths
    with open(filepath_input) as f_input, open(filepath_label) as f_label:
        reader_input = csv.DictReader(
            f_input,
            delimiter="\t",
            fieldnames=["text1", "text2"],
            quoting=csv.QUOTE_NONE,
        )
        reader_label = csv.DictReader(
            f_label,
            delimiter="\t",
            fieldnames=["score"],
            quoting=csv.QUOTE_NONE,
        )
        reader_input = map(tokenize, reader_input)
        reader_input = map(lambda x: Input(**x), reader_input)
        reader_label = map(lambda x: float(x["score"]), reader_label)
        dataset = map(
            lambda x: Example(input=x[0], score=x[1]), zip(reader_input, reader_label)
        )
        dataset = list(dataset)
        # Sort data by length to minimize padding in batcher
        dataset = sorted(
            dataset, key=lambda x: (len(x.input.text1), len(x.input.text2))
        )
        return dataset


def create_filepaths_sts(dirpath: str, datasets: List[str]) -> List[Tuple[str, str]]:
    filenames_input = map(lambda x: f"STS.input.{x}.txt", datasets)
    filenames_label = map(lambda x: f"STS.gs.{x}.txt", datasets)
    filepaths_input = map(lambda x: f"{dirpath}/{x}", filenames_input)
    filepaths_label = map(lambda x: f"{dirpath}/{x}", filenames_label)
    return list(zip(filepaths_input, filepaths_label))


def load_sts12(dirpath: str) -> Dict[str, List[Example]]:
    dataset_names = [
        "MSRpar",
        "MSRvid",
        "SMTeuroparl",
        "surprise.OnWN",
        "surprise.SMTnews",
    ]
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, dataset_names))
    return {k: v for k, v in zip(dataset_names, datasets)}


def load_sts13(dirpath: str) -> Dict[str, List[Example]]:
    dataset_names = ["FNWN", "headlines", "OnWN"]
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, dataset_names))
    return {k: v for k, v in zip(dataset_names, datasets)}


def load_sts14(dirpath: str) -> Dict[str, List[Example]]:
    dataset_names = [
        "deft-forum",
        "deft-news",
        "headlines",
        "images",
        "OnWN",
        "tweet-news",
    ]
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, dataset_names))
    return {k: v for k, v in zip(dataset_names, datasets)}


def load_sts15(dirpath: str) -> Dict[str, List[Example]]:
    dataset_names = [
        "answers-forums",
        "answers-students",
        "belief",
        "headlines",
        "images",
    ]
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, dataset_names))
    return {k: v for k, v in zip(dataset_names, datasets)}


def load_sts16(dirpath: str) -> Dict[str, List[Example]]:
    dataset_names = [
        "answer-answer",
        "headlines",
        "plagiarism",
        "postediting",
        "question-question",
    ]
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, dataset_names))
    return {k: v for k, v in zip(dataset_names, datasets)}


def evaluate_sts(
    dataset: Dict[str, List[Example]],
    param: Any,
    batcher: Callable[[List[Input], Any], np.ndarray],
) -> float:
    results = {}
    for name, _dataset in dataset.items():
        scores, labels = [], []
        for examples in batch(_dataset, 4):
            scores.append(batcher([x.input for x in examples], param))
            labels.append([x.score for x in examples])
        scores, labels = np.concatenate(scores), np.concatenate(labels)

        score_pearson = pearsonr(scores, labels)
        score_spearman = spearmanr(scores, labels)
        print(
            f"{name} : pearson = {score_pearson[0]:.4f}, spearman = {score_spearman[0]:.4f}"
        )
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

    print(
        f"ALL (weighted average) : Pearson = {wavg_pearson:.4f}, Spearman = {wavg_spearman:.4f}"
    )
    print(
        f"ALL (average) : Pearson = {avg_pearson:.4f}, Spearman = {avg_spearman:.4f}\n"
    )
    results["all"] = {
        "pearson": {"mean": avg_pearson, "wmean": wavg_pearson},
        "spearman": {"mean": avg_spearman, "wmean": wavg_spearman},
    }

    return results


parser = argparse.ArgumentParser()

if __name__ == "__main__":
    args = parser.parse_args()

    def batcher(inputs: List[Input], param: Any) -> np.ndarray:
        return np.random.rand(len(inputs))

    dataset = load_sts12("data/STS/STS12-en-test")
    """
    dataset = load_sts12("../SentEval/data/downstream/STS/STS12-en-test")
    dataset = {
        "noise": load_data_sts(
            (
                "../SimCSE/SentEval/data/downstream/STS/STS12-en-test/STS.input-retyped.MSRpar.txt",
                "../SimCSE/SentEval/data/downstream/STS/STS12-en-test/STS.gs.MSRpar.txt",
            )
        )
    }
    """
    print(dataset)
    print(evaluate_sts(dataset, {}, prepare, batcher))

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
