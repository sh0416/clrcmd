import argparse
import logging
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from clrcmd.data.dataset import STSBenchmarkDataset
from clrcmd.data.sts import load_sts_benchmark
from clrcmd.models import create_similarity_model, create_tokenizer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--model", type=str, default="bert-cls", choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"], help="Model")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
parser.add_argument("--data-dir", type=str, default="data", help="data dir")
parser.add_argument("--dataset", type=str, default="sts12", choices=["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"], help="dataset")
# fmt: on


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs("log", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filename=f"log/evaluate-{args.dataset}-{args.model}-{args.checkpoint}.log",
    )
    logger.info("** Command Line Arguments **")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Create tokenizer and model
    tokenizer = create_tokenizer(args.model)
    model = create_similarity_model(args.model).to(device)

    # Load dataset
    source = load_sts_benchmark(args.data_dir, args.dataset)
    datasets = {k: STSBenchmarkDataset(v, tokenizer) for k, v in source.items()}
    loaders = {k: DataLoader(v, batch_size=32) for k, v in datasets.items()}

    # Load method

    # Evaluate
    model.eval()
    with torch.no_grad():
        result, scores_all, labels_all = {}, [], []
        for source, loader in loaders.items():
            logger.info(f"Evaluate {source}")
            scores, labels = [], []
            for examples in tqdm(loader, desc=f"Evaluate {source}"):
                text1, text2, label = examples
                text1 = {k: v.to(device) for k, v in text1.items()}
                text2 = {k: v.to(device) for k, v in text2.items()}
                scores.append(model(text1, text2).cpu().numpy())
                labels.append(label.numpy())
            scores, labels = np.concatenate(scores), np.concatenate(labels)
            result[f"{source}_pearson"] = pearsonr(scores, labels)[0]
            result[f"{source}_spearman"] = spearmanr(scores, labels)[0]
            scores_all.append(scores)
            labels_all.append(labels)
        scores_all, labels_all = np.concatenate(scores_all), np.concatenate(labels_all)
        result[f"all_pearson_all"] = pearsonr(scores_all, labels_all)[0]
        result[f"all_spearman_all"] = spearmanr(scores_all, labels_all)[0]

    pearsons = [result[f"{source}_pearson"] for source in loaders]
    spearmans = [result[f"{source}_spearman"] for source in loaders]
    result[f"all_pearson_mean"] = np.average(pearsons)
    result[f"all_spearman_mean"] = np.average(spearmans)

    logger.info("** Result **")
    for metric_name, metric_value in result.items():
        logger.info(f"{metric_name = }, {metric_value = :.4f}")


if __name__ == "__main__":
    main()
