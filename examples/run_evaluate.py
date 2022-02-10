import argparse
import json
import logging

import torch
from transformers import AutoModel, AutoTokenizer

from clrcmd.config import ModelArguments
from clrcmd.data.sts import load_sts_benchmark
from clrcmd.evaluator import SemanticTextualSimilarityEvaluator
from clrcmd.models.models import create_contrastive_learning

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--model", type=str, default="bert-cls", choices=["bert-cls", "bert-avg", "roberta-cls", "roberta-avg"], help="Model")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
parser.add_argument("--data-dir", type=str, default="data", help="data dir")
parser.add_argument("--dataset", type=str, default="sts12", choices=["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"], help="dataset")
# fmt: on


def main():
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger.info("** Command Line Arguments **")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Load dataset
    source = load_sts_benchmark(args.data_dir, args.dataset)

    # Create model

    exit()
    # Load method

    # Evaluate
    evaluator = SemanticTextualSimilarityEvaluator(args.batch_size)
    result = evaluator.evaluate(model, dataset)
    logger.info("** Result **")
    for metric_name, metric_value in result.items():
        logger.info(f"{metric_name = }, {metric_value = }")


if __name__ == "__main__":
    main()
