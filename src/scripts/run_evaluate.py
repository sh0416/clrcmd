import argparse
import logging
import json
from transformers.models.auto.tokenization_auto import AutoTokenizer

import torch
from transformers.utils.dummy_pt_objects import AutoModel

from sentsim.data.sts import (
    load_sickr_dev,
    load_sickr_test,
    load_sickr_train,
    load_sources_sts,
    load_sts12,
    load_sts13,
    load_sts14,
    load_sts15,
    load_sts16,
    load_stsb_dev,
    load_stsb_test,
    load_stsb_train,
)
from sentsim.evaluator import SemanticTextualSimilarityEvaluator
from sentsim.models.bow import BagOfWord
from sentsim.models.random import RandomSimilarityModel
from sentsim.models.sbert import PytorchSemanticTextualSimilarityModel
from sentsim.models.models import create_contrastive_learning
from sentsim.config import ModelArguments

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--method",
    type=str,
    default="random",
    choices=["random", "bow", "sbert"],
    help="method",
)
parser.add_argument(
    "--word2vec-path",
    type=str,
    default="/nas/home/sh0416/data/fasttext/crawl-300d-2M.vec",
    help="Filepath for pre-trained word vector",
)
parser.add_argument("--model-args-path", type=str)
parser.add_argument(
    "--model-name-or-path",
    type=str,
    default="sentence-transformers/nli-roberta-base-v2",
    help="checkpoint",
)
parser.add_argument("--checkpoint", type=str)
parser.add_argument(
    "--dataset",
    type=str,
    default="STS12",
    choices=[
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSB-train",
        "STSB-dev",
        "STSB-test",
        "SICKR-train",
        "SICKR-dev",
        "SICKR-test",
        "custom",
    ],
    help="dataset",
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="/nas/home/sh0416/data/STS/STS12-en-test",
    help="data dir",
)
parser.add_argument("--sources", type=str, nargs="*", help="sources")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("** Command Line Arguments **")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Load dataset
    if args.dataset == "STS12":
        load_fn = load_sts12
    elif args.dataset == "STS13":
        load_fn = load_sts13
    elif args.dataset == "STS14":
        load_fn = load_sts14
    elif args.dataset == "STS15":
        load_fn = load_sts15
    elif args.dataset == "STS16":
        load_fn = load_sts16
    elif args.dataset == "STSB-train":
        load_fn = load_stsb_train
    elif args.dataset == "STSB-dev":
        load_fn = load_stsb_dev
    elif args.dataset == "STSB-test":
        load_fn = load_stsb_test
    elif args.dataset == "SICKR-train":
        load_fn = load_sickr_train
    elif args.dataset == "SICKR-dev":
        load_fn = load_sickr_dev
    elif args.dataset == "SICKR-test":
        load_fn = load_sickr_test
    elif args.dataset == "custom":
        load_fn = load_sources_sts
    else:
        raise argparse.ArgumentError("Invalid --dataset")
    if args.dataset == "custom":
        dataset = load_fn(args.data_dir, args.sources)
    else:
        dataset = load_fn(args.data_dir)

    # Load method
    if args.method == "random":
        model = RandomSimilarityModel()
    elif args.method == "bow":
        corpus = [
            s for examples in dataset.values() for s_pair, _ in examples for s in s_pair
        ]
        model = BagOfWord(args.word2vec_path, corpus)
    elif args.method == "sbert":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        with open(args.model_args_path) as f:
            model_args = ModelArguments(**json.load(f))
        model = create_contrastive_learning(model_args)
        model.load_state_dict(torch.load(args.checkpoint))
        model = PytorchSemanticTextualSimilarityModel(model.model, tokenizer)
    else:
        raise ValueError

    # Evaluate
    evaluator = SemanticTextualSimilarityEvaluator(args.batch_size)
    result = evaluator.evaluate(model, dataset)
    logger.info("** Result **")
    for metric_name, metric_value in result.items():
        logger.info(f"{metric_name = }, {metric_value = }")
