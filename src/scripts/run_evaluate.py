import argparse
import logging

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
from sentsim.models.sbert import SentenceBert

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--method",
    type=str,
    default="random",
    choices=["random", "bow", "sbert", "simcse", "rwmdcse", "simcse-ours"],
    help="method",
)
parser.add_argument(
    "--word2vec-path",
    type=str,
    default="/nas/home/sh0416/data/fasttext/crawl-300d-2M.vec",
    help="Filepath for pre-trained word vector",
)
parser.add_argument(
    "--pooler_type", type=str, choices=["cls", "avg"], default="cls", help="pooler type"
)
parser.add_argument(
    "--model-name-or-path",
    type=str,
    default="sentence-transformers/nli-roberta-base-v2",
    help="checkpoint",
)
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
        model = SentenceBert(args.model_name_or_path, args.pooler_type)
    elif args.method == "simcse":
        from sentence_benchmark.models.simcse import batcher, prepare
    elif args.method == "rwmdcse" or args.method == "simcse-ours":
        from sentence_benchmark.models.rwmdcse import batcher, prepare
    else:
        raise AttributeError()

    # Evaluate
    evaluator = SemanticTextualSimilarityEvaluator(args.batch_size)
    result = evaluator.evaluate(model, dataset)
    logger.info("** Result **")
    for metric_name, metric_value in result.items():
        logger.info(f"{metric_name = }, {metric_value = }")
