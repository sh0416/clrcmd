import argparse
import logging

from sentence_benchmark.data import (
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
from sentence_benchmark.evaluate import evaluate_sts

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--method",
    type=str,
    default="random",
    choices=["random", "bow", "sbert", "simcse"],
    help="method",
)
parser.add_argument("--pooler_type", type=str, choices=["avg", "rwmd"], default="avg")
parser.add_argument(
    "--checkpoint",
    type=str,
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
    "--data-dir", type=str, default="data/STS/STS12-en-test", help="data dir"
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
        from sentence_benchmark.models.random import batcher, prepare
    elif args.method == "bow":
        from sentence_benchmark.models.bow import batcher, prepare
    elif args.method == "sbert":
        from sentence_benchmark.models.sbert import batcher, prepare
    elif args.method == "simcse":
        from sentence_benchmark.models.simcse import batcher, prepare
    else:
        raise AttributeError()

    # Evaluate
    result = evaluate_sts(dataset, vars(args), prepare, batcher)
    logger.info("** Result **")
    for source, source_result in result.items():
        if source == "all":
            continue
        logger.info(f"  source: {source}")
        logger.info(f"    pearson: {source_result['pearson'][0]:.4f}")
        logger.info(f"    spearman: {source_result['spearman'][0]:.4f}")
    logger.info("  all")
    logger.info(f"    pearson  (all): {result['all']['pearson']['all']:.4f}")
    logger.info(f"    spearman (all): {result['all']['spearman']['all']:.4f}")
    logger.info(f"    pearson  (average): {result['all']['pearson']['mean']:.4f}")
    logger.info(f"    spearman (average): {result['all']['spearman']['mean']:.4f}")
    logger.info(f"    pearson  (waverage): {result['all']['pearson']['wmean']:.4f}")
    logger.info(f"    spearman (waverage): {result['all']['spearman']['wmean']:.4f}")
