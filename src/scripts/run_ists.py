import argparse
import json
import os
from typing import Tuple

import torch
from transformers import AutoTokenizer

from sentsim.config import ModelArguments
from sentsim.eval.ists import load_instances, preprocess_instances, inference
from sentsim.eval.ists import save_infered_instances

from sentsim.models.models import create_contrastive_learning

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data-dir",
    type=str,
    default="/nas/home/sh0416/data/semeval16/task2/test_goldStandard",
    help="data dir",
)
parser.add_argument(
    "--source",
    type=str,
    default="images",
    choices=["images", "headlines"],
    help="data source",
)
parser.add_argument(
    "--ckpt-dir",
    type=str,
    required=True,
    help="checkpoint directory",
)
parser.add_argument(
    "--ckpt-path",
    type=str,
    required=True,
    help="checkpoint path",
)


def create_filepaths(data_dir: str, source: str) -> Tuple[str, ...]:
    return (
        os.path.join(data_dir, f"STSint.testinput.{source}.sent1.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent2.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent1.chunk.txt"),
        os.path.join(data_dir, f"STSint.testinput.{source}.sent2.chunk.txt"),
    )


def main():
    args = parser.parse_args()

    instances = load_instances(*create_filepaths(args.data_dir, args.source))

    with open(os.path.join(args.ckpt_dir, "model_args.json")) as f:
        model_args = ModelArguments(**json.load(f))
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    prep_instances = preprocess_instances(tokenizer, instances)

    module = create_contrastive_learning(model_args)
    module.load_state_dict(torch.load(args.ckpt_path))
    infered_instances = inference(module.model, prep_instances)
    save_infered_instances(
        infered_instances, os.path.join(args.ckpt_dir, f"{args.source}.wa")
    )


if __name__ == "__main__":
    main()
