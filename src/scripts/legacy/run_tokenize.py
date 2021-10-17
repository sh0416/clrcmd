"""Tokenize given corpus"""
import argparse
import random

import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizerFast

from sentence_benchmark.tokenizer import RobertaTokenizerDropout

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--filepath",
    type=str,
    default=".data/wiki1m_for_simcse.txt",
    help="filepath",
)
parser.add_argument(
    "--bpe-dropout-prob",
    type=float,
    default=0.0,
    help="bpe dropout probability",
)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    args = parser.parse_args()
    # Load data
    data_files = {"train": args.filepath}
    datasets = load_dataset("text", data_files=data_files)

    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer2 = RobertaTokenizerDropout.from_pretrained("roberta-base")
    tokenizer2.bpe_dropout_prob = args.bpe_dropout_prob

    # Prepare features
    def tokenize(examples):
        text = examples["text"]

        input_strs = " ".join(tokenizer.tokenize(text))
        input_strs2 = " ".join(tokenizer2.tokenize(text))

        features = {"input_strs": input_strs, "input_strs2": input_strs2}
        return features

    # Tokenize data using BPE dropout
    tokenized_dataset = datasets["train"].map(tokenize, batched=False, num_proc=16)

    # Sanity check: The size of dataset must be 1,000,000
    print(len(tokenized_dataset))

    # Sanity check: Two tokenized sequences recover same text
    print(
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokenized_dataset[0]["input_strs"].split())
        )
    )
    print(
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(tokenized_dataset[0]["input_strs2"].split())
        )
    )

    # Save dataset as csv
    tokenized_dataset.to_csv(
        f"{args.filepath}_bpedropout_{args.bpe_dropout_prob}_roberta-base.csv",
        index=False,
    )
