import argparse
import json
import logging
from typing import List

import numpy as np
import torch
from tokenizations import get_alignments
from transformers import AutoTokenizer

from simcse.config import ModelArguments
from simcse.eval.ists import (
    extract_alignment_from_heatmap,
    load_chunked_sentence_pairs,
    load_sentence_pairs,
    pool_heatmap,
)
from simcse.models import create_contrastive_learning

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", type=str, required=True, help="data directory for ists"
)
parser.add_argument("--source", type=str, required=True, help="source for ists data")
parser.add_argument(
    "--model-args-path", type=str, required=True, help="path for model_args"
)
parser.add_argument(
    "--model-path", type=str, required=True, help="path for checkpointcheckpoint path"
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    with open(args.model_args_path) as f:
        model_args = ModelArguments(**json.load(f))

    sentences = load_sentence_pairs(args.data_dir, args.source)[:4]
    logger.info(f"Load {sentences[0] = }")

    sentences_chunk = load_chunked_sentence_pairs(args.data_dir, args.source)[:4]
    logger.info(f"Load {sentences_chunk[0] = }")

    sentences_word = list(map(lambda x: (x[0].split(), x[1].split()), sentences))
    logger.info(f"Compute {sentences_word[0] = }")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info(f"Load {tokenizer = }")

    def preprocess(x):
        def tokenize(s: str) -> List[str]:
            return tokenizer.convert_ids_to_tokens(
                tokenizer(s, add_special_tokens=False)["input_ids"]
            )

        return tokenize(x[0]), tokenize(x[1])

    sentences_token = list(map(preprocess, sentences))
    logger.info(f"Compute {sentences_token[0] = }")

    def preprocess(x_src, x_tgt):
        src2tgt1, _ = get_alignments(x_src[0], x_tgt[0])
        src2tgt2, _ = get_alignments(x_src[1], x_tgt[1])
        return src2tgt1, src2tgt2

    align_token2chunk = list(map(preprocess, sentences_token, sentences_chunk))
    logger.info(f"Compute {align_token2chunk[0] = }")

    align_chunk2word = list(map(preprocess, sentences_chunk, sentences_word))
    logger.info(f"Compute {align_chunk2word[0] = }")

    def preprocess(x):
        inputs1 = tokenizer(x[0], return_tensors="pt")
        inputs2 = tokenizer(x[1], return_tensors="pt")
        return inputs1, inputs2

    dataset = list(map(preprocess, sentences))
    logger.info(f"Compute {dataset[0] = }")

    module = create_contrastive_learning(model_args)
    module.load_state_dict(torch.load(args.model_path))
    with torch.no_grad():
        heatmaps = [
            module.model.compute_heatmap(x[0], x[1])[0, 1:-1, 1:-1].numpy()
            for x in dataset
        ]
    logger.info(f"Compute {heatmaps[0] = }")

    pooled_heatmaps = list(map(pool_heatmap, heatmaps, align_token2chunk))
    logger.info(f"Compute {pooled_heatmaps[0] = }")

    alignment = [[i] for i in np.argmax(pooled_heatmaps[0], axis=1)]
    logger.info(f"{alignment = }")

    for sent1_chunk, sent2_chunks in enumerate(alignment):
        sent1_words = [word_id + 1 for word_id in align_chunk2word[0][0][sent1_chunk]]
        sent2_words = [
            word_id + 1
            for sent2_chunk in sent2_chunks
            for word_id in align_chunk2word[0][1][sent2_chunk]
        ]
        print(sent1_words, sent2_words)


if __name__ == "__main__":
    main()
