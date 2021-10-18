import os
import re
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from simcse.models import ModelInput, SentenceSimilarityModel


def load_sentence_pairs(dirpath: str, source: str) -> List[Tuple[str, str]]:
    with open(os.path.join(dirpath, f"STSint.testinput.{source}.sent1.txt")) as f:
        sent1 = [x.strip() for x in f]
    with open(os.path.join(dirpath, f"STSint.testinput.{source}.sent2.txt")) as f:
        sent2 = [x.strip() for x in f]
    assert len(sent1) == len(sent2), "Two file has different length"
    assert all(map(lambda x: len(x) > 0, sent1)), "Some sentence are empty"
    assert all(map(lambda x: len(x) > 0, sent2)), "Some sentence are empty"
    return list(zip(sent1, sent2))


def load_chunked_sentence_pairs(
    dirpath: str, source: str
) -> List[Tuple[List[str], List[str]]]:
    """Load chunked sentence (splitted by gold segmenter)"""
    with open(os.path.join(dirpath, f"STSint.testinput.{source}.sent1.chunk.txt")) as f:
        sent1 = [x.strip() for x in f]
    with open(os.path.join(dirpath, f"STSint.testinput.{source}.sent2.chunk.txt")) as f:
        sent2 = [x.strip() for x in f]
    assert len(sent1) == len(sent2), "Two file has different length"
    assert all(map(lambda x: len(x) > 0, sent1)), "Some sentence are empty"
    assert all(map(lambda x: len(x) > 0, sent2)), "Some sentence are empty"
    pattern = re.compile(r"\[\s(.*?)\s\]")
    sent1 = [pattern.findall(x) for x in sent1]
    sent2 = [pattern.findall(x) for x in sent2]
    return list(zip(sent1, sent2))


TokensPair = Tuple[List[str], List[str]]
Alignment = List[TokensPair]


def pool_heatmap(
    heatmap: np.ndarray, align_pair: Tuple[List[List[int]], List[List[int]]]
) -> np.ndarray:
    sent1_max = max(max(x) for x in align_pair[0])
    sent2_max = max(max(x) for x in align_pair[1])
    heatmap_new = np.zeros((sent1_max + 1, sent2_max + 1))
    count = np.zeros((sent1_max + 1, sent2_max + 1))
    print(heatmap_new.shape, heatmap.shape)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            for k in align_pair[0][i]:
                for l in align_pair[1][j]:
                    heatmap_new[k, l] = max(heatmap_new[k, l], heatmap[i, j])
                    count[k, l] += 1
    # heatmap_new /= count
    return heatmap_new


def extract_alignment_from_heatmap(heatmap: np.ndarray) -> Alignment:
    argmax = np.argmax(heatmap, axis=1).cpu().detach().numpy()
    alignment = [([i], [x]) for i, x in enumerate(argmax)]
    return alignment


def convert_alignment_granularity(
    alignment: Alignment, granularity: List[List[str]]
) -> Alignment:
    alignment_new = []
    for tokens1, tokens2 in alignment:
        tokens1 = list(set(granularity[t] for t in tokens1))
        tokens2 = list(set(granularity[t] for t in tokens2))
        alignment_new.append((tokens1, tokens2))
    return alignment_new
