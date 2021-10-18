import os
import re
from typing import List, TypedDict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor

from simcse.models import ModelInput, SentenceSimilarityModel
import xml.etree.ElementTree as ET


class AlignmentPair(TypedDict):
    sent1_word_ids: List[int]  # indexed by word tokenization
    sent2_word_ids: List[int]  # indexed by word tokenization
    type: str
    score: Optional[float]
    comment: str


class Alignment(TypedDict):
    id: int
    sent1: str  # word tokenized sentence with space
    sent2: str  # word tokenized sentence with space
    pairs: List[AlignmentPair]


def load_alignment(filepath: str) -> List[Alignment]:
    data = []
    with open(filepath) as f:
        s = "<data>" + f.read().replace("<==>", "==") + "</data>"
        tree = ET.fromstring(s)
        for sentence in tree:
            example_id = int(sentence.attrib["id"])
            sent1, sent2 = sentence.text.strip().splitlines()
            assert sent1.startswith("// ") and sent2.startswith("// ")
            sent1, sent2 = sent1[3:], sent2[3:]
            pairs = []
            for x in sentence.find("alignment").text.strip().splitlines():
                pair_ids, type, score, comment = x.split(" // ")
                sent1_word_ids, sent2_word_ids = pair_ids.split(" == ")
                sent1_word_ids = [int(x) for x in sent1_word_ids.split()]
                sent2_word_ids = [int(x) for x in sent2_word_ids.split()]
                score = None if score == "NIL" else float(score)
                pairs.append(
                    {
                        "sent1_word_ids": sent1_word_ids,
                        "sent2_word_ids": sent2_word_ids,
                        "type": type,
                        "score": score,
                        "comment": comment,
                    }
                )
            data.append(
                {"id": example_id, "sent1": sent1, "sent2": sent2, "pairs": pairs}
            )
    return data


def save_alignment(data: List[Alignment], filepath: str):
    """Save alignments with formatted text"""
    with open(filepath, "w") as f:
        for example in data:
            f.write(f"<sentence id=\"{example['id']}\" status=\"\">\n")
            f.write(f"// {example['sent1']}\n")
            f.write(f"// {example['sent2']}\n")
            f.write("<source>\n")
            for id, word in enumerate(example["sent1"].split(), start=1):
                f.write(f"{id} {word} :\n")
            f.write("</source>\n")
            f.write("<translation>\n")
            for id, word in enumerate(example["sent2"].split(), start=1):
                f.write(f"{id} {word} :\n")
            f.write("</translation>\n")
            f.write("<alignment>\n")
            for a in example["pairs"]:
                str_sent1_word_ids = " ".join(map(str, a["sent1_word_ids"]))
                str_sent2_word_ids = " ".join(map(str, a["sent2_word_ids"]))
                str_score = "NIL" if a["score"] is None else str(a["score"])
                f.write(f"{str_sent1_word_ids} <==> {str_sent2_word_ids} // ")
                f.write(f"{a['type']} // {str_score} // {a['comment']}\n")
            f.write("</alignment>\n")
            f.write("</sentence>\n\n\n")


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
