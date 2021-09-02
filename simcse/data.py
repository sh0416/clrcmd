from typing import List, Tuple, Dict, Any, Union, Optional
import itertools
import torch
from dataclasses import dataclass
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

Pair = Tuple[int, int]


def create_intervals(tokens: List[str]) -> List[Pair]:
    start_pos = itertools.accumulate(map(len, tokens), initial=0)
    length = map(len, tokens)
    return list(map(lambda x: (x[0], x[0] + x[1]), zip(start_pos, length)))


def is_overlap(interval1: Pair, interval2: Pair) -> bool:
    assert interval1[0] < interval1[1]
    assert interval2[0] < interval2[1]
    l = sorted(
        [
            (interval1[0], 0),
            (interval1[1], 1),
            (interval2[0], 0),
            (interval2[1], 1),
        ],
        key=lambda x: (x[0], 1 - x[1]),
    )
    l = list(map(lambda x: x[1], l))
    # In this case, only two possible cases are yield in this logic,
    # (0, 0, 1, 1), which is overlapped, or (0, 1, 0, 1), which is exclusive
    return l == [0, 0, 1, 1]


def create_overlap_pairs_from_intervals(
    intervals1: List[Pair], intervals2: List[Pair]
) -> List[Tuple[Pair, Pair]]:
    pipeline = itertools.product(intervals1, intervals2)
    pipeline = filter(lambda x: is_overlap(x[0], x[1]), pipeline)
    return list(pipeline)


def create_perfect_overlap_pairs_from_intervals(
    intervals1: List[Pair], intervals2: List[Pair]
) -> List[Tuple[Pair, Pair]]:
    pipeline = itertools.product(intervals1, intervals2)
    pipeline = filter(lambda x: x[0] == x[1], pipeline)
    return list(pipeline)


def create_perfect_overlap_pairs_from_tokens(
    tokens1: List[str], tokens2: List[str]
) -> List[Tuple[int, int]]:
    intervals1 = create_intervals(tokens1)
    intervals2 = create_intervals(tokens2)
    # NOTE: Due to the special token, the index starts with 1
    interval2idx1 = {x: i for i, x in enumerate(intervals1, start=1)}
    interval2idx2 = {x: i for i, x in enumerate(intervals2, start=1)}
    pairs = create_perfect_overlap_pairs_from_intervals(intervals1, intervals2)
    # Index pair
    pairs = [(interval2idx1[x], interval2idx2[y]) for x, y in pairs]
    return pairs


@dataclass
class PairDataCollator(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pairs = []
        for x in features:
            _pairs = [
                y
                for y in x["pairs"]
                if y[0] < self.max_length and y[1] < self.max_length
            ]
            pairs.append(torch.tensor(_pairs, dtype=torch.long))
            del x["pairs"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["pairs"] = pairs
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
