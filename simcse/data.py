import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
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
    mlm: bool = False
    mlm_prob: float = 0.15

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pairs_list = []
        for x in features:
            pairs_list.append(x["pairs"])
            del x["pairs"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        for idx in range(len(pairs_list)):
            pairs = pairs_list[idx]
            pairs = filter(lambda x: x[0] < self.max_length, pairs)
            pairs = filter(lambda x: x[1] < self.max_length, pairs)
            pairs = [(idx, x, y) for x, y in pairs]
            pairs = [(idx, 0, 0)] + pairs
            pairs = torch.tensor(pairs, dtype=torch.long)
            pairs_list[idx] = pairs
        batch["pairs"] = torch.cat(pairs_list)
        if self.mlm:
            seq_len = batch["input_ids"].shape[2]
            input_ids_mlm, labels_mlm = self.mask_tokens(
                batch["input_ids"].view(-1, seq_len).clone()
            )
            batch["input_ids_mlm"] = input_ids_mlm.view(-1, 2, seq_len)
            batch["labels_mlm"] = labels_mlm.view(-1, 2, seq_len)

        return batch

    def mask_tokens(
        self,
        inputs: Tensor,
        special_tokens_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with
        # probability `self.mlm_prob`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
