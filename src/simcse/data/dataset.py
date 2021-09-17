import abc
import csv
import logging
import random
from typing import Any, Dict, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)


class ContrastiveLearningDataset(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        pass


class SimCSEDataset(ContrastiveLearningDataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        with open(filepath) as f:
            self.data = [x.strip() for x in f]

    def __getitem__(self, index: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        sentence = self.data[index]
        x = self.tokenizer.encode_plus(
            sentence,
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        return x, x

    def __len__(self) -> int:
        return len(self.data)


class ESimCSEDataset(ContrastiveLearningDataset):
    def __init__(
        self, filepath: str, tokenizer: RobertaTokenizer, dup_rate: float = 0.5
    ):
        self.tokenizer = tokenizer
        with open(filepath) as f:
            self.data = [x.strip() for x in f]
        self.dup_rate = dup_rate

    def __getitem__(self, index: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        sentence = self.data[index]
        tokens = self.tokenizer.tokenize(sentence)
        # Compute dup_len
        dup_len = random.randint(0, max(1, int(self.dup_rate * len(tokens))))
        # Compute positions
        dup_set = random.sample(range(len(tokens)), k=dup_len)
        # Compute repetition tokens
        tokens_new = []
        for pos, token in enumerate(tokens):
            tokens_new.append(token)
            if pos in dup_set:
                tokens_new.append(token)
        x = self.tokenizer.encode_plus(
            tokens,
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        x_pos = self.tokenizer.encode_plus(
            tokens_new,
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        return x, x_pos

    def __len__(self) -> int:
        return len(self.data)


class TokenizedContrastiveLearningDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        with open(filepath) as f:
            reader = csv.DictReader(f)
            self.data = list(reader)

    def __getitem__(self, index):
        def f(x):
            return self.tokenizer.encode_plus(
                self.tokenizer.convert_tokens_to_ids(x.split()),
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=32,
            )

        return f(self.data[index]["input_strs"]), f(self.data[index]["input_strs2"])

    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer: RobertaTokenizer):
    batch1 = tokenizer.pad([x for x, _ in batch], return_tensors="pt")
    batch2 = tokenizer.pad([x for _, x in batch], return_tensors="pt")
    return {
        "input_ids1": batch1["input_ids"],
        "attention_mask1": batch1["attention_mask"],
        "input_ids2": batch2["input_ids"],
        "attention_mask2": batch2["attention_mask"],
    }
