import abc
import csv
import logging
import random
from functools import partial
from typing import Dict, List, Tuple
import typing

from torch import Tensor
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from simcse.data.eda import eda

logger = logging.getLogger(__name__)


def _load_txt(filepath: str) -> List[str]:
    with open(filepath) as f:
        return [x.strip() for x in f]


def _load_csv(filepath: str) -> List[Tuple[str, ...]]:
    with open(filepath) as f:
        reader = csv.reader(f)
        return list(reader)


Row = typing.TypeVar("Row", str, Tuple[str, str])


class ContrastiveLearningDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        self.data = self._load_data(filepath)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        x, x_pos = self._create_tokenized_pair(self.data[index])
        f = partial(
            self.tokenizer.encode_plus,
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        return f(x), f(x_pos)

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def _load_data(self, filepath: str) -> List[Row]:
        pass

    @abc.abstractmethod
    def _create_tokenized_pair(self, row: Row) -> Tuple[List[str], List[str]]:
        pass


class SimCSEDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        return _load_txt(filepath)

    def _create_tokenized_pair(self, row: Row) -> Tuple[List[str], List[str]]:
        tokens = self.tokenizer.tokenize(row)
        return tokens, tokens


class ESimCSEDataset(ContrastiveLearningDataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer, dup_rate: float):
        super().__init__(filepath=filepath, tokenizer=tokenizer)
        self.dup_rate = dup_rate

    def _load_data(self, filepath: str) -> List[Row]:
        return _load_txt(filepath)

    def _create_tokenized_pair(self, row: Row) -> Tuple[List[str], List[str]]:
        tokens = self.tokenizer.tokenize(row)
        # Compute dup_len
        dup_len = random.randint(0, max(1, int(self.dup_rate * len(tokens))))
        # Compute positions
        dup_set = random.sample(range(len(tokens)), k=dup_len)
        # Compute repetition tokens
        tokens_pos = []
        for pos, token in enumerate(tokens):
            tokens_pos.append(token)
            if pos in dup_set:
                tokens_pos.append(token)
        return tokens, tokens_pos


class EDASimCSEDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        return _load_txt(filepath)

    def _create_tokenized_pair(self, row: Row) -> Tuple[List[str], List[str]]:
        sentence_pos = eda(row, num_aug=1)[0]
        assert len(sentence_pos) > 0, row
        tokens = self.tokenizer.tokenize(row)
        tokens_pos = self.tokenizer.tokenize(sentence_pos)
        return tokens, tokens_pos


class PairedContrastiveLearningDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        return _load_csv(filepath)

    def _create_tokenized_pair(self, row: Row) -> Tuple[List[str], List[str]]:
        tokens = self.tokenizer.tokenize(row[0])
        tokens_pos = self.tokenizer.tokenize(row[1])
        assert len(tokens) > 0, f"{row = }"
        assert len(tokens_pos) > 0, f"{row = }"
        return tokens, tokens_pos


def collate_fn(batch, tokenizer: RobertaTokenizer):
    batch1 = tokenizer.pad([x for x, _ in batch], return_tensors="pt")
    batch2 = tokenizer.pad([x for _, x in batch], return_tensors="pt")
    return {
        "input_ids1": batch1["input_ids"],
        "attention_mask1": batch1["attention_mask"],
        "input_ids2": batch2["input_ids"],
        "attention_mask2": batch2["attention_mask"],
    }
