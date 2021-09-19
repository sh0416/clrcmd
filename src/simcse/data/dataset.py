import abc
import csv
from functools import partial
import logging
import random
from typing import Dict, Tuple, List

from torch import Tensor
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)


class ContrastiveLearningDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        with open(filepath) as f:
            self.data = [x.strip() for x in f]

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
    def _create_tokenized_pair(self, s: str) -> Tuple[List[str], List[str]]:
        pass


class SimCSEDataset(ContrastiveLearningDataset):
    def _create_tokenized_pair(self, sentence: str) -> Tuple[List[str], List[str]]:
        tokens = self.tokenizer.tokenize(sentence)
        return tokens, tokens


class ESimCSEDataset(ContrastiveLearningDataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer, dup_rate: float):
        super().__init__(filepath=filepath, tokenizer=tokenizer)
        self.dup_rate = dup_rate

    def _create_tokenized_pair(self, sentence: str) -> Tuple[List[str], List[str]]:
        tokens = self.tokenizer.tokenize(sentence)
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


class TokenizedContrastiveLearningDataset(Dataset):
    """Deprecated"""

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
