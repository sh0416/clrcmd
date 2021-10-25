import abc
import csv
import logging
import random
import typing
from functools import partial
from typing import Dict, List, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from simcse.data.eda import eda

logger = logging.getLogger(__name__)


def _load_txt(filepath: str) -> List[str]:
    with open(filepath) as f:
        return [x.strip() for x in f]


def _load_csv(filepath: str) -> List[Tuple[str, ...]]:
    with open(filepath) as f:
        reader = csv.reader(f)
        return list(reader)


Row = typing.TypeVar("Row", str, Tuple[str, str], Tuple[str, str, str])
TokenizedTriplet = Tuple[List[str], List[str], Optional[List[str]]]
Example = Tuple[Dict[str, Tensor], Dict[str, Tensor], Optional[Dict[str, Tensor]]]


class ContrastiveLearningDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.data = self._load_data(filepath)

    def __getitem__(self, index: int) -> Example:
        x, x_pos, x_neg = self._create_tokenized_pair(self.data[index])
        f = partial(
            self.tokenizer.encode_plus,
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        if x_neg is not None:
            return f(x), f(x_pos), f(x_neg)
        else:
            return f(x), f(x_pos), None

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def _load_data(self, filepath: str) -> List[Row]:
        pass

    @abc.abstractmethod
    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
        pass


class WikiDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        return _load_txt(filepath)


class WikiIdentityDataset(WikiDataset):
    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
        tokens = self.tokenizer.tokenize(row)
        return tokens, tokens, None


class WikiRepetitionDataset(WikiDataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer, dup_rate: float):
        super().__init__(filepath=filepath, tokenizer=tokenizer)
        self.dup_rate = dup_rate

    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
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
        return tokens, tokens_pos, None


class WikiEDADataset(WikiDataset):
    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
        sentence_pos = eda(row, num_aug=1)[0]
        assert len(sentence_pos) > 0, row
        tokens = self.tokenizer.tokenize(row)
        tokens_pos = self.tokenizer.tokenize(sentence_pos)
        return tokens, tokens_pos, None


class NLIDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        with open(filepath) as f:
            reader = csv.DictReader(f)
            return [(row["sent0"], row["sent1"], row["hard_neg"]) for row in reader]

    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
        tokens = self.tokenizer.tokenize(row[0])
        tokens_pos = self.tokenizer.tokenize(row[1])
        tokens_neg = self.tokenizer.tokenize(row[2])
        return tokens, tokens_pos, tokens_neg


class KorNLIDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        with open(filepath) as f:
            reader = csv.DictReader(f)
            return [
                (row["sentence1"], row["sentence2"])
                for row in reader
                if row["gold_label"] == "entailment"
            ]

    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
        tokens = self.tokenizer.tokenize(row[0])
        tokens_pos = self.tokenizer.tokenize(row[1])
        return tokens, tokens_pos, None


class PairedContrastiveLearningDataset(ContrastiveLearningDataset):
    def _load_data(self, filepath: str) -> List[Row]:
        return _load_csv(filepath)

    def _create_tokenized_pair(self, row: Row) -> TokenizedTriplet:
        tokens = self.tokenizer.tokenize(row[0])
        tokens_pos = self.tokenizer.tokenize(row[1])
        assert len(tokens) > 0, f"{row = }"
        assert len(tokens_pos) > 0, f"{row = }"
        return tokens, tokens_pos, None


def collate_fn(
    batch: List[Example],
    tokenizer: RobertaTokenizer,
) -> Dict[str, Tensor]:
    batch_x, batch_pos, batch_neg = [], [], []
    for x, x_pos, x_neg in batch:
        batch_x.append(x)
        batch_pos.append(x_pos)
        if x_neg is not None:
            batch_neg.append(x_neg)
    assert len(batch_neg) == 0 or len(batch_x) == len(batch_neg)
    batch_x = tokenizer.pad(batch_x, return_tensors="pt")
    batch_pos = tokenizer.pad(batch_pos, return_tensors="pt")
    if len(batch_neg) > 0:
        batch_neg = tokenizer.pad(batch_neg, return_tensors="pt")
    else:
        batch_neg = None
    return {"inputs1": batch_x, "inputs2": batch_pos, "inputs_neg": batch_neg}
