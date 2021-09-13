import csv
import itertools
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class ContrastiveLearningDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        with open(filepath) as f:
            reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
            self.data = list(reader)

    def __getitem__(self, index):
        def f(x):
            return self.tokenizer(
                x,
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
