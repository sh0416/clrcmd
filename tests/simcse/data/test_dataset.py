import random
from functools import partial

import pytest
from transformers import AutoTokenizer

from simcse.data.dataset import WikiIdentityDataset, WikiRepetitionDataset


@pytest.fixture
def corpus_wiki():
    return ["my name is seonghyeon", "how are you?", "i'm fine thank you. and you?"]


@pytest.fixture
def filepath_wiki(tmpdir, corpus_wiki):
    # Write corpus to temporary directory and return their filepath
    tmpfile = tmpdir.join("wiki.txt")
    tmpfile.write("\n".join(corpus_wiki))
    return tmpfile.strpath


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base", use_fast=False)


def test_wiki_identity_dataset(filepath_wiki, tokenizer, corpus_wiki):
    # Create dataset
    dataset = WikiIdentityDataset(filepath_wiki, tokenizer)

    # Test dataset
    tokenize_fn = partial(
        tokenizer.__call__,
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    assert len(dataset) == len(corpus_wiki)
    for i in range(len(dataset)):
        true = tokenize_fn(corpus_wiki[i])
        pred1, pred2, pred3 = dataset[i]
        assert pred1 == pred2 == true
        assert pred3 == None


def test_wiki_repetition_dataset(filepath_wiki, tokenizer):
    random.seed(0)
    # Create dataset
    dataset = WikiRepetitionDataset(filepath_wiki, tokenizer, 0.5)
    # fmt: off
    true_dataset = [
        ({"input_ids": [0, 4783, 766, 16, 842, 41860, 4717, 261, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
         {"input_ids": [0, 4783, 4783, 766, 16, 842, 842, 41860, 4717, 261, 261, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}),
        ({"input_ids": [0, 9178, 32, 47, 116, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "attention_mask": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]},
         {"input_ids": [0, 9178, 32, 47, 116, 116, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "attention_mask": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}),
        ({"input_ids": [0, 118, 437, 2051, 3392, 47, 4, 8, 47, 116, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
         {"input_ids": [0, 118, 437, 2051, 2051, 3392, 47, 47, 4, 8, 47, 47, 116, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    ]
    # fmt :on
    assert len(dataset) == 3
    for i in range(len(dataset)):
        true1, true2 = true_dataset[i]
        pred1, pred2, pred3 = dataset[i]
        assert pred1 == true1
        assert pred2 == true2
        assert pred3 == None
