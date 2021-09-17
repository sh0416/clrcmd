import pytest

from transformers import RobertaTokenizer
from src.simcse.data.dataset import ContrastiveLearningDataset


@pytest.fixture
def corpus():
    return ["my name is seonghyeon", "how are you?", "i'm fine thank you. and you?"]


def test_tmpdir(tmpdir, corpus):
    tmpdir.join("corpus.txt").write("\n".join(corpus))
    assert tmpdir.join("corpus.txt").read() == "\n".join(corpus)


def test_contrastive_learning_dataset(tmpdir, corpus):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    filename = "corpus.txt"
    tmpfile = tmpdir.join(filename)
    tmpfile.write("\n".join(corpus))
    filepath = tmpfile.strpath
    dataset = ContrastiveLearningDataset(filepath, tokenizer)
    print(dataset[0])
