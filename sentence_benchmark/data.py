import csv
from typing import Dict, List, NamedTuple, Tuple


class Input(NamedTuple):
    text1: List[str]
    text2: List[str]


class Example(NamedTuple):
    input: Input
    score: float


def tokenize(row: Tuple[str, str]) -> Tuple[List[str], List[str]]:
    return row[0].split(), row[1].split()


def detokenize(row: Tuple[List[str], List[str]]) -> Tuple[str, str]:
    return " ".join(row[0]), " ".join(row[1])


def load_data_sts(filepaths: Tuple[str, str]) -> List[Example]:
    filepath_input, filepath_label = filepaths
    with open(filepath_input) as f_input, open(filepath_label) as f_label:
        reader_input = csv.reader(
            f_input, delimiter="\t", quoting=csv.QUOTE_NONE
        )
        reader_label = csv.reader(
            f_label, delimiter="\t", quoting=csv.QUOTE_NONE
        )
        reader_input = map(tokenize, reader_input)
        reader_input = map(
            lambda x: Input(text1=x[0], text2=x[1]), reader_input
        )
        reader_label = map(
            lambda x: float(x[0]) if len(x) > 0 else None, reader_label
        )
        reader = filter(
            lambda x: x[1] is not None, zip(reader_input, reader_label)
        )
        dataset = map(lambda x: Example(input=x[0], score=x[1]), reader)
        dataset = list(dataset)
        for idx, row in enumerate(dataset, start=1):
            assert (
                len(row.input.text1) > 0
            ), f"Text1 is empty: Row {idx}: {row}"
            assert (
                len(row.input.text2) > 0
            ), f"Text2 is empty: Row {idx}: {row}"
        return dataset


def create_filepaths_sts(
    dirpath: str, sources: List[str]
) -> List[Tuple[str, str]]:
    filenames_input = map(lambda x: f"STS.input.{x}.txt", sources)
    filenames_label = map(lambda x: f"STS.gs.{x}.txt", sources)
    filepaths_input = map(lambda x: f"{dirpath}/{x}", filenames_input)
    filepaths_label = map(lambda x: f"{dirpath}/{x}", filenames_label)
    return list(zip(filepaths_input, filepaths_label))


def load_sources(dirpath: str, sources: List[str]) -> Dict[str, List[Example]]:
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, sources))
    return {k: v for k, v in zip(sources, datasets)}


def save_dataset(dirpath: str, dataset: Dict[str, List[Example]]) -> None:
    sources = list(dataset.keys())
    filepaths = create_filepaths_sts(dirpath, sources)
    for source, (filepath_input, filepath_label) in zip(sources, filepaths):
        _dataset = dataset[source]
        with open(filepath_input, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter="\t", quotechar=None, quoting=csv.QUOTE_NONE
            )
            writer.writerows(
                (
                    detokenize((row.input.text1, row.input.text2))
                    for row in _dataset
                )
            )
        with open(filepath_label, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter="\t", quotechar=None, quoting=csv.QUOTE_NONE
            )
            writer.writerows((row.score,) for row in _dataset)


def load_sts12(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "MSRpar",
        "MSRvid",
        "SMTeuroparl",
        "surprise.OnWN",
        "surprise.SMTnews",
    ]
    dataset = load_sources(dirpath, sources)
    assert len(dataset["MSRpar"]) == 750, len(dataset["MSRpar"])
    assert len(dataset["MSRvid"]) == 750, len(dataset["MSRvid"])
    assert len(dataset["SMTeuroparl"]) == 459, len(dataset["SMTeuroparl"])
    assert len(dataset["surprise.OnWN"]) == 750, len(dataset["surprise.OnWN"])
    assert len(dataset["surprise.SMTnews"]) == 399, len(
        dataset["surprise.SMTnews"]
    )
    return dataset


def load_sts13(dirpath: str) -> Dict[str, List[Example]]:
    sources = ["FNWN", "headlines", "OnWN"]
    dataset = load_sources(dirpath, sources)
    assert len(dataset["FNWN"]) == 189
    assert len(dataset["headlines"]) == 750
    assert len(dataset["OnWN"]) == 561
    return dataset


def load_sts14(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "deft-forum",
        "deft-news",
        "headlines",
        "images",
        "OnWN",
        "tweet-news",
    ]
    dataset = load_sources(dirpath, sources)
    assert len(dataset["deft-forum"]) == 450
    assert len(dataset["deft-news"]) == 300
    assert len(dataset["headlines"]) == 750
    assert len(dataset["images"]) == 750
    assert len(dataset["OnWN"]) == 750
    assert len(dataset["tweet-news"]) == 750
    return dataset


def load_sts15(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "answers-forums",
        "answers-students",
        "belief",
        "headlines",
        "images",
    ]
    dataset = load_sources(dirpath, sources)
    assert len(dataset["answers-forums"]) == 375, len(
        dataset["answers-forums"]
    )
    assert len(dataset["answers-students"]) == 750, len(
        dataset["answers-students"]
    )
    assert len(dataset["belief"]) == 375, len(dataset["belief"])
    assert len(dataset["headlines"]) == 750, len(dataset["headlines"])
    assert len(dataset["images"]) == 750, len(dataset["images"])
    return dataset


def load_sts16(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "answer-answer",
        "headlines",
        "plagiarism",
        "postediting",
        "question-question",
    ]
    dataset = load_sources(dirpath, sources)
    assert len(dataset["answer-answer"]) == 254
    assert len(dataset["headlines"]) == 249
    assert len(dataset["plagiarism"]) == 230
    assert len(dataset["postediting"]) == 244
    assert len(dataset["question-question"]) == 209
    return dataset
