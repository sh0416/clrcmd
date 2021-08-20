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


def load_data_sts(filepaths: Tuple[str, str]) -> List[Example]:
    filepath_input, filepath_label = filepaths
    with open(filepath_input) as f_input, open(filepath_label) as f_label:
        reader_input = csv.reader(f_input, delimiter="\t", quoting=csv.QUOTE_NONE)
        reader_label = csv.reader(f_label, delimiter="\t", quoting=csv.QUOTE_NONE)
        reader_input = map(tokenize, reader_input)
        reader_input = map(lambda x: Input(text1=x[0], text2=x[1]), reader_input)
        reader_label = map(lambda x: float(x[0]) if len(x) > 0 else None, reader_label)
        reader = filter(lambda x: x[1] is not None, zip(reader_input, reader_label))
        dataset = map(lambda x: Example(input=x[0], score=x[1]), reader)
        dataset = list(dataset)
        for idx, row in enumerate(dataset, start=1):
            assert len(row.input.text1) > 0, f"Text1 is empty: Row {idx}: {row}"
            assert len(row.input.text2) > 0, f"Text2 is empty: Row {idx}: {row}"
        # Sort data by length to minimize padding in batcher
        dataset = sorted(
            dataset, key=lambda x: (len(x.input.text1), len(x.input.text2))
        )
        return dataset


def create_filepaths_sts(dirpath: str, datasets: List[str]) -> List[Tuple[str, str]]:
    filenames_input = map(lambda x: f"STS.input.{x}.txt", datasets)
    filenames_label = map(lambda x: f"STS.gs.{x}.txt", datasets)
    filepaths_input = map(lambda x: f"{dirpath}/{x}", filenames_input)
    filepaths_label = map(lambda x: f"{dirpath}/{x}", filenames_label)
    return list(zip(filepaths_input, filepaths_label))


def load_sources(dirpath: str, sources: List[str]) -> Dict[str, List[Example]]:
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, sources))
    return {k: v for k, v in zip(sources, datasets)}


def load_sts12(dirpath: str) -> Dict[str, List[Example]]:
    sources = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.OnWN", "surprise.SMTnews"]
    datasets = load_sources(dirpath, sources)
    assert len(datasets["MSRpar"]) == 750, len(datasets["MSRpar"])
    assert len(datasets["MSRvid"]) == 750, len(datasets["MSRvid"])
    assert len(datasets["SMTeuroparl"]) == 459, len(datasets["SMTeuroparl"])
    assert len(datasets["surprise.OnWN"]) == 750, len(datasets["surprise.OnWN"])
    assert len(datasets["surprise.SMTnews"]) == 399, len(datasets["surprise.SMTnews"])
    return datasets


def load_sts13(dirpath: str) -> Dict[str, List[Example]]:
    sources = ["FNWN", "headlines", "OnWN"]
    datasets = load_sources(dirpath, sources)
    return datasets


def load_sts14(dirpath: str) -> Dict[str, List[Example]]:
    sources = ["deft-forum", "deft-news", "headlines", "images", "OnWN", "tweet-news"]
    datasets = load_sources(dirpath, sources)
    return datasets


def load_sts15(dirpath: str) -> Dict[str, List[Example]]:
    sources = ["answers-forums", "answers-students", "belief", "headlines", "images"]
    datasets = load_sources(dirpath, sources)
    assert len(datasets["answers-forums"]) == 375, len(datasets["answers-forums"])
    assert len(datasets["answers-students"]) == 750, len(datasets["answers-students"])
    assert len(datasets["belief"]) == 375, len(datasets["belief"])
    assert len(datasets["headlines"]) == 750, len(datasets["headlines"])
    assert len(datasets["images"]) == 750, len(datasets["images"])
    return datasets


def load_sts16(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "answer-answer",
        "headlines",
        "plagiarism",
        "postediting",
        "question-question",
    ]
    datasets = load_sources(dirpath, sources)
    return datasets
