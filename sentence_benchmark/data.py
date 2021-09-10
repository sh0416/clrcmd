import csv
import os
from typing import Dict, List, NamedTuple, Tuple

Input = Tuple[str, str]


class Example(NamedTuple):
    input: Input
    score: float


def _check_dataset(dataset: List[Example]):
    for idx, example in enumerate(dataset, start=1):
        assert len(example.input[0]) > 0, f"{idx = }: {example = }"
        assert len(example.input[1]) > 0, f"{idx = }: {example = }"
        assert type(example.score) == float, f"{idx = }: {example = }"


def load_data_sickr(filepath: str) -> List[Example]:
    """Load file which follows SICKR format

    :param filepath: Filepath where the data is stored
    :return: List of Examples
    """

    def _create(row: Tuple[str, ...]) -> Example:
        return Example(input=(row[1], row[2]), score=float(row[3]))

    with open(filepath) as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        # Skip first line
        _ = next(reader)
        dataset = list(map(_create, reader))
    _check_dataset(dataset)
    return dataset


def load_data_stsb(filepath: str) -> List[Example]:
    """Load file which follows STSB format

    :param filepath: Filepath where the data is stored
    :return: List of Examples
    """

    def _create(row: Tuple[str, ...]) -> Example:
        return Example(input=(row[5], row[6]), score=float(row[4]))

    with open(filepath) as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        dataset = list(map(_create, reader))
    _check_dataset(dataset)
    return dataset


def load_data_sts(filepaths: Tuple[str, str]) -> List[Example]:
    """Load file which follows STS format

    :param filepaths: Filepaths corresponding to input and label
    :return: List of Examples
    """

    def _create(row: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> Example:
        row_input, row_label = row
        return Example(input=row_input, score=float(row_label[0]))

    filepath_input, filepath_label = filepaths
    with open(filepath_input) as f_input, open(filepath_label) as f_label:
        reader_input = csv.reader(f_input, delimiter="\t", quoting=csv.QUOTE_NONE)
        reader_label = csv.reader(f_label, delimiter="\t", quoting=csv.QUOTE_NONE)
        reader = filter(lambda x: len(x[1]) > 0, zip(reader_input, reader_label))
        dataset = list(map(_create, reader))
    _check_dataset(dataset)
    return dataset


def create_filepaths_sts(dirpath: str, sources: List[str]) -> List[Tuple[str, str]]:
    """Given sources, create list of filepaths

    :param dirpath: Directory where the whole files located
    :param sources: List of source names
    :return: List of tuple of filepaths where the data files located
    """
    filenames_input = map(lambda x: f"STS.input.{x}.txt", sources)
    filenames_label = map(lambda x: f"STS.gs.{x}.txt", sources)
    filepaths_input = map(lambda x: os.path.join(dirpath, x), filenames_input)
    filepaths_label = map(lambda x: os.path.join(dirpath, x), filenames_label)
    return list(zip(filepaths_input, filepaths_label))


def load_sources_sts(dirpath: str, sources: List[str]) -> Dict[str, List[Example]]:
    """Give sources, load dataset. Assume that the files follow STS format

    :param dirpath: Directory where the whole files located
    :param sources: List of source names
    :return: Dict of dataset that maps source name to list of examples
    """
    datasets = map(load_data_sts, create_filepaths_sts(dirpath, sources))
    return {k: v for k, v in zip(sources, datasets)}


def save_dataset(dirpath: str, dataset: Dict[str, List[Example]]) -> None:
    """Given dataset, save the dataset in dirpath

    :param dirpath: Directory where the dataset will be saved
    :type dirpath: str
    :param dataset: Dataset that will be saved
    :type dataset: Dict[str, List[Example]]
    """
    sources = list(dataset.keys())
    filepaths = create_filepaths_sts(dirpath, sources)
    for source, (filepath_input, filepath_label) in zip(sources, filepaths):
        _dataset = dataset[source]
        with open(filepath_input, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter="\t", quotechar=None, quoting=csv.QUOTE_NONE
            )
            writer.writerows(
                (example.input[0], example.input[1]) for example in _dataset
            )
        with open(filepath_label, "w", newline="") as f:
            writer = csv.writer(
                f, delimiter="\t", quotechar=None, quoting=csv.QUOTE_NONE
            )
            writer.writerows((example.score,) for example in _dataset)


def load_sts12(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "MSRpar",
        "MSRvid",
        "SMTeuroparl",
        "surprise.OnWN",
        "surprise.SMTnews",
    ]
    dataset = load_sources_sts(dirpath, sources)
    assert len(dataset["MSRpar"]) == 750, len(dataset["MSRpar"])
    assert len(dataset["MSRvid"]) == 750, len(dataset["MSRvid"])
    assert len(dataset["SMTeuroparl"]) == 459, len(dataset["SMTeuroparl"])
    assert len(dataset["surprise.OnWN"]) == 750, len(dataset["surprise.OnWN"])
    assert len(dataset["surprise.SMTnews"]) == 399, len(dataset["surprise.SMTnews"])
    return dataset


def load_sts13(dirpath: str) -> Dict[str, List[Example]]:
    sources = ["FNWN", "headlines", "OnWN"]
    dataset = load_sources_sts(dirpath, sources)
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
    dataset = load_sources_sts(dirpath, sources)
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
    dataset = load_sources_sts(dirpath, sources)
    assert len(dataset["answers-forums"]) == 375
    assert len(dataset["answers-students"]) == 750
    assert len(dataset["belief"]) == 375
    assert len(dataset["headlines"]) == 750
    assert len(dataset["images"]) == 750
    return dataset


def load_sts16(dirpath: str) -> Dict[str, List[Example]]:
    sources = [
        "answer-answer",
        "headlines",
        "plagiarism",
        "postediting",
        "question-question",
    ]
    dataset = load_sources_sts(dirpath, sources)
    assert len(dataset["answer-answer"]) == 254
    assert len(dataset["headlines"]) == 249
    assert len(dataset["plagiarism"]) == 230
    assert len(dataset["postediting"]) == 244
    assert len(dataset["question-question"]) == 209
    return dataset


def load_stsb_train(dirpath: str) -> Dict[str, List[Example]]:
    return {"train": load_data_stsb(os.path.join(dirpath, "sts-train.csv"))}


def load_stsb_dev(dirpath: str) -> Dict[str, List[Example]]:
    return {"dev": load_data_stsb(os.path.join(dirpath, "sts-dev.csv"))}


def load_stsb_test(dirpath: str) -> Dict[str, List[Example]]:
    return {"test": load_data_stsb(os.path.join(dirpath, "sts-test.csv"))}


def load_sickr_train(dirpath: str) -> Dict[str, List[Example]]:
    return {"train": load_data_sickr(os.path.join(dirpath, "SICK_train.txt"))}


def load_sickr_dev(dirpath: str) -> Dict[str, List[Example]]:
    return {"dev": load_data_sickr(os.path.join(dirpath, "SICK_trial.txt"))}


def load_sickr_test(dirpath: str) -> Dict[str, List[Example]]:
    filepath = os.path.join(dirpath, "SICK_test_annotated.txt")
    return {"test": load_data_sickr(filepath)}
