import os

import pytest
import torch

from simcse.eval.ists import (
    convert_alignment_granularity,
    extract_alignment_from_heatmap,
    load_chunked_sentence_pairs,
    load_sentence_pairs,
)


@pytest.fixture
def filepath_sent1(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent1.txt")
    tmpfile.write(
        """both bulbs a and c still have a closed path
terminal 1 and the positive terminal are connected.
positive battery is seperated by a gap from terminal 2
There is no difference between the two terminals.
the switch has to be contained in the same path as the bulb and the battery"""
    )
    return tmpfile.strpath


@pytest.fixture
def filepath_sent2(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent2.txt")
    tmpfile.write(
        """Bulbs A and C are still in closed paths
Terminal 1 and the positive terminal are separated by the gap
Terminal 2 and the positive terminal are separated by the gap
The terminals are in the same state.
The switch and the bulb have to be in the same path."""
    )
    return tmpfile.strpath


@pytest.fixture
def filepath_sent1_chunk(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent1.chunk.txt")
    tmpfile.write(
        """[ both ] [ bulbs a and c ] [ still ] [ have ] [ a closed path ]
[ terminal 1 and the positive terminal ] [ are connected. ]
[ positive battery ] [ is seperated ] [ by a gap ] [ from terminal 2 ]
[ There ] [ is ] [ no difference ] [ between the two terminals. ]
[ the switch ] [ has to be contained ] [ in the same path ] [ as ] [ the bulb and the battery ]"""
    )
    return tmpfile.strpath


@pytest.fixture
def filepath_sent2_chunk(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent2.chunk.txt")
    tmpfile.write(
        """[ Bulbs A and C ] [ are ] [ still ] [ in closed paths ]
[ Terminal 1 and the positive terminal ] [ are separated ] [ by the gap ]
[ Terminal 2 and the positive terminal ] [ are separated ] [ by the gap ]
[ The terminals ] [ are ] [ in the same state. ]
[ The switch and the bulb ] [ have to be ] [ in the same path ]"""
    )
    return tmpfile.strpath


def test_load_sentence_pairs(tmpdir, filepath_sent1, filepath_sent2):
    sentence_pairs = load_sentence_pairs(tmpdir.strpath, "answers-students")
    with open(filepath_sent1) as f:
        sent1 = [x.strip() for x in f]
    with open(filepath_sent2) as f:
        sent2 = [x.strip() for x in f]
    assert sentence_pairs == list(zip(sent1, sent2))


def test_load_chunked_sentence_pairs(
    tmpdir, filepath_sent1_chunk, filepath_sent2_chunk
):
    pred = load_chunked_sentence_pairs(tmpdir.strpath, "answers-students")
    # fmt: off
    true = [
        (["both", "bulbs a and c", "still", "have", "a closed path"],
         ["Bulbs A and C", "are", "still", "in closed paths"]),
        (["terminal 1 and the positive terminal", "are connected."],
         ["Terminal 1 and the positive terminal", "are separated", "by the gap"]),
        (["positive battery", "is seperated", "by a gap", "from terminal 2"],
         ["Terminal 2 and the positive terminal", "are separated", "by the gap"]),
        (["There", "is", "no difference", "between the two terminals."],
         ["The terminals", "are", "in the same state."]),
        (["the switch", "has to be contained", "in the same path", "as", "the bulb and the battery"],
         ["The switch and the bulb", "have to be", "in the same path"])
    ]
    # fmt: on
    assert pred == true


def test_extract_alignment_from_heatmap():
    torch.manual_seed(0)
    heatmap = torch.rand(4, 6)
    pred = extract_alignment_from_heatmap(heatmap)
    true = [([0], [1]), ([1], [1]), ([2], [5]), ([3], [3])]
    assert pred == true
