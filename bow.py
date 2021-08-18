"""Bag of word model"""
from typing import Any
from evaluate import ExampleRaw, ExamplePreprocessed, Input


def prepare(example: ExampleRaw, param: Any) -> ExamplePreprocessed:
    return ExamplePreprocessed(
        input=Input(text1=example.text1, text2=example.text2), score=example.score
    )


def batcher():
    pass
