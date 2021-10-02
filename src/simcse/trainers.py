from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.utils import logging

from sentence_benchmark.data import (
    Input,
    load_sts12,
    load_sts13,
    load_sts14,
    load_sts15,
    load_sts16,
    load_stsb_dev,
    load_stsb_test,
    load_sickr_test,
)
from sentence_benchmark.evaluate import evaluate_sts
from sentence_benchmark.models.rwmdcse import prepare, batcher

logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    def evaluate(
        self, ignore_keys=None, metric_key_prefix="eval", all: bool = False
    ) -> Dict[str, float]:

        param = {"batch_size": 64, "model": self.model, "tokenizer": self.tokenizer}
        self.model.eval()
        if all:
            metrics = {}
            # STS12
            dataset = load_sts12("/nas/home/sh0416/data/STS/STS12-en-test")
            metrics["STS12"] = evaluate_sts(dataset, param, prepare, batcher)
            # STS13
            dataset = load_sts13("/nas/home/sh0416/data/STS/STS13-en-test")
            metrics["STS13"] = evaluate_sts(dataset, param, prepare, batcher)
            # STS14
            dataset = load_sts14("/nas/home/sh0416/data/STS/STS14-en-test")
            metrics["STS14"] = evaluate_sts(dataset, param, prepare, batcher)
            # STS15
            dataset = load_sts15("/nas/home/sh0416/data/STS/STS15-en-test")
            metrics["STS15"] = evaluate_sts(dataset, param, prepare, batcher)
            # STS16
            dataset = load_sts16("/nas/home/sh0416/data/STS/STS16-en-test")
            metrics["STS16"] = evaluate_sts(dataset, param, prepare, batcher)
            # STSB
            dataset = load_stsb_dev("/nas/home/sh0416/data/STS/STSBenchmark")
            metrics["STSB-dev"] = evaluate_sts(dataset, param, prepare, batcher)
            dataset = load_stsb_test("/nas/home/sh0416/data/STS/STSBenchmark")
            metrics["STSB-test"] = evaluate_sts(dataset, param, prepare, batcher)
            # SICKR
            dataset = load_sickr_test("/nas/home/sh0416/data/SICK")
            metrics["SICKR-test"] = evaluate_sts(dataset, param, prepare, batcher)
        else:
            dataset = load_stsb_dev("/nas/home/sh0416/data/STS/STSBenchmark")
            result = evaluate_sts(dataset, param, prepare, batcher)
            stsb_spearman = result["all"]["spearman"]["all"]
            metrics = {"eval_stsb_spearman": stsb_spearman}
        self.model.train()
        self.log(metrics)
        return metrics
