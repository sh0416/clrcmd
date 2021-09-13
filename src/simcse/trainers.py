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
)
from sentence_benchmark.evaluate import evaluate_sts

logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    def evaluate(
        self, ignore_keys=None, metric_key_prefix="eval", all: bool = False
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(inputs: List[Input], param: Dict) -> Dict:
            param["batch_size"] = 64
            return param

        def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
            batch1 = self.tokenizer.batch_encode_plus(
                [x[0] for x in inputs], return_tensors="pt"
            )
            batch2 = self.tokenizer.batch_encode_plus(
                [x[1] for x in inputs], return_tensors="pt"
            )
            batch1 = {k: v.to(self.args.device) for k, v in batch1.items()}
            batch2 = {k: v.to(self.args.device) for k, v in batch2.items()}
            with torch.no_grad():
                score = self.model.compute_similarity(
                    input_ids1=batch1["input_ids"],
                    input_ids2=batch2["input_ids"],
                    attention_mask1=batch1["attention_mask"],
                    attention_mask2=batch2["attention_mask"],
                )
                return score.cpu().numpy()

        self.model.eval()
        if all:
            metrics = {}
            # STS12
            dataset = load_sts12("data/STS/STS12-en-test")
            metrics["STS12"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS13
            dataset = load_sts13("data/STS/STS13-en-test")
            metrics["STS13"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS14
            dataset = load_sts14("data/STS/STS14-en-test")
            metrics["STS14"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS15
            dataset = load_sts15("data/STS/STS15-en-test")
            metrics["STS15"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS16
            dataset = load_sts16("data/STS/STS16-en-test")
            metrics["STS16"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STSB
            dataset = load_stsb_dev("data/STS/STSBenchmark")
            metrics["STSB-dev"] = evaluate_sts(dataset, {}, prepare, batcher)
            dataset = load_stsb_test("data/STS/STSBenchmark")
            metrics["STSB-test"] = evaluate_sts(dataset, {}, prepare, batcher)
        else:
            dataset = load_stsb_dev("data/STS/STSBenchmark")
            result = evaluate_sts(dataset, {}, prepare, batcher)
            stsb_spearman = result["all"]["spearman"]["all"]
            metrics = {"eval_stsb_spearman": stsb_spearman}
        self.log(metrics)
        return metrics
