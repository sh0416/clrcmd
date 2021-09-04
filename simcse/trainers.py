from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
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
            return param

        def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
            text1 = [" ".join(x.text1) for x in inputs]
            text2 = [" ".join(x.text2) for x in inputs]
            batch1 = self.tokenizer.batch_encode_plus(
                text1, return_tensors="pt", padding=True
            )
            batch2 = self.tokenizer.batch_encode_plus(
                text2, return_tensors="pt", padding=True
            )
            batch1 = {k: v.to(self.args.device) for k, v in batch1.items()}
            batch2 = {k: v.to(self.args.device) for k, v in batch2.items()}
            with torch.no_grad():
                outputs1 = self.model(**batch1, sent_emb=True)
                outputs2 = self.model(**batch2, sent_emb=True)
                input_ids1 = batch1["input_ids"]
                input_ids2 = batch2["input_ids"]
                pad_id = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.pad_token
                )
                input_ids1_valid = input_ids1 != pad_id
                input_ids1_valid = input_ids1_valid & (input_ids1 < 10)
                input_ids2_valid = input_ids2 != pad_id
                input_ids2_valid = input_ids2_valid & (input_ids2 < 10)
                input_mask = input_ids1[:, :, None] == input_ids2[:, None, :]
                input_mask = input_mask & input_ids1_valid[:, :, None]
                input_mask = input_mask & input_ids2_valid[:, None, :]
                assert torch.all(input_mask[:, 0, 0])
                # (batch_size, seq_len, seq_len)
                outputs1 = outputs1.last_hidden_state
                outputs2 = outputs2.last_hidden_state
                # (batch_size, seq_len, hidden_size)
                score_pair = F.cosine_similarity(
                    outputs1[:, :, None, :], outputs2[:, None, :, :], dim=3
                )
                # (batch_size, seq_len, seq_len)
                input_mask = input_mask.view(input_mask.shape[0], -1).float()
                score_pair = score_pair.view(score_pair.shape[0], -1)
                # (batch_size, seq_len*seq_len)
                score = (score_pair * input_mask).sum(dim=1)
                score = score / input_mask.sum(dim=1)
                # score = score_pair[:, 0]
                return score.cpu().numpy()

        self.model.eval()
        if all:
            metrics = {}
            # STS12
            dataset = load_sts12(".data/STS/STS12-en-test")
            metrics["STS12"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS13
            dataset = load_sts13(".data/STS/STS13-en-test")
            metrics["STS13"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS14
            dataset = load_sts14(".data/STS/STS14-en-test")
            metrics["STS14"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS15
            dataset = load_sts15(".data/STS/STS15-en-test")
            metrics["STS15"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STS16
            dataset = load_sts16(".data/STS/STS16-en-test")
            metrics["STS16"] = evaluate_sts(dataset, {}, prepare, batcher)
            # STSB
            dataset = load_stsb_dev(".data/STS/STSBenchmark")
            metrics["STSB-dev"] = evaluate_sts(dataset, {}, prepare, batcher)
            dataset = load_stsb_test(".data/STS/STSBenchmark")
            metrics["STSB-test"] = evaluate_sts(dataset, {}, prepare, batcher)
        else:
            dataset = load_stsb_dev(".data/STS/STSBenchmark")
            result = evaluate_sts(dataset, {}, prepare, batcher)
            stsb_spearman = result["all"]["spearman"]["all"]
            metrics = {"eval_stsb_spearman": stsb_spearman}
        self.log(metrics)
        return metrics
