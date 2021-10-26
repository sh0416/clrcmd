from typing import Dict

from sentsim.data.sts import (
    load_sickr_test,
    load_sts12,
    load_sts13,
    load_sts14,
    load_sts15,
    load_sts16,
    load_stsb_dev,
    load_stsb_test,
)
from transformers import Trainer
from transformers.utils import logging

from sentsim.evaluator import SemanticTextualSimilarityEvaluator
from sentsim.models.sbert import PytorchSemanticTextualSimilarityModel

logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    def evaluate(
        self, ignore_keys=None, metric_key_prefix="eval", all: bool = False
    ) -> Dict[str, float]:
        evaluator = SemanticTextualSimilarityEvaluator(64)

        self.model.eval()
        model = PytorchSemanticTextualSimilarityModel(self.model.model, self.tokenizer)
        if all:
            metrics = {}
            # STS12
            dataset = load_sts12(f"{self.args.eval_file}/STS12-en-test")
            metrics["STS12"] = evaluator.evaluate(model, dataset)
            # STS13
            dataset = load_sts13(f"{self.args.eval_file}/STS/STS13-en-test")
            metrics["STS13"] = evaluator.evaluate(model, dataset)
            # STS14
            dataset = load_sts14(f"{self.args.eval_file}/STS/STS14-en-test")
            metrics["STS14"] = evaluator.evaluate(model, dataset)
            # STS15
            dataset = load_sts15(f"{self.args.eval_file}/STS/STS15-en-test")
            metrics["STS15"] = evaluator.evaluate(model, dataset)
            # STS16
            dataset = load_sts16(f"{self.args.eval_file}/STS/STS16-en-test")
            metrics["STS16"] = evaluator.evaluate(model, dataset)
            # STSB
            dataset = load_stsb_dev(f"{self.args.eval_file}/STS/STSBenchmark")
            metrics["STSB-dev"] = evaluator.evaluate(model, dataset)
            dataset = load_stsb_test(f"{self.args.eval_file}/STS/STSBenchmark")
            metrics["STSB-test"] = evaluator.evaluate(model, dataset)
            # SICKR
            dataset = load_sickr_test(f"{self.args.eval_file}/SICK")
            metrics["SICKR-test"] = evaluator.evaluate(model, dataset)
        else:
            dataset = load_stsb_dev(f"{self.args.eval_file}/STS/STSBenchmark")
            result = evaluator.evaluate(model, dataset)
            stsb_spearman = result["all_spearman_all"]
            metrics = {"eval_stsb_spearman": stsb_spearman}
        self.model.train()
        self.log(metrics)
        return metrics
