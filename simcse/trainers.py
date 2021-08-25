from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.utils import logging

from sentence_benchmark.evaluate import evaluate_sts

import numpy as np

from sentence_benchmark.data import Input, load_stsb_dev

logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
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
                outputs1 = self.model(
                    **batch1,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                )
                outputs2 = self.model(
                    **batch2,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                )
                outputs1 = outputs1.pooler_output
                outputs2 = outputs2.pooler_output
                score = F.cosine_similarity(outputs1, outputs2, dim=1)
                return score.cpu().numpy()

        # NOTE: I don't know how to evaluate SICK-R because it train something
        self.model.eval()
        dataset_stsb = load_stsb_dev(".data/STS/STSBenchmark")
        # dataset_sickr = load_sickr_dev(".data/SICK")
        result_stsb = evaluate_sts(dataset_stsb, {}, prepare, batcher)
        # result_sickr = evaluate_sts(dataset_sickr, {}, prepare, batcher)

        stsb_spearman = result_stsb["all"]["spearman"]["mean"]
        # sickr_spearman = result_sickr["all"]["spearman"][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            #    "eval_sickr_spearman": sickr_spearman,
        }
        self.log(metrics)
        return metrics
