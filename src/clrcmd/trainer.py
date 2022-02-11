from typing import Dict, Union, Optional, List, Any, Tuple
import torch
import torch.nn as nn

from transformers import EvalPrediction, Trainer
from transformers.utils import logging

from clrcmd.data.sts import (
    load_sickr_test,
    load_sts12,
    load_sts13,
    load_sts14,
    load_sts15,
    load_sts16,
    load_stsb_dev,
    load_stsb_test,
)
from scipy.stats import spearmanr

logger = logging.get_logger(__name__)


def compute_metrics(x: EvalPrediction) -> Dict[str, float]:
    return {"spearman": spearmanr(x.predictions, x.label_ids).correlation}


class STSTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            inputs1 = self._prepare_inputs(inputs["inputs1"])
            inputs2 = self._prepare_inputs(inputs["inputs2"])
            label = self._prepare_inputs(inputs["label"])
            score = model.model(inputs1, inputs2)
        return (None, score, label)
