from typing import Dict, List

import numpy as np
import torch
from transformers import RobertaTokenizer

from sentence_benchmark.data import Input
from simcse.models import create_contrastive_learning


def prepare(inputs: List[Input], param: Dict) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param["device"] = device
    if param.get("checkpoint", None) is not None:
        tokenizer = RobertaTokenizer.from_pretrained(param["checkpoint"])
        model = create_contrastive_learning(param["model_args"])
        model.to(device)
        param["tokenizer"] = tokenizer
        param["model"] = model
    assert "tokenizer" in param
    assert "model" in param
    assert "device" in param
    return param


def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
    sentence = [x[0] for x in inputs] + [x[1] for x in inputs]
    batch = param["tokenizer"].batch_encode_plus(
        sentence, return_tensors="pt", padding=True
    )
    batch1 = {k: v[: len(inputs)].to(param["device"]) for k, v in batch.items()}
    batch2 = {k: v[len(inputs) :].to(param["device"]) for k, v in batch.items()}
    param["model"].eval()
    with torch.no_grad():
        score = param["model"].compute_similarity(inputs1=batch1, inputs2=batch2)
    param["model"].train()
    return score.cpu().numpy()
