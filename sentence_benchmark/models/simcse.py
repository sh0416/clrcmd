from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from sentence_benchmark.data import Input


def prepare(inputs: List[Input], param: Dict) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param["device"] = device
    tokenizer = AutoTokenizer.from_pretrained(param["checkpoint"])
    param["tokenizer"] = tokenizer
    model = AutoModel.from_pretrained(param["checkpoint"])
    model.to(device)
    model.eval()
    param["model"] = model
    return param


def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
    text1 = [" ".join(x.text1) for x in inputs]
    text2 = [" ".join(x.text2) for x in inputs]
    batch1 = param["tokenizer"].batch_encode_plus(
        text1, return_tensors="pt", padding=True
    )
    batch2 = param["tokenizer"].batch_encode_plus(
        text2, return_tensors="pt", padding=True
    )
    batch1 = {k: v.to(param["device"]) for k, v in batch1.items()}
    batch2 = {k: v.to(param["device"]) for k, v in batch2.items()}
    with torch.no_grad():
        outputs1 = param["model"](
            **batch1, output_hidden_states=True, return_dict=True
        )
        outputs2 = param["model"](
            **batch2, output_hidden_states=True, return_dict=True
        )
        # outputs1 = outputs1.last_hidden_state[:, 0, :]
        # outputs2 = outputs2.last_hidden_state[:, 0, :]

        def masked_sum(x: Tensor, mask: Tensor, dim: int) -> Tensor:
            """Sum (masked version)

            :param x: Input tensor
            :type x: Tensor
            :param mask: Mask tensor. Could be broadcastable.
            :type mask: Tensor
            :param dim:
            :type dim: int
            :return: Result of sum
            :rtype: Tensor
            """
            return torch.sum(x * mask.float(), dim=dim)

        def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
            """Mean (masked version)

            :param x: Input tensor
            :type x: Tensor
            :param mask: Mask tensor. Could be broadcastable.
            :type mask: Tensor
            :param dim:
            :type dim: int
            :return: Result of mean
            :rtype: Tensor
            """
            return masked_sum(x, mask, dim) / torch.sum(mask.float(), dim)

        outputs1 = masked_mean(
            outputs1.last_hidden_state,
            batch1["attention_mask"][:, :, None],
            dim=1,
        )
        outputs2 = masked_mean(
            outputs2.last_hidden_state,
            batch2["attention_mask"][:, :, None],
            dim=1,
        )
        score = F.cosine_similarity(outputs1, outputs2, dim=1)
        return score.cpu().numpy()
