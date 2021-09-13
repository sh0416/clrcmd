from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from sentence_benchmark.data import Input
from sentence_benchmark.utils import masked_mean


def prepare(inputs: List[Input], param: Dict) -> Dict:
    if param["checkpoint"] is None:
        checkpoint = "princeton-nlp/unsup-simcse-roberta-base"
    else:
        checkpoint = param["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    param["tokenizer"] = tokenizer
    param["device"] = device
    param["model"] = model
    return param


def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
    tokenizer, model, device = param["tokenizer"], param["model"], param["device"]
    batch1 = tokenizer.batch_encode_plus(
        [x[0] for x in inputs], return_tensors="pt", padding=True
    )
    batch2 = tokenizer.batch_encode_plus(
        [x[1] for x in inputs], return_tensors="pt", padding=True
    )
    batch1 = {k: v.to(device) for k, v in batch1.items()}
    batch2 = {k: v.to(device) for k, v in batch2.items()}
    with torch.no_grad():
        outputs1 = model(**batch1, output_hidden_states=True, return_dict=True)
        outputs2 = model(**batch2, output_hidden_states=True, return_dict=True)

        if param["pooler_type"] == "avg":
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
        elif param["pooler_type"] == "rwmd":
            pairwise_sim = F.cosine_similarity(
                outputs1.last_hidden_state[:, :, None, :],
                outputs2.last_hidden_state[:, None, :, :],
                dim=3,
            )
            output1 = torch.max(pairwise_sim, dim=2)[0].mean(dim=1)
            output2 = torch.max(pairwise_sim, dim=1)[0].mean(dim=1)
            score = torch.where(output1 > output2, output1, output2)
        return score.cpu().numpy()
