import logging
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel

from simcse.config import ModelArguments
from simcse.utils import masked_mean

logger = logging.getLogger(__name__)


class Pooler(nn.Module):
    """Poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type: str):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask: Tensor, hidden_states: Tuple[Tensor]) -> Tensor:
        if self.pooler_type == "cls":
            return hidden_states[-1][:, 0]
        elif self.pooler_type == "avg":
            last_hidden = hidden_states[-1]
            return masked_mean(last_hidden, attention_mask[:, :, None], dim=1)
        elif self.pooler_type == "avg_first_last":
            hidden = (hidden_states[0] + hidden_states[-1]) / 2.0
            return masked_mean(hidden, attention_mask[:, :, None], dim=1)
        elif self.pooler_type == "avg_top2":
            hidden = (hidden_states[-1] + hidden_states[-2]) / 2.0
            return masked_mean(hidden, attention_mask[:, :, None], dim=1)
        else:
            raise NotImplementedError()


def dist_all_gather(x: Tensor) -> Tensor:
    """Boilerplate code for all gather in distributed setting

    The first dimension could be different

    :param x: Tensor to be gathered
    :type x: Tensor
    :return: Tensor after gathered. For the gradient flow, current rank is
             replaced to original tensor
    :rtype: Tensor
    """
    assert dist.is_initialized(), "The process is not in DDP setting"
    world_size = dist.get_world_size()
    # 1. Get size acroess processes
    x_numel_list = [torch.tensor(x.numel(), device=x.device) for _ in range(world_size)]
    dist.all_gather(x_numel_list, torch.tensor(x.numel(), device=x.device))
    # 2. Infer maximum size
    max_size = max(x.item() for x in x_numel_list)
    # 3. Communitcate tensor with padded version
    _x_list = [torch.empty((max_size,), device=x.device) for _ in range(world_size)]
    _x = torch.cat(
        (
            x.contiguous().view(-1),
            torch.empty((max_size - x.numel(),), device=x.device),
        )
    )
    dist.all_gather(_x_list, _x)
    # 4. Remove padded data to change original shape
    x_list = [_x[:n].view(-1, *x.shape[1:]) for n, _x in zip(x_numel_list, _x_list)]
    # Since `all_gather` results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    x_list[dist.get_rank()] = x
    return torch.cat(x_list, dim=0)


class ContrastiveLearning(nn.Module):
    def __init__(
        self, config: PretrainedConfig, backbone: PreTrainedModel, temp: float
    ):
        super().__init__()
        self.backbone = backbone
        self.cl_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()
        )
        self.temp = temp

    def forward(
        self,
        inputs1: Dict[str, Tensor],
        inputs2: Dict[str, Tensor],
        inputs_neg: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor]:
        sim = self._compute_batchwise_similarity(inputs1, inputs2, inputs_neg)
        sim = sim / self.temp
        # (batch_size, batch_size)
        labels = torch.arange(sim.shape[0], dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return (loss,)

    def _compute_representation(
        self,
        inputs1: Dict[str, Tensor],
        inputs2: Dict[str, Tensor],
        inputs_neg: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tuple[Tensor], Tuple[Tensor], Optional[Tuple[Tensor]]]:
        if inputs_neg is not None:
            input_ids = (
                inputs1["input_ids"],
                inputs2["input_ids"],
                inputs_neg["input_ids"],
            )
            attention_mask = (
                inputs1["attention_mask"],
                inputs2["attention_mask"],
                inputs_neg["attention_mask"],
            )
        else:
            input_ids = (inputs1["input_ids"], inputs2["input_ids"])
            attention_mask = (inputs1["attention_mask"], inputs2["attention_mask"])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        outputs = self.backbone(input_ids, attention_mask)
        hidden_states1, hidden_states2 = [], []
        if inputs_neg is not None:
            hidden_states_neg = []
        else:
            hidden_states_neg = None
        for x in outputs.hidden_states:
            if inputs_neg is not None:
                hidden_state1, hidden_state2, hidden_state_neg = torch.chunk(x, 3)
            else:
                hidden_state1, hidden_state2 = torch.chunk(x, 2)
            hidden_states1.append(hidden_state1)
            hidden_states2.append(hidden_state2)
            if inputs_neg is not None:
                hidden_states_neg.append(hidden_state_neg)
        return hidden_states1, hidden_states2, hidden_states_neg


class SimpleContrastiveLearning(ContrastiveLearning):
    def __init__(
        self,
        config: PretrainedConfig,
        backbone: PreTrainedModel,
        temp: float,
        loss_mlm: bool,
        pooler_type: str,
    ):
        super().__init__(config, backbone, temp)
        self._pooler = Pooler(pooler_type)
        if loss_mlm:
            self.lm_head = RobertaLMHead(config)

    def _compute_batchwise_similarity(
        self,
        inputs1: Dict[str, Tensor],
        inputs2: Dict[str, Tensor],
        inputs_neg: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        outputs1, outputs2, outputs_neg = self._compute_representation(
            inputs1, inputs2, inputs_neg
        )
        outputs1 = self._pooler(inputs1["attention_mask"], outputs1)
        outputs2 = self._pooler(inputs2["attention_mask"], outputs2)
        if outputs_neg is not None:
            outputs_neg = self._pooler(inputs_neg["attention_mask"], outputs_neg)
        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            outputs1, outputs2 = dist_all_gather(outputs1), dist_all_gather(outputs2)
            if outputs_neg is not None:
                outputs_neg = dist_all_gather(outputs_neg)
        sim = F.cosine_similarity(outputs1[:, None, :], outputs2[None, :, :], dim=2)
        if outputs_neg is not None:
            # (batch_size, hidden_dim) x (batch_size, hidden_dim)
            sim_neg = F.cosine_similarity(outputs1, outputs_neg, dim=1)
            # (batch_size)
            sim = torch.cat((sim, sim_neg[:, None]), dim=1)
            # (batch_size, hidden_dim + 1)
        return sim

    def compute_similarity(
        self, inputs1: Dict[str, Tensor], inputs2: Dict[str, Tensor]
    ) -> Tensor:
        outputs1, outputs2, _ = self._compute_representation(inputs1, inputs2)
        outputs1 = self._pooler(inputs1["attention_mask"], outputs1)
        outputs2 = self._pooler(inputs2["attention_mask"], outputs2)
        score = F.cosine_similarity(outputs1, outputs2)
        return score


class TokenContrastiveLearning(ContrastiveLearning):
    def __init__(
        self,
        config: PretrainedConfig,
        backbone: PreTrainedModel,
        temp: float,
        loss_mlm: bool,
        layer_idx: Optional[int],
    ):
        super().__init__(config, backbone, temp)
        self.layer_idx = -1 if layer_idx is None else layer_idx
        if loss_mlm:
            self.lm_head = RobertaLMHead(config)

    def compute_similarity(
        self, inputs1: Dict[str, Tensor], inputs2: Dict[str, Tensor]
    ) -> Tensor:
        outputs1, outputs2, _ = self._compute_representation(inputs1, inputs2)
        outputs1 = self.cl_head(outputs1[self.layer_idx])
        outputs2 = self.cl_head(outputs2[self.layer_idx])
        sim = self._compute_pairwise_similarity(
            outputs1, outputs2, inputs1["attention_mask"], inputs2["attention_mask"]
        )
        # (batch, seq_len, seq_len)
        score = self._aggregate_pairwise_similarity(sim)
        # (batch,)
        return score

    def _compute_batchwise_similarity(
        self,
        inputs1: Dict[str, Tensor],
        inputs2: Dict[str, Tensor],
        inputs_neg: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        outputs1, outputs2, outputs_neg = self._compute_representation(
            inputs1, inputs2, inputs_neg
        )
        outputs1 = self.cl_head(outputs1[self.layer_idx])
        outputs2 = self.cl_head(outputs2[self.layer_idx])
        if outputs_neg is not None:
            outputs_neg = self.cl_head(outputs_neg[self.layer_idx])
        # (batch, seq_len, hidden_dim)
        attention_mask1 = inputs1["attention_mask"]
        attention_mask2 = inputs2["attention_mask"]
        if inputs_neg is not None:
            attention_mask_neg = inputs_neg["attention_mask"]
        if dist.is_initialized():
            outputs1 = dist_all_gather(outputs1)
            outputs2 = dist_all_gather(outputs2)
            attention_mask1 = dist_all_gather(attention_mask1)
            attention_mask2 = dist_all_gather(attention_mask2)
            if outputs_neg is not None:
                outputs_neg = dist_all_gather(outputs_neg)
                attention_mask_neg = dist_all_gather(attention_mask_neg)
        # Compute RWMD
        sim = self._compute_rwmd(
            outputs1,
            outputs2,
            attention_mask1,
            attention_mask2,
            outputs_neg,
            attention_mask_neg,
        )
        return sim

    def _compute_pairwise_similarity(
        self,
        outputs1: Tensor,
        outputs2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tensor:
        assert all(
            x == y or x == 1 or y == 1
            for x, y in zip(outputs1.shape[:-2], outputs2.shape[:-2])
        ), f"Preceding dim doesn't match {outputs1.shape = }, {outputs2.shape = }"
        assert outputs1.shape[-1] == outputs2.shape[-1], "Hidden dim doesn't match"
        sim = F.cosine_similarity(
            outputs1[..., :, None, :], outputs2[..., None, :, :], dim=-1
        )
        inf = torch.tensor(float("-inf"), device=sim.device)
        sim = torch.where(attention_mask1[..., :, None].bool(), sim, inf)
        sim = torch.where(attention_mask2[..., None, :].bool(), sim, inf)
        return sim

    def _compute_indice(self, sim: Tensor, dim: int) -> Tensor:
        return torch.max(sim, dim=dim)[1]

    def _compute_rwmd(
        self,
        outputs1: Tensor,
        outputs2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
        outputs_neg: Optional[Tensor] = None,
        attention_mask_neg: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute relaxed word mover distance

        :param outputs1: (batch1, seq_len1, hidden_dim), torch.float
        :param outputs2: (batch2, seq_len2, hidden_dim), torch.float
        :param attention_mask1: (batch1, seq_len1), torch.bool
        :param attention_mask2: (batch2, seq_len2), torch.bool
        :param outputs_neg: (batch1, seq_len1, hidden_dim), torch.float
        :param attention_mask_neg: (batch1, seq_len1), torch.bool
        :return: (batch1, batch2(+1))
        """
        batch1, seq_len1, hidden_dim = outputs1.shape
        batch2, seq_len2, _ = outputs2.shape
        with torch.no_grad():
            indice1 = torch.empty(
                (batch1, batch2, seq_len1), dtype=torch.long, device=outputs1.device
            )
            indice2 = torch.empty(
                (batch1, batch2, seq_len2), dtype=torch.long, device=outputs2.device
            )
            for i in range(0, outputs1.shape[0], 8):
                for j in range(0, outputs2.shape[0], 8):
                    sim = self._compute_pairwise_similarity(
                        outputs1[i : i + 8, None, :, :],
                        outputs2[None, j : j + 8, :, :],
                        attention_mask1[i : i + 8, None, :],
                        attention_mask2[None, j : j + 8, :],
                    )
                    indice1[i : i + 8, j : j + 8, :] = self._compute_indice(sim, dim=-1)
                    indice2[i : i + 8, j : j + 8, :] = self._compute_indice(sim, dim=-2)
            indice1, indice2 = indice1.unsqueeze(-1), indice2.unsqueeze(-1)
            # (batch1, batch2, seq_len1), (batch1, batch2, seq_len2)
            if outputs_neg is not None:
                sim_neg = self._compute_pairwise_similarity(
                    outputs1, outputs_neg, attention_mask1, attention_mask_neg
                )
                indice_neg1 = self._compute_indice(sim_neg, dim=-1)
                indice_neg2 = self._compute_indice(sim_neg, dim=-2)
                indice_neg1 = indice_neg1.unsqueeze(-1)
                indice_neg2 = indice_neg2.unsqueeze(-1)
            # (batch1, seq_len1)
        if outputs_neg is not None:
            # NOTE: Pair negative
            sim1 = F.cosine_similarity(
                outputs1,
                torch.gather(
                    outputs_neg, dim=1, index=indice_neg1.expand(-1, -1, hidden_dim)
                ),
                dim=-1,
            )
            sim2 = F.cosine_similarity(
                torch.gather(
                    outputs1, dim=1, index=indice_neg2.expand(-1, -1, hidden_dim)
                ),
                outputs_neg,
                dim=-1,
            )
            sim1 = masked_mean(sim1, ~torch.isinf(sim1), dim=-1)
            sim2 = masked_mean(sim2, ~torch.isinf(sim2), dim=-1)
            sim_neg = (sim1 + sim2) / 2
        # NOTE: Pairwise negative
        outputs1, outputs2 = outputs1.unsqueeze(1), outputs2.unsqueeze(0)
        sim1 = F.cosine_similarity(
            outputs1,
            torch.gather(
                outputs2.expand((batch1, -1, -1, -1)),
                dim=2,
                index=indice1.expand((-1, -1, -1, hidden_dim)),
            ),
            dim=-1,
        )
        # (batch1, batch2, seq_len1)
        # (batch1, batch2, seq_len2, hidden_dim)
        sim2 = F.cosine_similarity(
            torch.gather(
                outputs1.expand((-1, batch2, -1, -1)),
                dim=2,
                index=indice2.expand((-1, -1, -1, hidden_dim)),
            ),
            outputs2,
            dim=-1,
        )
        # (batch1, batch2, seq_len2)
        sim1 = masked_mean(sim1, ~torch.isinf(sim1), dim=-1)
        sim2 = masked_mean(sim2, ~torch.isinf(sim2), dim=-1)
        sim = (sim1 + sim2) / 2
        if outputs_neg is not None:
            sim = torch.cat((sim, sim_neg.unsqueeze(-1)), dim=1)
        return sim

    def _aggregate_pairwise_similarity(self, sim: Tensor) -> Tensor:
        sim_left = torch.max(sim, dim=-1)[0]
        sim_right = torch.max(sim, dim=-2)[0]
        sim_left = masked_mean(sim_left, ~torch.isinf(sim_left), dim=-1)
        sim_right = masked_mean(sim_right, ~torch.isinf(sim_right), dim=-1)
        return (sim_left + sim_right) / 2


def create_contrastive_learning(model_args: ModelArguments) -> ContrastiveLearning:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, output_hidden_states=True, **asdict(model_args)
    )
    backbone = AutoModel.from_pretrained(model_args.model_name_or_path, config=config)
    if model_args.loss_rwmd:
        return TokenContrastiveLearning(
            config, backbone, model_args.temp, model_args.loss_mlm, model_args.layer_idx
        )
    else:
        return SimpleContrastiveLearning(
            config,
            backbone,
            model_args.temp,
            model_args.loss_mlm,
            model_args.pooler_type,
        )
