import logging
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel

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
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], f"unrecognized pooling type {self.pooler_type}"

    def forward(
        self,
        attention_mask: Tensor,
        outputs: BaseModelOutputWithPoolingAndCrossAttentions,
    ) -> Tensor:
        if self.pooler_type == "cls":
            if self.training:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0]
        elif self.pooler_type == "cls_before_pooler":
            return outputs.last_hidden_state[:, 0]
        elif self.pooler_type == "avg":
            last_hidden = outputs.last_hidden_state
            attention_mask = attention_mask[:, :, None]
            hidden = last_hidden * attention_mask
            pooled_sum = hidden.sum(dim=1)
            masked_sum = attention_mask.sum(dim=1)
            return pooled_sum / masked_sum
        elif self.pooler_type == "avg_first_last":
            assert outputs.hidden_states is not None
            hidden_states = outputs.hidden_states
            attention_mask = attention_mask[:, :, None]
            hidden = (hidden_states[0] + hidden_states[-1]) / 2.0
            hidden = hidden * attention_mask
            pooled_sum = hidden.sum(dim=1)
            masked_sum = attention_mask.sum(dim=1)
            return pooled_sum / masked_sum
        elif self.pooler_type == "avg_top2":
            assert outputs.hidden_states is not None
            hidden_states = outputs.hidden_states
            attention_mask = attention_mask[:, :, None]
            hidden = (hidden_states[-1] + hidden_states[-2]) / 2.0
            hidden = hidden * attention_mask
            pooled_sum = hidden.sum(dim=1)
            masked_sum = attention_mask.sum(dim=1)
            return pooled_sum / masked_sum
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


class RobertaForContrastiveLearning(RobertaModel):
    def __init__(self, config, pooler_type: str, loss_mlm: bool, temp: float):
        super().__init__(config, add_pooling_layer=False)
        self.temp = temp
        self.cl_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()
        )
        self._pooler = Pooler(pooler_type)
        if loss_mlm:
            self.lm_head = RobertaLMHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tuple[Tensor]:
        outputs1, outputs2 = self._compute_representation(
            input_ids1, input_ids2, attention_mask1, attention_mask2
        )
        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            outputs1, outputs2 = dist_all_gather(outputs1), dist_all_gather(outputs2)
        sim = F.cosine_similarity(outputs1[:, None, :], outputs2[None, :, :], dim=2)
        sim = sim / self.temp
        # (batch_size, batch_size)
        labels = torch.arange(sim.shape[1], dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return (loss,)

    def compute_similarity(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tensor:
        outputs1, outputs2 = self._compute_representation(
            input_ids1, input_ids2, attention_mask1, attention_mask2
        )
        score = F.cosine_similarity(outputs1, outputs2)
        return score

    def _compute_representation(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        input_ids = torch.cat((input_ids1, input_ids2))
        attention_mask = torch.cat((attention_mask1, attention_mask2))
        outputs = super().forward(input_ids, attention_mask)
        pooler_output1, pooler_output2 = torch.chunk(
            self.cl_head(outputs.last_hidden_state[:, 0]), 2
        )
        last_hidden_state1, last_hidden_state2 = torch.chunk(
            outputs.last_hidden_state, 2
        )
        hidden_states1 = tuple(torch.chunk(x, 2)[0] for x in outputs.hidden_states)
        hidden_states2 = tuple(torch.chunk(x, 2)[1] for x in outputs.hidden_states)
        outputs1 = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=last_hidden_state1,
            hidden_states=hidden_states1,
            pooler_output=pooler_output1,
        )
        outputs2 = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=last_hidden_state2,
            hidden_states=hidden_states2,
            pooler_output=pooler_output2,
        )
        outputs1 = self._pooler(attention_mask1, outputs1)
        outputs2 = self._pooler(attention_mask2, outputs2)
        return outputs1, outputs2


class RobertaForTokenContrastiveLearning(RobertaModel):
    def __init__(self, config, loss_mlm: bool, temp: float):
        super().__init__(config, add_pooling_layer=False)
        self.temp = temp
        self.cl_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()
        )
        if loss_mlm:
            self.lm_head = RobertaLMHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tuple[Tensor]:
        outputs1, outputs2 = self._compute_representation(
            input_ids1, input_ids2, attention_mask1, attention_mask2
        )
        # (batch, seq_len, hidden_dim)
        if dist.is_initialized():
            outputs1 = dist_all_gather(outputs1)
            outputs2 = dist_all_gather(outputs2)
        sim = self._compute_pairwise_similarity(
            outputs1[:, None, :, :],
            outputs2[None, :, :, :],
            attention_mask1[:, None, :],
            attention_mask2[None, :, :],
        )
        # (batch, batch, seq_len, seq_len)
        sim = self._aggregate_pairwise_similarity(sim) / self.temp
        # (batch, batch)
        label = torch.arange(sim.shape[1], dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, label)
        return (loss,)

    def compute_similarity(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tuple[Tensor]:
        outputs1, outputs2 = self._compute_representation(
            input_ids1, input_ids2, attention_mask1, attention_mask2
        )
        sim = self._compute_pairwise_similarity(
            outputs1, outputs2, attention_mask1, attention_mask2
        )
        # (batch, seq_len, seq_len)
        score = self._aggregate_pairwise_similarity(sim)
        # (batch,)
        return score

    def _compute_representation(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        input_ids = torch.cat((input_ids1, input_ids2))
        attention_mask = torch.cat((attention_mask1, attention_mask2))
        outputs = super().forward(input_ids, attention_mask)
        outputs1, outputs2 = torch.chunk(self.cl_head(outputs.last_hidden_state), 2)
        return outputs1, outputs2

    def _compute_pairwise_similarity(
        self,
        outputs1: Tensor,
        outputs2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tensor:
        sim = F.cosine_similarity(
            outputs1[..., :, None, :], outputs2[..., None, :, :], dim=-1
        )
        inf = torch.tensor(float("-inf"), device=sim.device)
        sim = torch.where(attention_mask1[..., :, None].bool(), sim, inf)
        sim = torch.where(attention_mask2[..., None, :].bool(), sim, inf)
        return sim

    def _aggregate_pairwise_similarity(self, sim: Tensor) -> Tensor:
        sim_left = torch.max(sim, dim=-1)[0]
        sim_right = torch.max(sim, dim=-2)[0]
        sim_left = masked_mean(sim_left, ~torch.isinf(sim_left), dim=-1)
        sim_right = masked_mean(sim_right, ~torch.isinf(sim_right), dim=-1)
        return (sim_left + sim_right) / 2
