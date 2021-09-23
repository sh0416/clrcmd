import logging
from typing import Callable, Tuple, Optional

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
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)
        self.cl_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()
        )
        self.init_weights()

    def _compute_representation(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
        input_ids_neg: Optional[Tensor] = None,
        attention_mask_neg: Optional[Tensor] = None,
    ) -> Tuple[
        BaseModelOutputWithPoolingAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        Optional[BaseModelOutputWithPoolingAndCrossAttentions],
    ]:
        if input_ids_neg is not None:
            input_ids = (input_ids1, input_ids2, input_ids_neg)
        else:
            input_ids = (input_ids1, input_ids2)
        input_ids = torch.cat(input_ids)
        if attention_mask_neg is not None:
            attention_mask = (attention_mask1, attention_mask2, attention_mask_neg)
        else:
            attention_mask = (attention_mask1, attention_mask2)
        attention_mask = torch.cat(attention_mask)
        outputs = super().forward(input_ids, attention_mask)
        pooler_outputs = self.cl_head(outputs.last_hidden_state[:, 0])
        if input_ids_neg is not None:
            pooler_output1, pooler_output2, pooler_output_neg = torch.chunk(
                pooler_outputs, 3
            )
            last_hidden_state1, last_hidden_state2, last_hidden_state_neg = torch.chunk(
                outputs.last_hidden_state, 3
            )
            hidden_states1, hidden_states2, hidden_states_neg = [], [], []
            for x in outputs.hidden_states:
                hidden_state1, hidden_state2, hidden_state_neg = torch.chunk(x, 3)
                hidden_states1.append(hidden_state1)
                hidden_states2.append(hidden_state2)
                hidden_states_neg.append(hidden_state_neg)
        else:
            pooler_output1, pooler_output2 = torch.chunk(pooler_outputs, 2)
            last_hidden_state1, last_hidden_state2 = torch.chunk(
                outputs.last_hidden_state, 2
            )
            hidden_states1, hidden_states2 = [], []
            for x in outputs.hidden_states:
                hidden_state1, hidden_state2 = torch.chunk(x, 2)
                hidden_states1.append(hidden_state1)
                hidden_states2.append(hidden_state2)
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
        if input_ids_neg is not None:
            outputs_neg = BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=last_hidden_state_neg,
                hidden_states=hidden_states_neg,
                pooler_output=pooler_output_neg,
            )
        else:
            outputs_neg = None
        return outputs1, outputs2, outputs_neg


class RobertaForSimpleContrastiveLearning(RobertaForContrastiveLearning):
    def __init__(self, config, pooler_type: str, loss_mlm: bool, temp: float):
        super().__init__(config)
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
        input_ids_neg: Optional[Tensor] = None,
        attention_mask_neg: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        outputs1, outputs2, outputs_neg = self._compute_representation(
            input_ids1,
            input_ids2,
            attention_mask1,
            attention_mask2,
            input_ids_neg,
            attention_mask_neg,
        )
        outputs1 = self._pooler(attention_mask1, outputs1)
        outputs2 = self._pooler(attention_mask2, outputs2)
        if outputs_neg is not None:
            outputs_neg = self._pooler(attention_mask_neg, outputs_neg)
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
        sim = sim / self.temp
        # (batch_size, batch_size)
        labels = torch.arange(sim.shape[0], dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return (loss,)

    def compute_similarity(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tensor:
        outputs1, outputs2, _ = self._compute_representation(
            input_ids1, input_ids2, attention_mask1, attention_mask2
        )
        outputs1 = self._pooler(attention_mask1, outputs1)
        outputs2 = self._pooler(attention_mask2, outputs2)
        score = F.cosine_similarity(outputs1, outputs2)
        return score


class RobertaForTokenContrastiveLearning(RobertaForContrastiveLearning):
    def __init__(self, config, loss_mlm: bool, temp: float):
        super().__init__(config)
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
        input_ids_neg: Optional[Tensor] = None,
        attention_mask_neg: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        outputs1, outputs2, outputs_neg = self._compute_representation(
            input_ids1,
            input_ids2,
            attention_mask1,
            attention_mask2,
            input_ids_neg,
            attention_mask_neg,
        )
        outputs1 = self.cl_head(outputs1.last_hidden_state)
        outputs2 = self.cl_head(outputs2.last_hidden_state)
        # (batch, seq_len, hidden_dim)
        if dist.is_initialized():
            outputs1 = dist_all_gather(outputs1)
            outputs2 = dist_all_gather(outputs2)
            attention_mask1 = dist_all_gather(attention_mask1)
            attention_mask2 = dist_all_gather(attention_mask2)
        # Compute RWMD
        sim = self._compute_rwmd(outputs1, outputs2, attention_mask1, attention_mask2)
        # (batch, batch)
        label = torch.arange(sim.shape[0], dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, label)
        return (loss,)

    def compute_similarity(
        self,
        input_ids1: Tensor,
        input_ids2: Tensor,
        attention_mask1: Tensor,
        attention_mask2: Tensor,
    ) -> Tensor:
        outputs1, outputs2, _ = self._compute_representation(
            input_ids1, input_ids2, attention_mask1, attention_mask2
        )
        outputs1 = self.cl_head(outputs1.last_hidden_state)
        outputs2 = self.cl_head(outputs2.last_hidden_state)
        sim = self._compute_pairwise_similarity(
            outputs1, outputs2, attention_mask1, attention_mask2
        )
        # (batch, seq_len, seq_len)
        score = self._aggregate_pairwise_similarity(sim)
        # (batch,)
        return score

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
        with torch.no_grad():
            batch1, batch2 = outputs1.shape[0], outputs2.shape[0]
            seq_len1, seq_len2 = outputs1.shape[1], outputs2.shape[1]
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
        _outputs2 = outputs2[None, :, :, :].expand((outputs1.shape[0], -1, -1, -1))
        # (batch1, batch2, seq_len2, hidden_dim)
        _indice1 = indice1[:, :, :, None].expand((-1, -1, -1, outputs2.shape[2]))
        # (batch1, batch2, seq_len1, hidden_dim)
        _outputs2 = torch.gather(_outputs2, dim=2, index=_indice1)
        # (batch1, batch2, seq_len1, hidden_dim)
        sim1 = F.cosine_similarity(outputs1[:, None, :, :], _outputs2, dim=-1)
        sim1 = sim1 / self.temp
        # (batch1, batch2, seq_len1)
        _outputs1 = outputs1[:, None, :, :].expand((-1, outputs2.shape[0], -1, -1))
        # (batch1, batch2, seq_len1, hidden_dim)
        _indice2 = indice2[:, :, :, None].expand((-1, -1, -1, outputs1.shape[2]))
        # (batch1, batch2, seq_len2, hidden_dim)
        _outputs1 = torch.gather(_outputs1, dim=2, index=_indice2)
        # (batch1, batch2, seq_len2, hidden_dim)
        sim2 = F.cosine_similarity(_outputs1, outputs2[None, :, :, :], dim=-1)
        sim2 = sim2 / self.temp
        # (batch1, batch2, seq_len2)
        sim1 = masked_mean(sim1, ~torch.isinf(sim1), dim=-1)
        sim2 = masked_mean(sim2, ~torch.isinf(sim2), dim=-1)
        return (sim1 + sim2) / 2

    def _aggregate_pairwise_similarity(self, sim: Tensor) -> Tensor:
        sim_left = torch.max(sim, dim=-1)[0]
        sim_right = torch.max(sim, dim=-2)[0]
        sim_left = masked_mean(sim_left, ~torch.isinf(sim_left), dim=-1)
        sim_right = masked_mean(sim_right, ~torch.isinf(sim_right), dim=-1)
        return (sim_left + sim_right) / 2
