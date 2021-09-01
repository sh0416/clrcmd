import logging
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import (
    BertLMPredictionHead,
    BertModel,
    BertPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
    RobertaPreTrainedModel,
)

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
            hidden_states = outputs.hidden_states
            attention_mask = attention_mask[:, :, None]
            hidden = (hidden_states[0] + hidden_states[-1]) / 2.0
            hidden = hidden * attention_mask
            pooled_sum = hidden.sum(dim=1)
            masked_sum = attention_mask.sum(dim=1)
            return pooled_sum / masked_sum
        elif self.pooler_type == "avg_top2":
            hidden_states = outputs.hidden_states
            attention_mask = attention_mask[:, :, None]
            hidden = (hidden_states[-1] + hidden_states[-2]) / 2.0
            hidden = hidden * attention_mask
            pooled_sum = hidden.sum(dim=1)
            masked_sum = attention_mask.sum(dim=1)
            return pooled_sum / masked_sum
        else:
            raise NotImplementedError()


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.init_weights()


def dist_all_gather(x: Tensor) -> Tensor:
    """Boilerplate code for all gather in distributed setting

    :param x: Tensor to be gathered
    :type x: Tensor
    :return: Tensor after gathered. For the gradient flow, current rank is
             replaced to original tensor
    :rtype: Tensor
    """
    xlist = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=xlist, tensor=x.contiguous())
    # Since `all_gather` results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    xlist[dist.get_rank()] = x
    return torch.cat(xlist, dim=0)


def compute_loss_simclr(
    output1: BaseModelOutputWithPoolingAndCrossAttentions,
    output2: BaseModelOutputWithPoolingAndCrossAttentions,
    attention_mask1: Tensor,
    attention_mask2: Tensor,
    pooler_fn: Callable,
    temp: float,
    is_training: bool,
) -> Tensor:
    """Compute SimCLR loss in sentence-level

    :param output1: Model output for first view
    :type output1: BaseModelOutputWithPoolingAndCrossAttentions
    :param output2: Model output for second view
    :type output2: BaseModelOutputWithPoolingAndCrossAttentions
    :param attention_mask1: Attention mask
    :type attention_mask1: FloatTensor(batch_size, seq_len)
    :param attention_mask2: Attention mask
    :type attention_mask2: FloatTensor(batch_size, seq_len)
    :param pooler_fn: Function for extracting sentence representation
    :type pooler_fn: Callable[[BaseModelOutputWithPoolingAndCrossAttentions],
                              FloatTensor(batch_size, hidden_dim)]
    :param temp: Temperature for cosine similarity
    :type temp: float
    :param is_training: Flag indicating whether is training or not
    :type is_training: bool
    :return: Scalar loss
    :rtype: FloatTensor()
    """
    output1 = pooler_fn(attention_mask1, output1)
    output2 = pooler_fn(attention_mask2, output2)
    # Gather all embeddings if using distributed training
    if dist.is_initialized() and is_training:
        output1, output2 = dist_all_gather(output1), dist_all_gather(output2)

    sim = F.cosine_similarity(output1[None, :, :], output2[:, None, :], dim=2)
    sim = sim / temp
    # (batch_size, batch_size)
    labels = torch.arange(sim.shape[1], dtype=torch.long, device=sim.device)
    loss = F.cross_entropy(sim, labels)
    logger.debug(f"{loss = :.4f}")
    return loss


def compute_loss_simclr_token(
    output1: Tensor, output2: Tensor, pair: List[Tensor], temp: float
) -> Tensor:
    """Compute SimCLR loss in token-level

    :param output1: Bert output for the first sentence
    :type output1: FloatTensor(batch_size, seq_len, hidden_dim)
    :param ouptut2: Bert output for the second sentence
    :type output2: FloatTensor(batch_size, seq_len, hidden_dim)
    :param pair: Pair for computing similarity between token
    :type pair: List[LongTensor(2, num_pairs)]
    :param temp: Temperature for cosine similarity
    :type temp: float
    :return: Scalar loss
    :rtype: FloatTensor()
    """
    assert output1.shape[0] == output2.shape[0] == len(pair)
    batch_idx = [torch.full((x.shape[1],), i) for i, x in enumerate(pair)]
    seq_idx1 = [x[0] for x in pair]
    seq_idx2 = [x[1] for x in pair]
    batch_idx = torch.cat(batch_idx)
    seq_idx1 = torch.cat(seq_idx1)
    seq_idx2 = torch.cat(seq_idx2)
    output1 = output1[batch_idx, seq_idx1]
    output2 = output2[batch_idx, seq_idx2]
    # (num_pairs, hidden_dim)
    sim = F.cosine_similarity(output1[None, :, :], output2[:, None, :], dim=2)
    sim = sim / temp  # Apply temperature
    # (num_pairs, num_pairs)
    label = torch.arange(sim.shape[1], dtype=torch.long, device=sim.device)
    loss = F.cross_entropy(sim, label)
    logger.debug(f"{loss = :.4f}")
    return loss


def compute_representation(
    encoder: Callable,
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
) -> Tuple[
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
]:
    """Compute bert contextual representation for contrastive learning"""
    batch_size = input_ids.shape[0]
    input_ids = torch.cat([input_ids[:, 0], input_ids[:, 1]])
    attention_mask = torch.cat([attention_mask[:, 0], attention_mask[:, 1]])
    if token_type_ids is not None:
        token_type_ids = torch.cat(
            [token_type_ids[:, 0], token_type_ids[:, 1]]
        )
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    last_hidden1 = outputs.last_hidden_state[:batch_size]
    last_hidden2 = outputs.last_hidden_state[batch_size:]
    if outputs.pooler_output is not None:
        pooler_output1 = outputs.pooler_output[:batch_size]
        pooler_output2 = outputs.pooler_output[batch_size:]
    if outputs.hidden_states is not None:
        hidden_states1 = tuple(x[:batch_size] for x in outputs.hidden_states)
        hidden_states2 = tuple(x[batch_size:] for x in outputs.hidden_states)
    else:
        hidden_states1, hidden_states2 = None, None
    outputs1 = BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output1,
        last_hidden_state=last_hidden1,
        hidden_states=hidden_states1,
    )
    outputs2 = BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output2,
        last_hidden_state=last_hidden2,
        hidden_states=hidden_states2,
    )
    return outputs1, outputs2


def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    pairs=None,
) -> Tuple[Tensor]:
    outputs1, outputs2 = compute_representation(
        encoder, input_ids, attention_mask, token_type_ids
    )
    attention_mask1 = attention_mask[:, 0, :]
    attention_mask2 = attention_mask[:, 1, :]
    loss = compute_loss_simclr(
        outputs1,
        outputs2,
        attention_mask1,
        attention_mask2,
        cls.pooler,
        cls.model_args.temp,
        cls.training,
    )
    if cls.model_args.loss_token:
        assert pairs is not None
        loss += cls.model_args.coeff_loss_token * compute_loss_simclr_token(
            outputs1.last_hidden_state,
            outputs2.last_hidden_state,
            pairs,
            cls.model_args.temp,
        )

    return (loss,)


def sentemb_forward(
    cls, encoder, input_ids=None, attention_mask=None, token_type_ids=None
) -> BaseModelOutputWithPoolingAndCrossAttentions:
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    return cls.pooler(attention_mask, outputs)


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config)

        cl_init(self, config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pairs=None,
        sent_emb=False,
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            return cl_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pairs=pairs,
            )
