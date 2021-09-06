from dataclasses import dataclass
import logging
from re import I
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
    RobertaPreTrainedModel,
)

logger = logging.getLogger(__name__)


@dataclass
class BaseModelOutputWithHead(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor] = None
    token_output: torch.FloatTensor = None


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
        outputs: BaseModelOutputWithHead,
    ) -> Tensor:
        if self.pooler_type == "cls":
            if self.training:
                return outputs.token_output[:, 0]
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
    cls.mlp = nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.Tanh(),
    )
    cls.init_weights()


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
    # 1. Get size acroess processes
    x_numel_list = [
        torch.tensor(x.numel(), device=x.device)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(x_numel_list, torch.tensor(x.numel(), device=x.device))
    # 2. Infer maximum size
    max_size = max(x.item() for x in x_numel_list)
    # 3. Communitcate tensor with padded version
    _x_list = [
        torch.empty((max_size,), device=x.device)
        for _ in range(dist.get_world_size())
    ]
    _x = torch.cat(
        (
            x.contiguous().view(-1),
            torch.empty((max_size - x.numel(),), device=x.device),
        )
    )
    dist.all_gather(_x_list, _x)
    # 4. Remove padded data to change original shape
    x_list = [
        _x[:n].view(-1, *x.shape[1:]) for n, _x in zip(x_numel_list, _x_list)
    ]
    # Since `all_gather` results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    x_list[dist.get_rank()] = x
    return torch.cat(x_list, dim=0)


def compute_loss_simclr(
    outputs: BaseModelOutputWithHead,
    token_outputs: Tensor,
    attention_mask: Tensor,
    pooler_fn: Callable,
    temp: float,
    is_training: bool,
) -> Tensor:
    """Compute SimCLR loss in sentence-level

    :param outputs: Model output for first view
    :type outputs: BaseModelOutputWithPooling
    :param attention_mask: Attention mask
    :type attention_mask: FloatTensor(batch_size, 2, seq_len)
    :param pooler_fn: Function for extracting sentence representation
    :type pooler_fn: Callable[[Tensor, BaseModelOutputWithHead],
                              FloatTensor(batch_size, hidden_dim)]
    :param temp: Temperature for cosine similarity
    :type temp: float
    :param is_training: Flag indicating whether is training or not
    :type is_training: bool
    :return: Scalar loss
    :rtype: FloatTensor()
    """
    attention_mask1 = attention_mask[:, 0, :]
    attention_mask2 = attention_mask[:, 1, :]
    if outputs.hidden_states is not None:
        hidden_states1 = outputs.hidden_states[:, 0, :]
        hidden_states2 = outputs.hidden_states[:, 1, :]
    else:
        hidden_states1, hidden_states2 = None, None
    output1 = BaseModelOutputWithHead(
        last_hidden_state=outputs.last_hidden_state[:, 0, :],
        hidden_states=hidden_states1,
        token_output=token_outputs[:, 0, :, :],
    )
    output2 = BaseModelOutputWithHead(
        last_hidden_state=outputs.last_hidden_state[:, 1, :],
        hidden_states=hidden_states2,
        token_output=token_outputs[:, 1, :, :],
    )
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
    inputs: Tensor,
    outputs: Tensor,
    pairs: List[Tensor],
    temp: float,
    is_training: bool,
) -> Tensor:
    """Compute SimCLR loss in token-level

    :param inputs: Bert input for the first sentence
    :type inputs: LongTensor(batch_size, 2, seq_len)
    :param outputs: Bert output for the first sentence
    :type outputs: FloatTensor(batch_size, 2, seq_len, hidden_dim)
    :param pairs: Pair for computing similarity between token
    :type pairs: List[LongTensor(num_pairs, 2, 2)]
    :param temp: Temperature for cosine similarity
    :type temp: float
    :param is_training: indicator whether training or not
    :type is_training: bool
    :return: Scalar loss
    :rtype: FloatTensor()
    """
    # NOTE: Keep in mind that this code should be compatible in DDP setting
    # Assertion: Batch size is equal for all tensor
    assert inputs.shape[0:3] == outputs.shape[0:3]
    # 1. Index input and output tensor
    input1 = inputs[pairs[:, 0], 0, pairs[:, 1]]
    input2 = inputs[pairs[:, 0], 1, pairs[:, 2]]
    assert torch.equal(input1, input2), "Different input pair is not supported"
    output1 = outputs[pairs[:, 0], 0, pairs[:, 1]]
    output2 = outputs[pairs[:, 0], 1, pairs[:, 2]]
    # (num_pairs,), (num_pairs, hidden_dim)
    # 2. (Optional) Gather all embeddings if using distributed training
    if dist.is_initialized() and is_training:
        input1 = dist_all_gather(input1)
        output1, output2 = dist_all_gather(output1), dist_all_gather(output2)
    # (valid_num_pairs,), (valid_num_pairs, hidden_dim)
    # 3. Sort and split output tensor based on input
    sorted_val, sorted_indice = torch.sort(input1)
    output1, output2 = output1[sorted_indice], output2[sorted_indice]
    # (valid_num_pairs,), (valid_num_pairs, hidden_dim)
    val, counts = torch.unique(sorted_val, sorted=True, return_counts=True)
    counts = counts.tolist()
    output1 = torch.split(output1, counts)
    output2 = torch.split(output2, counts)
    output1 = [x for x in output1 if x.shape[0] > 20]
    output2 = [x for x in output2 if x.shape[0] > 20]
    # list(torch.FloatTensor(valid_num_pairs_per_token, hidden_dim))
    # 4. Calculate temperature aware cosine similarity
    sim = [
        F.cosine_similarity(x1[None, :, :], x2[:, None, :], dim=2) / temp
        for x1, x2 in zip(output1, output2)
    ]
    # list(FloatTensor(num_pairs_per_symbol, num_pairs_per_symbol))
    label = [
        torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        for x in sim
    ]
    loss = torch.stack([F.cross_entropy(x, l) for x, l in zip(sim, label)])
    loss = loss.mean()
    return loss


def compute_representation(
    encoder: Callable,
    input_ids: Tensor,
    attention_mask: Tensor,
    token_type_ids: Tensor,
) -> Tuple[BaseModelOutputWithHead, BaseModelOutputWithHead]:
    """Compute bert contextual representation for contrastive learning"""
    input_ids = input_ids.view(-1, input_ids.shape[2])
    attention_mask = attention_mask.view(-1, attention_mask.shape[2])
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[2])
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    dims = outputs.last_hidden_state.shape[1:]
    last_hidden = outputs.last_hidden_state.view(-1, 2, *dims)
    if outputs.hidden_states is not None:
        dims = outputs.hidden_states.shape[1:]
        hidden_states = outputs.hidden_states.view(-1, 2, *dims)
    else:
        hidden_states = None
    outputs = BaseModelOutput(
        last_hidden_state=last_hidden,
        hidden_states=hidden_states,
    )
    return outputs


def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    pairs=None,
    input_ids_mlm=None,
    labels_mlm=None,
) -> Tuple[Tensor]:
    outputs = compute_representation(
        encoder, input_ids, attention_mask, token_type_ids
    )
    token_outputs = cls.mlp(outputs.last_hidden_state)
    loss = compute_loss_simclr(
        outputs,
        token_outputs,
        attention_mask,
        cls.pooler,
        cls.model_args.temp,
        cls.training,
    )
    if cls.model_args.loss_mlm:
        outputs_mlm = compute_representation(
            encoder, input_ids_mlm, attention_mask, token_type_ids
        )
        last_hidden = outputs_mlm.last_hidden_state
        last_hidden = last_hidden.view(-1, *last_hidden.shape[2:])
        logits_mlm = cls.lm_head(last_hidden)
        logits_mlm = logits_mlm.view(-1, cls.config.vocab_size)
        labels_mlm = labels_mlm.view(-1)
        loss_mlm = F.cross_entropy(logits_mlm, labels_mlm)
        loss += cls.model_args.coeff_loss_mlm * loss_mlm
    if cls.model_args.loss_token:
        assert pairs is not None
        loss += cls.model_args.coeff_loss_token * compute_loss_simclr_token(
            input_ids, token_outputs, pairs, cls.model_args.temp, cls.training
        )
    return (loss,)


def sentemb_forward(
    cls, encoder, input_ids=None, attention_mask=None, token_type_ids=None
) -> BaseModelOutputWithHead:
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    token_output = cls.mlp(outputs.last_hidden_state)
    outputs = BaseModelOutputWithHead(
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
        token_output=token_output,
    )
    return outputs


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config)
        if self.model_args.loss_mlm:
            self.lm_head = RobertaLMHead(config)
        cl_init(self, config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pairs=None,
        input_ids_mlm=None,
        labels_mlm=None,
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
                input_ids_mlm=input_ids_mlm,
                labels_mlm=labels_mlm,
            )
