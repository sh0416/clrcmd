import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
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


class MLPLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(dim_in, dim_out), nn.Tanh())

    def forward(self, features: Tensor) -> Tensor:
        return self.layer(features)


class TemperatureSimilarity(nn.Module):
    def __init__(self, temp: float):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=2)
        self.temp = temp

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute cosine similarity and apply temperature smoothing.
        The tensor could be broadcastable.

        :param x: First tensor
        :type x: Tensor(d1, d2, d3)
        :param y: Second tensor
        :type y: Tensor(d1, d2, d3)
        :return: Cosine similarity
        :rtype: Tensor(d1, d2)
        """
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
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

    def forward(self, attention_mask: Tensor, outputs: Tensor) -> Tensor:
        if self.pooler_type in ["cls_before_pooler", "cls"]:
            last_hidden = outputs.last_hidden_state
            return last_hidden[:, 0]
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
    cls.mlp = MLPLayer(config.hidden_size, config.hidden_size)
    cls.sim = TemperatureSimilarity(temp=cls.model_args.temp)
    cls.init_weights()


def cl_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    batch_size, num_sent, seq_len = input_ids.shape

    mlm_outputs = None
    # Flatten input for encoding (bs * num_sent, len)
    input_ids = input_ids.view(-1, seq_len)
    attention_mask = attention_mask.view(-1, seq_len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, seq_len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view(-1, seq_len)
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view(
        batch_size, num_sent, pooler_output.size(-1)
    )
    # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(world_size)]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[rank] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(world_size)]
        z2_list = [torch.zeros_like(z2) for _ in range(world_size)]
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since `all_gather` results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[rank] = z1
        z2_list[rank] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # (batch_size, 1, hidden_dim) x (1, batch_size, hidden_dim) = (batch_size, batch_size)

    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        # (batch_size, 1, hidden_dim) x (1, num_negative, hidden_dim) = (batch_size, num_negative)
        cos_sim = torch.cat([cos_sim, z1_z3_cos], dim=1)
        # (batch_size, batch_size + num_negative)

    labels = torch.arange(cos_sim.size(0), dtype=torch.long, device=cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [
                [0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1))
                + [0.0] * i
                + [z3_weight]
                + [0.0] * (z1_z3_cos.size(-1) - i - 1)
                for i in range(z1_z3_cos.size(-1))
            ]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size),
            mlm_labels.view(-1),
        )
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    return_dict = (
        return_dict if return_dict is not None else cls.config.use_return_dict
    )
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = (
        return_dict if return_dict is not None else cls.config.use_return_dict
    )

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.model_args = kwargs["model_args"]
        self.bert = BertModel(config)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
