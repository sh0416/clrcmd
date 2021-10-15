import logging
from dataclasses import asdict
from typing import Dict, Optional, Tuple, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from simcse.config import ModelArguments
from simcse.utils import masked_mean

logger = logging.getLogger(__name__)


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


ModelInput = Dict[str, Tensor]


def concatenate_model_inputs(
    inputs: Tuple[ModelInput, ...]
) -> Tuple[ModelInput, List[int]]:
    sections = [i["input_ids"].shape[0] for i in inputs]
    inputs = {k: torch.cat([i[k] for i in inputs]) for k in inputs[0].keys()}
    return inputs, sections


class SentenceRepresentationModelPretrainedLastHidden(nn.Module):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

    def forward(
        self, inputs: Tuple[ModelInput, ...]
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Provide sentence representation from CLS

        :param inputs: {"input_ids": tensor(batch_size, seq_len), "attention_mask": tensor(batch_size, seq_len)}
        :return: tensor(batch_size, hidden_dim)
        """
        inputs_concat, sections = concatenate_model_inputs(inputs)
        outputs_concat = self.model(**inputs_concat).last_hidden_state
        return torch.split(outputs_concat, sections)


class SentenceRepresentationModelPretrainedCLS(nn.Module):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)

    def forward(self, inputs: Tuple[ModelInput, ...]) -> Tuple[Tensor, ...]:
        """Provide sentence representation from CLS

        :param inputs: {"input_ids": tensor(batch_size, seq_len), "attention_mask": tensor(batch_size, seq_len)}
        :return: tensor(batch_size, hidden_dim)
        """
        inputs_concat, sections = concatenate_model_inputs(inputs)
        outputs_concat = self.model(**inputs_concat).last_hidden_dim[:, 0]
        return torch.split(outputs_concat, sections)


class SentenceSimilarityModelPretrained(nn.Module):
    def __init__(
        self,
        representation_model: nn.Module,
        similarity: nn.Module,
        pairwise_similarity: nn.Module,
        hidden_size: int,
    ):
        super().__init__()
        self.representation_model = representation_model
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.similarity = similarity
        self.pairwise_similarity = pairwise_similarity  # For training

    def forward(self, inputs1: ModelInput, inputs2: ModelInput) -> Tensor:
        """Provide similarity between two sentences

        :param inputs1: model input for sentence1.
        :param inputs2: model input for sentence2.
        :return: similarity score.
        """
        assert set(inputs1.keys()) == set(inputs2.keys())
        outputs1, outputs2 = self.forward_pretrained_model((inputs1, inputs2))
        # For rwmd similarity, add mask information
        outputs1 = (outputs1, inputs1["attention_mask"])
        outputs2 = (outputs2, inputs2["attention_mask"])
        return self.similarity(outputs1, outputs2)

    def forward_pretrained_model(
        self, inputs: Tuple[ModelInput, ...]
    ) -> Tuple[Tensor, ...]:
        """Provide sentence representation of two given sentences

        :param inputs: model input for sentence. No broadcastable
        :return: **internal representation** for two sentences
        """
        return tuple(self.head(x) for x in self.representation_model(inputs))


def compute_masked_cosine_similarity(
    x1: Tensor, x2: Tensor, mask1: Tensor, mask2: Tensor
) -> Tensor:
    sim = F.cosine_similarity(x1[..., :, None, :], x2[..., None, :, :], dim=-1)
    inf = torch.tensor(float("-inf"), device=sim.device)
    sim = torch.where(mask1[..., :, None].bool(), sim, inf)
    sim = torch.where(mask2[..., None, :].bool(), sim, inf)
    return sim


class RelaxedWordMoverSimilarity(nn.Module):
    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch, seq_len1, hidden_dim), (batch, seq_len1)), torch.float
        :param x2: ((batch, seq_len2, hidden_dim), (batch, seq_len2)), torch.float
        :return: (batch)
        """
        x1, mask1 = x1
        x2, mask2 = x2
        hidden_dim = x1.shape[2]
        # Compute max indice batchwise
        with torch.no_grad():
            sim = compute_masked_cosine_similarity(x1, x2, mask1, mask2)
            # (batch, seq_len1, seq_len2)
            indice1 = torch.max(sim, dim=-1)[1]  # (batch, seq_len1)
            indice2 = torch.max(sim, dim=-2)[1]  # (batch, seq_len2)
            indice1, indice2 = indice1.unsqueeze(-1), indice2.unsqueeze(-1)
        # Construct computational graph
        sim1 = F.cosine_similarity(
            x1,  # (batch, seq_len1, hidden_dim)
            torch.gather(
                x2,
                dim=1,
                index=indice1.expand((-1, -1, hidden_dim)),
            ),  # (batch, seq_len1, hidden_dim)
            dim=-1,
        )
        # (batch, seq_len1)
        sim2 = F.cosine_similarity(
            torch.gather(
                x1,
                dim=1,
                index=indice2.expand((-1, -1, hidden_dim)),
            ),  # (batch, seq_len2, hidden_dim)
            x2,  # (batch, seq_len2, hidden_dim)
            dim=-1,
        )
        # (batch, seq_len2)
        sim1 = masked_mean(sim1, mask1.bool(), dim=-1)
        sim2 = masked_mean(sim2, mask2.bool(), dim=-1)
        sim = (sim1 + sim2) / 2
        return sim


class PairwiseRelaxedWordMoverSimilarity(nn.Module):
    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch1, seq_len1, hidden_dim), (batch1, seq_len1)), torch.float
        :param x2: ((batch2, seq_len2, hidden_dim), (batch2, seq_len2)), torch.float
        :return: (batch1, batch2)
        """
        x1, mask1 = x1
        x2, mask2 = x2
        batch1, seq_len1, hidden_dim = x1.shape
        batch2, seq_len2, _ = x2.shape
        # Compute max indice batchwise
        with torch.no_grad():
            indice1 = torch.empty(
                (batch1, batch2, seq_len1), dtype=torch.long, device=x1.device
            )
            indice2 = torch.empty(
                (batch1, batch2, seq_len2), dtype=torch.long, device=x2.device
            )
            for i in range(0, batch1, 8):
                for j in range(0, batch2, 8):
                    sim = compute_masked_cosine_similarity(
                        x1[i : i + 8, None, :, :],
                        x2[None, j : j + 8, :, :],
                        mask1[i : i + 8, None, :],
                        mask2[None, j : j + 8, :],
                    )
                    indice1[i : i + 8, j : j + 8, :] = torch.max(sim, dim=-1)[1]
                    indice2[i : i + 8, j : j + 8, :] = torch.max(sim, dim=-2)[1]
            indice1, indice2 = indice1.unsqueeze(-1), indice2.unsqueeze(-1)
        # Construct computational graph for RWMD
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(0)
        sim1 = F.cosine_similarity(
            x1,  # (batch1, 1, seq_len1, hidden_dim)
            torch.gather(
                x2.expand((batch1, -1, -1, -1)),
                dim=2,
                index=indice1.expand((-1, -1, -1, hidden_dim)),
            ),  # (batch1, batch2, seq_len1, hidden_dim)
            dim=-1,
        )
        # (batch1, batch2, seq_len1)
        sim2 = F.cosine_similarity(
            torch.gather(
                x1.expand((-1, batch2, -1, -1)),
                dim=2,
                index=indice2.expand((-1, -1, -1, hidden_dim)),
            ),  # (batch1, batch2, seq_len2, hidden_dim)
            x2,  # (1, batch2, seq_len2, hidden_dim)
            dim=-1,
        )
        # (batch1, batch2, seq_len2)
        batchwise_mask1 = mask1[:, None, :].expand_as(sim1)
        batchwise_mask2 = mask2[None, :, :].expand_as(sim2)
        sim1 = masked_mean(sim1, batchwise_mask1.bool(), dim=-1)
        sim2 = masked_mean(sim2, batchwise_mask2.bool(), dim=-1)
        sim = (sim1 + sim2) / 2
        return sim


class InBatchContrastiveLearningModule(nn.Module):
    def __init__(
        self, similarity_model: SentenceSimilarityModelPretrained, temp: float
    ):
        super().__init__()
        self.similarity_model = similarity_model
        self.temp = temp

    def forward(
        self,
        inputs1: ModelInput,
        inputs2: ModelInput,
        inputs_neg: Optional[ModelInput] = None,
    ) -> Tuple[Tensor]:
        if inputs_neg is not None:
            x1, x2, x_neg = self.similarity_model.forward_pretrained_model(
                (inputs1, inputs2, inputs_neg)
            )
        else:
            x1, x2 = self.similarity_model.forward_pretrained_model((inputs1, inputs2))
        # DDP: Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            x1, x2 = dist_all_gather(x1), dist_all_gather(x2)
            mask1 = dist_all_gather(inputs1["attention_mask"])
            mask2 = dist_all_gather(inputs2["attention_mask"])
            if x_neg is not None:
                x_neg = dist_all_gather(x_neg)
                mask_neg = dist_all_gather(inputs_neg["attention_mask"])
        sim = self.similarity_model.pairwise_similarity((x1, mask1), (x2, mask2))
        if inputs_neg is not None:
            sim_neg = self.similarity_model.similarity((x1, mask1), (x_neg, mask_neg))
            sim = torch.cat((sim, sim_neg.unsqueeze(1)), dim=1)
        sim = sim / self.temp
        # (batch_size, batch_size)
        labels = torch.arange(sim.shape[0], dtype=torch.long, device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return (loss,)


def create_contrastive_learning(
    model_args: ModelArguments,
) -> InBatchContrastiveLearningModule:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, output_hidden_states=True, **asdict(model_args)
    )
    if model_args.loss_rwmd:
        representation_model = SentenceRepresentationModelPretrainedLastHidden(
            model_args.model_name_or_path, config=config
        )
        similarity = RelaxedWordMoverSimilarity()
        pairwise_similarity = PairwiseRelaxedWordMoverSimilarity()
    else:
        representation_model = SentenceRepresentationModelPretrainedCLS(
            model_args.model_name_or_path, config=config
        )
        similarity = nn.CosineSimilarity(dim=-1)
        pairwise_similarity = nn.CosineSimilarity(dim=-1)
    model = SentenceSimilarityModelPretrained(
        representation_model, similarity, pairwise_similarity, config.hidden_size
    )
    return InBatchContrastiveLearningModule(model, model_args.temp)
