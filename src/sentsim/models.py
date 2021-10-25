import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from simcse.config import ModelArguments
from simcse.utils import masked_mean
from torch import Tensor
from transformers import AutoConfig, AutoModel
from transformers.utils.dummy_pt_objects import PreTrainedModel

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


class LastHiddenSentenceRepresentationModel(nn.Module):
    def __init__(self, model: PreTrainedModel, hidden_size: int):
        super().__init__()
        self.model = model
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

    def forward(self, inputs: ModelInput) -> Tuple[Tensor, Tensor]:
        # Return representation with mask
        mask = inputs["attention_mask"].bool()
        return self.head(self.model(**inputs).last_hidden_state), mask


class CLSSentenceRepresentationModel(nn.Module):
    def __init__(self, model: PreTrainedModel, hidden_size: int):
        super().__init__()
        self.model = model
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

    def forward(self, inputs: ModelInput) -> Tensor:
        return self.head(self.model(**inputs).last_hidden_state[:, 0])


class SentenceSimilarityModel(nn.Module):
    def __init__(self, representation_model: nn.Module, similarity: nn.Module):
        super().__init__()
        self.representation_model = representation_model
        self.similarity = similarity

    def forward(self, inputs1: ModelInput, inputs2: ModelInput) -> Tensor:
        """Provide similarity between two sentences

        :param inputs1: model input for sentence1.
        :param inputs2: model input for sentence2.
        :return: similarity score.
        """
        x1, x2 = self.representation_model(inputs1), self.representation_model(inputs2)
        return self.similarity(x1, x2)

    def compute_heatmap(self, inputs1: ModelInput, inputs2: ModelInput) -> Tensor:
        x1, x2 = self.representation_model(inputs1), self.representation_model(inputs2)
        return self.similarity.compute_heatmap(x1, x2)


def compute_heatmap(x1: Tensor, x2: Tensor) -> Tensor:
    # Compute indice that produces maximum similarity
    return F.cosine_similarity(x1.unsqueeze(-2), x2.unsqueeze(-3), dim=-1)


def compute_alignment(
    x1: Tensor, x2: Tensor, mask1: Tensor, mask2: Tensor
) -> Tuple[Tensor, Tensor]:
    sim = compute_heatmap(x1, x2)
    # Set similarity of invalid position to negative inf
    inf = torch.tensor(float("-inf"), device=sim.device)
    sim = torch.where(mask1.unsqueeze(-1).bool(), sim, inf)
    sim = torch.where(mask2.unsqueeze(-2).bool(), sim, inf)
    indice1 = torch.max(sim, dim=-1)[1]
    indice2 = torch.max(sim, dim=-2)[1]
    return indice1, indice2


class RelaxedWordMoverSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch, seq_len1, hidden_dim), (batch, seq_len1)), torch.float
        :param x2: ((batch, seq_len2, hidden_dim), (batch, seq_len2)), torch.float
        :return: (batch)
        """
        (x1, mask1), (x2, mask2) = x1, x2
        with torch.no_grad():
            indice1, indice2 = compute_alignment(x1, x2, mask1, mask2)
        # Construct computational graph
        sim1 = self.cos(
            x1, torch.gather(x2, dim=1, index=indice1.unsqueeze(-1).expand_as(x1))
        )
        # (batch, seq_len1)
        sim2 = self.cos(
            torch.gather(x1, dim=1, index=indice2.unsqueeze(-1).expand_as(x2)), x2
        )
        # (batch, seq_len2)
        sim1 = masked_mean(sim1, mask1.bool(), dim=-1)
        sim2 = masked_mean(sim2, mask2.bool(), dim=-1)
        sim = (sim1 + sim2) / 2
        return sim

    def compute_heatmap(
        self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]
    ) -> Tensor:
        (x1, _), (x2, _) = x1, x2
        return compute_heatmap(x1, x2)


class PairwiseRelaxedWordMoverSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x1: Tuple[Tensor, Tensor], x2: Tuple[Tensor, Tensor]) -> Tensor:
        """Compute relaxed word mover similarity

        :param x1: ((batch1, seq_len1, hidden_dim), (batch1, seq_len1)), torch.float
        :param x2: ((batch2, seq_len2, hidden_dim), (batch2, seq_len2)), torch.float
        :return: (batch1, batch2)
        """
        (x1, mask1), (x2, mask2) = x1, x2
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
                    _indice1, _indice2 = compute_alignment(
                        x1[i : i + 8, None, :, :],
                        x2[None, j : j + 8, :, :],
                        mask1[i : i + 8, None, :],
                        mask2[None, j : j + 8, :],
                    )
                    indice1[i : i + 8, j : j + 8, :] = _indice1
                    indice2[i : i + 8, j : j + 8, :] = _indice2
        # Construct computational graph for RWMD
        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(0)
        sim1 = self.cos(
            x1,  # (batch1, 1, seq_len1, hidden_dim)
            torch.gather(
                x2.expand((batch1, -1, -1, -1)),
                dim=2,
                index=indice1.unsqueeze(-1).expand((-1, -1, -1, hidden_dim)),
            ),  # (batch1, batch2, seq_len1, hidden_dim)
        )
        # (batch1, batch2, seq_len1)
        sim2 = self.cos(
            torch.gather(
                x1.expand((-1, batch2, -1, -1)),
                dim=2,
                index=indice2.unsqueeze(-1).expand((-1, -1, -1, hidden_dim)),
            ),  # (batch1, batch2, seq_len2, hidden_dim)
            x2,  # (1, batch2, seq_len2, hidden_dim)
        )
        # (batch1, batch2, seq_len2)
        batchwise_mask1 = mask1[:, None, :].expand_as(sim1)
        batchwise_mask2 = mask2[None, :, :].expand_as(sim2)
        sim1 = masked_mean(sim1, batchwise_mask1.bool(), dim=-1)
        sim2 = masked_mean(sim2, batchwise_mask2.bool(), dim=-1)
        sim = (sim1 + sim2) / 2
        return sim


class PairwiseCosineSimilarity(nn.Module):
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)


class InBatchContrastiveLearningModule(nn.Module):
    def __init__(
        self,
        model: SentenceSimilarityModel,
        pairwise_similarity: nn.Module,
        temp: float,
    ):
        super().__init__()
        self.model = model
        self.pairwise_similarity = pairwise_similarity
        self.temp = temp

    def forward(
        self,
        inputs1: ModelInput,
        inputs2: ModelInput,
        inputs_neg: Optional[ModelInput] = None,
        inputs_mlm: Optional[ModelInput] = None,
        labels_mlm: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        if inputs_neg is not None:
            inputs = {
                k: torch.cat((inputs1[k], inputs2[k], inputs_neg[k]), dim=0)
                for k in inputs1.keys()
            }
        else:
            inputs = {
                k: torch.cat((inputs1[k], inputs2[k]), dim=0) for k in inputs1.keys()
            }
        x = self.model.representation_model(inputs)
        if inputs_neg is not None:
            sections = (
                inputs1["input_ids"].shape[0],
                inputs2["input_ids"].shape[0],
                inputs_neg["input_ids"].shape[0],
            )
            x1, x2, x_neg = list(
                zip(torch.split(x[0], sections), torch.split(x[1], sections))
            )
        else:
            sections = inputs1["input_ids"].shape[0], inputs2["input_ids"].shape[0]
            x1, x2 = list(zip(torch.split(x[0], sections), torch.split(x[1], sections)))
        # DDP: Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            x1 = dist_all_gather(x1[0]), dist_all_gather(x1[1])
            x2 = dist_all_gather(x2[0]), dist_all_gather(x2[1])
            if x_neg is not None:
                x_neg = dist_all_gather(x_neg[0]), dist_all_gather(x_neg[1])
        sim = self.pairwise_similarity(x1, x2)
        if inputs_neg is not None:
            sim_neg = self.model.similarity(x1, x_neg)
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
    pretrained_model = AutoModel.from_pretrained(
        model_args.model_name_or_path, config=config
    )
    if model_args.loss_rwmd:
        representation_model = LastHiddenSentenceRepresentationModel(
            pretrained_model, config.hidden_size
        )
        similarity = RelaxedWordMoverSimilarity()
        pairwise_similarity = PairwiseRelaxedWordMoverSimilarity()
    else:
        representation_model = CLSSentenceRepresentationModel(
            pretrained_model, config.hidden_size
        )
        similarity = nn.CosineSimilarity(dim=-1)
        pairwise_similarity = PairwiseCosineSimilarity()
    model = SentenceSimilarityModel(representation_model, similarity)
    return InBatchContrastiveLearningModule(model, pairwise_similarity, model_args.temp)
