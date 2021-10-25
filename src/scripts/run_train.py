import json
import logging
import os
from dataclasses import asdict
from functools import partial

import torch
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from sentsim.config import DataTrainingArguments, ModelArguments, OurTrainingArguments
from sentsim.data.dataset import (
    NLIDataset,
    PairedContrastiveLearningDataset,
    WikiEDADataset,
    WikiIdentityDataset,
    WikiRepetitionDataset,
    collate_fn,
)
from sentsim.models.models import create_contrastive_learning
from sentsim.trainer import CLTrainer

logger = logging.getLogger(__name__)


def train(args):
    model_args, data_args, training_args = args

    # Save arguments
    os.makedirs(training_args.output_dir, exist_ok=True)
    filepath = os.path.join(training_args.output_dir, "model_args.json")
    with open(filepath, "w") as f:
        json.dump(asdict(model_args), f)
    filepath = os.path.join(training_args.output_dir, "data_args.json")
    with open(filepath, "w") as f:
        json.dump(asdict(data_args), f)
    filepath = os.path.join(training_args.output_dir, "training_args.json")
    with open(filepath, "w") as f:
        json.dump(training_args.to_dict(), f)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16} "
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=False
        )
        model = create_contrastive_learning(model_args)
        model.train()
    else:
        raise NotImplementedError()

    if data_args.data_type == "snli_mnli":
        train_dataset = SNLIMNLIDataset(data_args.train_file, tokenizer)
    elif data_args.data_type == "wiki":
        train_dataset = WikiIdentityDataset(data_args.train_file, tokenizer)
    elif data_args.method == "simcse-sup":
        train_dataset = NLIDataset(data_args.train_file, tokenizer)
    else:
        raise ValueError

    trainer = CLTrainer(
        model=model,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.model_args = model_args

    # Training
    train_result = trainer.train()

    if trainer.is_world_process_zero():
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    # Evaluation
    if trainer.is_world_process_zero():
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(all=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f:
            logger.info("***** Eval results *****")
            f.write(
                f"{results['STS12']['all']['spearman']['all']:.4f},"
                f"{results['STS13']['all']['spearman']['all']:.4f},"
                f"{results['STS14']['all']['spearman']['all']:.4f},"
                f"{results['STS15']['all']['spearman']['all']:.4f},"
                f"{results['STS16']['all']['spearman']['all']:.4f},"
                f"{results['STSB-test']['all']['spearman']['all']:.4f},"
                f"{results['SICKR-test']['all']['spearman']['all']:.4f},"
                f"{results['STSB-dev']['all']['spearman']['all']:.4f}"
            )
    else:
        results = None

    return results


def main():
    torch.set_printoptions(precision=2, threshold=1e-7, sci_mode=False)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    args = parser.parse_args_into_dataclasses()
    train(args)


if __name__ == "__main__":
    main()
