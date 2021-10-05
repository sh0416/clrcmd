import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Optional

import torch
import transformers
from torch.utils.data import ConcatDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from simcse.data.dataset import (
    NLIDataset,
    PairedContrastiveLearningDataset,
    WikiEDADataset,
    WikiIdentityDataset,
    WikiRepetitionDataset,
    collate_fn,
)
from simcse.models import create_contrastive_learning
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    # SimCSE's arguments
    temp: float = field(default=0.05, metadata={"help": "Temperature for softmax."})
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": (
                "What kind of pooler to use (cls, cls_before_pooler, avg, "
                "avg_top2, avg_first_last)."
            )
        },
    )
    loss_rwmd: bool = field(
        default=False,
        metadata={"help": "Whether to use rwmd metric learning objective."},
    )
    mlp_only_train: bool = field(
        default=False, metadata={"help": "Use MLP only during training"}
    )
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "Dropout prob"})
    loss_mlm: bool = field(default=False, metadata={"help": "Add MLM loss"})
    coeff_loss_mlm: float = field(
        default=0.1,
        metadata={"help": "Coefficient for masked language model objective"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: str = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."},
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization."
                " Sequences longer than this will be truncated."
            )
        },
    )
    method: str = field(default="simcse-unsup", metadata={"help": "Training method"})
    add_typo_corpus: bool = field(
        default=False, metadata={"help": "Add github typo corpus"}
    )
    typo_corpus_filepath: str = field(
        default=None, metadata={"help": "Typo corpus path"}
    )
    dup_rate: float = field(
        default=0.08, metadata={"help": "Duplication rate for ESimCSE"}
    )

    def __post_init__(self):
        assert self.train_file is not None, "No --train_file is set"
        extension = os.path.splitext(self.train_file)[1]
        assert extension in [
            ".csv",
            ".json",
            ".txt",
        ], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=os.path.join(
            "/home/sh0416/checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
        metadata={
            "help": (
                "The output directory where the model predictions and"
                " checkpoints will be written."
            )
        },
    )

    def __post_init__(self):
        if (
            os.path.exists(self.output_dir)
            and os.listdir(self.output_dir)
            and self.do_train
            and not self.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self.output_dir}) already exists and"
                " is not empty. Use --overwrite_output_dir to overcome."
            )
        return super().__post_init__()


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
            model_args.model_name_or_path, add_prefix_space=True
        )
        model = create_contrastive_learning(model_args)
        model.train()
    else:
        raise NotImplementedError()

    if data_args.method == "simcse-unsup":
        train_dataset = WikiIdentityDataset(data_args.train_file, tokenizer)
    elif data_args.method == "simcse-sup":
        train_dataset = NLIDataset(data_args.train_file, tokenizer)
    elif data_args.method == "esimcse":
        train_dataset = WikiRepetitionDataset(
            data_args.train_file, tokenizer, dup_rate=data_args.dup_rate
        )
    elif data_args.method == "edasimcse":
        train_dataset = WikiEDADataset(data_args.train_file, tokenizer)
    else:
        raise ValueError

    if data_args.add_typo_corpus:
        typo_dataset = PairedContrastiveLearningDataset(
            data_args.typo_corpus_filepath, tokenizer
        )
        train_dataset = ConcatDataset((train_dataset, typo_dataset))
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
