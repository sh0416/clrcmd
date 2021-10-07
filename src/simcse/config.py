import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from transformers import TrainingArguments


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
