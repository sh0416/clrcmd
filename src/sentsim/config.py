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
    # Loss
    loss_type: str = field(
        default="sbert-cls",
        metadata={
            "help": "Loss type for training",
            "choices": [
                "sbert-cls",
                "sbert-avg",
                "simcse-cls",
                "simcse-avg",
                "rwmdcse",
            ],
        },
    )
    # Temperature softmax
    temp: float = field(default=0.05, metadata={"help": "Temperature for softmax."})
    # Dropout
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "Dropout prob"})
    # Miscalleneous
    mlp_only_train: bool = field(
        default=False, metadata={"help": "Use MLP only during training"}
    )
    coeff_mlm: float = field(
        default=0.1,
        metadata={"help": "Coefficient for masked language model objective"},
    )

    # RWMD's arguments
    loss_rwmd: bool = field(
        default=False,
        metadata={"help": "Whether to use rwmd metric learning objective."},
    )
    layer_idx: Optional[int] = field(
        default=None, metadata={"help": "Index of final layer"}
    )
    dense_rwmd: bool = field(
        default=False, metadata={"help": "Whether to use dense rwmd"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_type: str = field(
        default="simcse-nli", metadata={"choices": ["simcse-nli", "wiki"]}
    )
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
    eval_file: str = field(default=None, metadata={"help": "rootpath for STS"})
    output_dir: str = field(
        default=f"/home/sh0416/checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
