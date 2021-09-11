import enum
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    HfArgumentParser,
    RobertaTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from simcse.data import PairDataCollator, create_perfect_overlap_pairs_from_tokens
from simcse.models import RobertaForTokenContrastiveLearning
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    loss_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use token alignment contrastive learning" " objective."
            )
        },
    )
    coeff_loss_token: float = field(
        default=0.1,
        metadata={
            "help": (
                "Coefficient for token alignment contrastive learning"
                " objective (only effective if --loss_token)."
            )
        },
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

    # Huggingface's original arguments.
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use (via the"
                " datasets library)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "The percentage of the train set used as validation set in"
                " case there's no validation split"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
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
        default=os.path.join("result", datetime.now().strftime("%Y%m%d_%H%M%S")),
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

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training
    # and evaluation files (see below) or just provide the name of one of the
    # public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub)
    #
    # For CSV/JSON files, this script will use the column called 'text' or the
    # first column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one
    # local process can concurrently download the dataset.
    data_files = {"train": data_args.train_file}
    ext = os.path.splitext(data_args.train_file)[1]
    if ext == ".txt":
        ext = "text"
    elif ext == ".csv":
        ext = "csv"
    elif ext == ".json":
        ext = "json"
    datasets = load_dataset(ext, data_files=data_files, cache_dir=".data/")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config_kwargs = {
        "hidden_dropout_prob": model_args.hidden_dropout_prob,
    }
    if model_args.pooler_type in ["avg_top2", "avg_first_last"]:
        config_kwargs["output_hidden_states"] = True
    if model_args.model_name_or_path:
        if "roberta" in model_args.model_name_or_path:
            tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, **config_kwargs
            )
            print(f"{config = }")
            model = RobertaForTokenContrastiveLearning.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                pooler_type=model_args.pooler_type,
                loss_mlm=model_args.loss_mlm,
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    # For the case where the tokenizer and model is different
    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    sent0_cname, sent1_cname = "input_strs", "input_strs2"

    def prepare_features(examples):
        for idx, text in enumerate(examples[sent0_cname]):
            assert text is not None, f"Example {idx = } is None"
        for idx, text in enumerate(examples[sent1_cname]):
            assert text is not None, f"Example {idx = } is None"

        total = len(examples[sent0_cname])

        tokens1 = [x.split() for x in examples[sent0_cname]]
        tokens2 = [x.split() for x in examples[sent1_cname]]
        pairs = [
            torch.tensor(
                create_perfect_overlap_pairs_from_tokens(x, y),
                dtype=torch.long,
            )
            for x, y in zip(tokens1, tokens2)
        ]
        sentences1 = [tokenizer.convert_tokens_to_ids(x) for x in tokens1]
        sentences2 = [tokenizer.convert_tokens_to_ids(x) for x in tokens2]
        text1 = tokenizer.batch_decode(sentences1)
        text2 = tokenizer.batch_decode(sentences2)
        for x, y in zip(text1, text2):
            assert x == y, f"{x} != {y}"
        sentences = sentences1 + sentences2

        sent_features = tokenizer.batch_encode_plus(
            sentences,
            is_split_into_words=True,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        features = {}
        for k, v in sent_features.items():
            features[k] = [(v[i], v[i + total]) for i in range(total)]
        features["pairs"] = pairs
        return features

    column_names = datasets["train"].column_names
    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    trainer = CLTrainer(
        model=model,
        data_collator=PairDataCollator(
            tokenizer,
            max_length=data_args.max_seq_length,
            mlm=model_args.loss_mlm,
        ),
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
    logger.info("*** Evaluate ***")
    results = trainer.evaluate(all=True)

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    return results


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    args = parser.parse_args_into_dataclasses()

    _, _, training_args = args
    # Set the verbosity to info of the Transformers logger
    # (on main process only):
    logger.info(f"{training_args.local_rank = }")
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    train(args)


if __name__ == "__main__":
    main()
