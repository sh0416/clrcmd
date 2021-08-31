import json
import logging
import os
import sys
from dataclasses import dataclass, field
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
from transformers.file_utils import (
    cached_property,
    is_torch_tpu_available,
    torch_required,
)
from transformers.trainer_utils import is_main_process

from simcse.models import RobertaForCL
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05, metadata={"help": "Temperature for softmax."}
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": (
                "What kind of pooler to use (cls, cls_before_pooler, avg, "
                "avg_top2, avg_first_last)."
            )
        },
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        },
    )
    do_mlm: bool = field(
        default=False,
        metadata={"help": "Whether to use MLM auxiliary objective."},
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        },
    )
    mlp_only_train: bool = field(
        default=False, metadata={"help": "Use MLP only during training"}
    )
    dropout_prob: float = field(default=0.1, metadata={"help": "Dropout prob"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )

    # SimCSE's arguments
    train_file: str = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."},
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"
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
    # Evaluation
    # By default, we evaluate STS (dev) during training (for selecting best
    # checkpoints) and evaluate both STS and transfer tasks (dev) at the end of
    # training. Using --eval_transfer will allow evaluating both STS and
    # transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script
        # and it's the path to a json file, let's parse
        # it to get our arguments.
        json_file = os.path.abspath(sys.argv[1])
        args = parser.parse_json_file(json_file=json_file)
    else:
        args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = args

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and"
            " is not empty. Use --overwrite_output_dir to overcome."
        )

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "model_args.json")) as f:
        json.dump(training_args, f)
    with open(os.path.join(training_args.output_dir, "data_args.json")) as f:
        json.dump(data_args, f)
    with open(
        os.path.join(training_args.output_dir, "training_args.json")
    ) as f:
        json.dump(training_args, f)
    # Setup logging
    if is_main_process(training_args.local_rank):
        level = logging.INFO
    else:
        level = logging.WARN
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16} "
    )
    # Set the verbosity to info of the Transformers logger
    # (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
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
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "dropout_prob": model_args.dropout_prob,
    }
    if model_args.pooler_type in ["avg_top2", "avg_first_last"]:
        config_kwargs["output_hidden_states"] = True
    if model_args.model_name_or_path:
        if "roberta" in model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                **config_kwargs,
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.model_name_or_path:
        if "roberta" in model_args.model_name_or_path:
            tokenizer = RobertaTokenizer.from_pretrained(
                model_args.model_name_or_path, **tokenizer_kwargs
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    if model_args.model_name_or_path:
        if "roberta" in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
            )
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    # For the case where the tokenizer and model is different
    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    sent0_cname = "input_strs"
    sent1_cname = "input_strs2"

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "

        sentences1 = [
            tokenizer.convert_tokens_to_ids(x.split())
            for x in examples[sent0_cname]
        ]
        sentences2 = [
            tokenizer.convert_tokens_to_ids(x.split())
            for x in examples[sent1_cname]
        ]
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

        features = {
            key: [
                [sent_features[key][i], sent_features[key][i + total]]
                for i in range(total)
            ]
            for key in sent_features
        }
        return features

    if training_args.do_train:
        column_names = datasets["train"].column_names
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        print(train_dataset[0]["input_ids"])
    else:
        raise ValueError("no --do_train is set")

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.model_args = model_args

    # Training
    if training_args.do_train:
        train_result = trainer.train()

        if trainer.is_world_process_zero():
            output_train_file = os.path.join(
                training_args.output_dir, "train_results.txt"
            )
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
