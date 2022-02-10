import argparse
import logging
import os

import torch
from transformers import TrainingArguments, set_seed, default_data_collator

from clrcmd.data.dataset import ContrastiveLearningCollator, NLIContrastiveLearningDataset
from clrcmd.models import create_contrastive_learning, create_tokenizer
from clrcmd.trainer import CLTrainer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--data-dir", type=str, help="Data directory", default="data")
parser.add_argument("--model", type=str, help="Model", choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"], default="bert-cls")
parser.add_argument("--output-dir", type=str, help="Output directory", default="ckpt")
parser.add_argument("--temp", type=float, help="Softmax temperature", default=0.05)
# fmt: on


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", filename="log/train.log"
    )
    args = parser.parse_args()
    logger.info("Hyperparameters")
    for k, v in vars(args).items():
        logger.info(f"{k} = {v}")

    training_args = TrainingArguments(
        args.output_dir,
        per_device_train_batch_size=128,
        learning_rate=5e-5,
        num_train_epochs=3,
        fp16=True,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
    )
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
    tokenizer = create_tokenizer(args.model)
    model = create_contrastive_learning(args.model, args.temp)
    model.train()

    train_dataset = NLIContrastiveLearningDataset(
        os.path.join(args.data_dir, "nli_for_simcse.csv"), tokenizer
    )

    trainer = CLTrainer(
        model=model,
        data_collator=ContrastiveLearningCollator(),
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    exit()

    # Training

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
                f"{results['STS12']['all_spearman_all']:.4f},"
                f"{results['STS13']['all_spearman_all']:.4f},"
                f"{results['STS14']['all_spearman_all']:.4f},"
                f"{results['STS15']['all_spearman_all']:.4f},"
                f"{results['STS16']['all_spearman_all']:.4f},"
                f"{results['STSB-test']['all_spearman_all']:.4f},"
                f"{results['SICKR-test']['all_spearman_all']:.4f},"
                f"{results['STSB-dev']['all_spearman_all']:.4f}"
            )
    else:
        results = None

    return results


if __name__ == "__main__":
    main()
