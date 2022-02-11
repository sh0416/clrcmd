import argparse
import logging
import os

from transformers import TrainingArguments, default_data_collator, set_seed

from clrcmd.data.dataset import (
    ContrastiveLearningCollator,
    NLIContrastiveLearningDataset,
    STSBenchmarkDataset,
)
from clrcmd.data.sts import load_sts_benchmark
from clrcmd.models import create_contrastive_learning, create_tokenizer
from clrcmd.trainer import STSTrainer, compute_metrics

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# fmt: off
parser.add_argument("--data-dir", type=str, help="Data directory", default="data")
parser.add_argument("--model", type=str, help="Model", choices=["bert-cls", "bert-avg", "bert-rcmd", "roberta-cls", "roberta-avg", "roberta-rcmd"], default="bert-cls")
parser.add_argument("--output-dir", type=str, help="Output directory", default="ckpt")
parser.add_argument("--temp", type=float, help="Softmax temperature", default=0.05)
# fmt: on


def main():
    args = parser.parse_args()
    training_args = TrainingArguments(
        os.path.join(args.output_dir, args.model),
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        num_train_epochs=1,
        fp16=True,
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_spearman",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
    )
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filename=f"log/train-{args.model}.log",
        )
    logger.info("Hyperparameters")
    for k, v in vars(args).items():
        logger.info(f"{k} = {v}")

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
    eval_dataset = STSBenchmarkDataset(
        load_sts_benchmark(args.data_dir, "stsb-dev")["dev"], tokenizer
    )

    trainer = STSTrainer(
        model=model,
        data_collator=ContrastiveLearningCollator(),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    logger.info(train_result)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-best"))
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
