import os
import optuna
import logging
from argparse import Namespace
from datetime import datetime
from optuna import Trial
from typing import Callable, Dict, Tuple

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers.hf_argparser import HfArgumentParser
from run_train_simcse import (
    ModelArguments,
    DataTrainingArguments,
    OurTrainingArguments,
    train,
)

logger = logging.getLogger(__name__)
Arguments = Tuple[ModelArguments, DataTrainingArguments, OurTrainingArguments]


def cleanup_trial(trial: Trial, output: Dict) -> float:
    def f(d: Dict, prefix: str):
        for k, v in d.items():
            if type(v) == dict:
                f(v, f"{prefix}_{k}")
            else:
                trial.set_user_attr(f"{prefix}_{k}", v)

    f(output, "")
    return output["STSB-dev"]["all"]["spearman"]["all"]


def search_hparams(
    sample_configuration: Callable[[Trial], Namespace],
    compute_algorithm: Callable[[Namespace], Dict],
    cleanup_trial: Callable[[Trial, Dict], float],
):
    def f(trial):
        if dist.get_rank() != -1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, device=torch.device(dist.get_rank())
            )
        hparams = sample_configuration(trial)
        outputs = compute_algorithm(hparams)
        objective = cleanup_trial(trial, outputs)
        return objective

    return f


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    # NOTE: Inside `parse_args_into_dataclasses`, call `init_dist_process`
    #       So, we don't have to call it manually
    args: Arguments = parser.parse_args_into_dataclasses()

    # Setup logging
    if dist.get_rank() == 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    def sample_configuration(trial: Trial) -> Arguments:
        model_args, data_args, training_args = args

        # Fixed hyperparameters
        data_args.max_seq_length = 32
        training_args.output_dir = os.path.join(
            "result", datetime.now().strftime("%Y%m%d_%H%M")
        )
        training_args.per_device_train_batch_size = 64
        training_args.per_device_eval_batch_size = 128
        training_args.gradient_accumulation_steps = 2

        # Search hyperparameters
        training_args.seed = trial.suggest_categorical(
            "seed", [0, 1, 2, 3, 4, 5]
        )
        training_args.learning_rate = trial.suggest_categorical(
            "learning_rate", [5e-6, 1e-5, 5e-5, 1e-4]
        )
        model_args.pooler_type = trial.suggest_categorical(
            "pooler_type",
            ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        )

        model_args.hidden_dropout_prob = trial.suggest_categorical(
            "hidden_dropout_prob", [0.01, 0.05, 0.1, 0.15]
        )
        return model_args, data_args, training_args

    n_trials = 40
    target_fn = search_hparams(sample_configuration, train, cleanup_trial)
    study_name = "study_loss_token"
    storage = f"sqlite:///study1.db"
    if dist.get_rank() == 0:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.RandomSampler(0),
        )
        study.optimize(target_fn, n_trials=n_trials)
        df = study.trials_dataframe()
        df.to_csv(f"{study_name}.csv", index=False)
    else:
        for _ in range(n_trials):
            try:
                target_fn(None)
            except optuna.TrialPruned:
                pass


if __name__ == "__main__":
    main()
