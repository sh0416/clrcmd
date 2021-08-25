import collections
import math
import os
import time
import warnings
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.data.data_collator import (
    DataCollator,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging

from sentence_benchmark.evaluate import evaluate_sts

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

import copy
from datetime import datetime

import numpy as np
from filelock import FileLock
from transformers.optimization import Adafactor, AdamW, get_scheduler

from sentence_benchmark.data import Input, load_sickr_dev, load_stsb_dev

logger = logging.get_logger(__name__)


class CLTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(inputs: List[Input], param: Dict) -> Dict:
            return param

        def batcher(inputs: List[Input], param: Dict) -> np.ndarray:
            text1 = [" ".join(x.text1) for x in inputs]
            text2 = [" ".join(x.text2) for x in inputs]
            batch1 = self.tokenizer.batch_encode_plus(
                text1, return_tensors="pt", padding=True
            )
            batch2 = self.tokenizer.batch_encode_plus(
                text2, return_tensors="pt", padding=True
            )
            batch1 = {k: v.to(self.args.device) for k, v in batch1.items()}
            batch2 = {k: v.to(self.args.device) for k, v in batch2.items()}
            with torch.no_grad():
                outputs1 = self.model(
                    **batch1,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                )
                outputs2 = self.model(
                    **batch2,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                )
                outputs1 = outputs1.pooler_output
                outputs2 = outputs2.pooler_output
                score = F.cosine_similarity(outputs1, outputs2, dim=1)
                return score.cpu().numpy()

        # NOTE: I don't know how to evaluate SICK-R because it train something
        self.model.eval()
        dataset_stsb = load_stsb_dev(".data/STS/STSBenchmark")
        # dataset_sickr = load_sickr_dev(".data/SICK")
        result_stsb = evaluate_sts(dataset_stsb, {}, prepare, batcher)
        # result_sickr = evaluate_sts(dataset_sickr, {}, prepare, batcher)

        stsb_spearman = result_stsb["all"]["spearman"]["mean"]
        # sickr_spearman = result_sickr["all"]["spearman"][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            #    "eval_sickr_spearman": sickr_spearman,
        }
        self.log(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save.
        assert (
            _model_unwrap(model) is self.model
        ), "internal model should be a reference to self.model"

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                # Only save model when it is the best one
                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                # Save optimizer and scheduler
                if self.sharded_dpp:
                    self.optimizer.consolidate_state_dict()

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    with warnings.catch_warnings(
                        record=True
                    ) as caught_warnings:
                        xm.save(
                            self.lr_scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        reissue_pt_warnings(caught_warnings)
                elif self.is_world_process_zero() and not self.deepspeed:
                    # deepspeed.save_checkpoint above saves model/optim/sched
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    with warnings.catch_warnings(
                        record=True
                    ) as caught_warnings:
                        torch.save(
                            self.lr_scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                    reissue_pt_warnings(caught_warnings)

                # Save the Trainer state
                if self.is_world_process_zero():
                    self.state.save_to_json(
                        os.path.join(output_dir, "trainer_state.json")
                    )
        else:
            # Save model checkpoint
            checkpoint_folder = (
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune

                    run_id = tune.get_trial_id()
                run_name = (
                    self.hp_name(trial)
                    if self.hp_name is not None
                    else f"run-{run_id}"
                )
                output_dir = os.path.join(
                    self.args.output_dir, run_name, checkpoint_folder
                )
            else:
                output_dir = os.path.join(
                    self.args.output_dir, checkpoint_folder
                )

                self.store_flos()

            self.save_model(output_dir)
            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)

            # Save optimizer and scheduler
            if self.sharded_dpp:
                self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                    reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                # deepspeed.save_checkpoint above saves model/optim/sched
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, "optimizer.pt"),
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                reissue_pt_warnings(caught_warnings)

            # Save the Trainer state
            if self.is_world_process_zero():
                self.state.save_to_json(
                    os.path.join(output_dir, "trainer_state.json")
                )

            # Maybe delete some older checkpoints.
            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)
