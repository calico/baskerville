import datetime
import gc
import json
import math
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from schedulefree import AdamWScheduleFree

from baskerville import dataset
from baskerville import metrics
from baskerville.types import BatchData, ModelOutput

import re

# Seconds between live progress lines in non-interactive (GCP) runs.
_PROGRESS_INTERVAL_S = 60.0


def _emit_progress(desc, epoch, step, total, t0, last_emit):
    """Emit a newline-terminated progress line, at most every _PROGRESS_INTERVAL_S.

    tqdm draws its bar with a carriage return, which never surfaces in
    line-oriented log sinks (Cloud Logging). On GCP we disable the bar and print
    discrete status lines to stderr instead, so the epoch's step/rate/ETA can be
    followed live. Returns the (possibly updated) last-emit timestamp.
    """
    now = time.time()
    if now - last_emit < _PROGRESS_INTERVAL_S:
        return last_emit
    elapsed = max(now - t0, 1e-9)
    rate = step / elapsed
    line = f"[progress] {desc} epoch {epoch} step {step}"
    if total:
        line += f"/{total}"
        if rate > 0:
            eta = int((total - step) / rate)
            line += f" ETA {datetime.timedelta(seconds=eta)}"
    line += f" {rate:.2f} it/s"
    print(line, file=sys.stderr, flush=True)
    return now


STOP_STAT_NAMES = {
    "weighted_r_r2": {"r": 1, "r2": 0.25},
    "loss": {"loss": -1},
    "r": {"r": 1},
    "r2": {"r2": 1},
}


def parse_stop_stat(stop_stat):
    """Early-stop statistic as {metric: weight}, from a dict or a named preset.

    Metric keys are DatasetMetrics.compute() keys (loss, r, r2, spec, spec/g, ...).
    """
    if isinstance(stop_stat, dict):
        return stop_stat
    if stop_stat not in STOP_STAT_NAMES:
        raise ValueError(
            f"stop_stat '{stop_stat}' not recognized. "
            f"Valid names: {list(STOP_STAT_NAMES)}, or a {{metric: weight}} dict"
        )
    return STOP_STAT_NAMES[stop_stat]


class Trainer:
    """Model training class.

    Args:
      params (dict): Training parameters dictionary.
      train_data: Dataset object or list of Dataset objects.
      eval_data: Dataset object or list of Dataset objects.
      out_dir (str): Output directory name.
    """

    def __init__(
        self,
        params: dict,
        train_data,
        eval_data,
        out_dir: str,
        model_heads=None,
        checkpoint_callback=None,
        prune_callback=None,
    ):
        self.params = params
        self.train_data = train_data
        self.eval_data = eval_data
        self.out_dir = out_dir
        self.model_heads = model_heads
        # Optional no-arg callable invoked after each per-epoch checkpoint write
        # and once more after the final COMPLETE/FAILED marker. Kept generic so
        # the trainer stays storage-agnostic; hound_train wires this to a GCS
        # sync of out_dir when running on the GCP backend.
        self.checkpoint_callback = checkpoint_callback
        # Optional callable(filename) invoked when an old model_best snapshot is
        # pruned from disk, so its GCS copy can be removed too (the per-epoch
        # sync is upload-only and would otherwise leak snapshots in the bucket).
        self.prune_callback = prune_callback
        self.model_di = self.params.get("model_di", None)
        self.batch_size = self.params["batch_size"]
        self.num_workers = self.params.get("num_workers", 0)
        self.num_eval_workers = self.params.get("num_eval_workers", None)

        if self.num_eval_workers is None:
            self.num_eval_workers = self.num_workers

        # device
        if torch.cuda.is_available():
            self.device = "cuda"
            pin_mem = self.params.get("pin_mem", True)
            persistent_workers = self.params.get("persistent_workers", True)
        elif torch.backends.mps.is_available():
            self.device = "mps"
            pin_mem = False
            persistent_workers = True
        else:
            self.device = "cpu"
            pin_mem = False
            persistent_workers = True

        if self.num_workers == 0:
            persistent_workers = False

        # get oversampling parameter
        upsampling_rates = self.params.get("upsampling_rates", None)

        # combine training datasets into single dataloader
        self.num_datasets = len(self.train_data)
        train_multidata = dataset.MultiDataset(self.train_data)

        train_sampler = dataset.MultiSampler(
            train_multidata,
            batch_size=self.batch_size,
            upsampling_rates=upsampling_rates,
            mode="train",
        )

        # Use custom collate function
        if isinstance(self.train_data[0], dataset.SeqDatasetTeacher):
            collate_fn = self.train_data[0].collate_fn
            pin_mem = False
        else:
            collate_fn = self._collate_multi

        self.train_dataload = torch.utils.data.DataLoader(
            train_multidata,
            batch_sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

        # create separate eval dataloaders, using basic in-order sampling
        self.eval_dataload = []
        for ed in self.eval_data:
            self.eval_dataload.append(
                torch.utils.data.DataLoader(
                    ed,
                    batch_size=self.batch_size,
                    num_workers=self.num_eval_workers,
                    drop_last=False,
                    pin_memory=pin_mem,
                    persistent_workers=False,
                    collate_fn=BatchData.collate,
                )
            )

        # define the subset of datasets to evaluate early stopping on
        self.early_stop_datasets = self.params.get(
            "early_stop_datasets", len(self.eval_dataload)
        )

        # after how many epochs should all species (including non-early-stop species) be validated
        self.full_validation_epoch_rate = self.params.get(
            "full_validation_epoch_rate", 1
        )

        # compute batches/epoch
        self.train_epochs_min = self.params.get("train_epochs_min", 1)
        self.train_epochs_max = self.params.get("train_epochs_max", 10000)

        self.train_batches_max = self.params.get("train_batches_max", None)
        self.save_best_max = self.params.get("save_best_max", 5)

        self.save_check_after_epoch = self.params.get("save_check_after_epoch", 20)
        self.save_check_epoch_rate = self.params.get("save_check_epoch_rate", None)

        self.save_check_after_epoch2 = self.params.get("save_check_after_epoch2", None)
        self.save_check_epoch_rate2 = self.params.get("save_check_epoch_rate2", None)

        # loss
        self.spec_weight = self.params.get("spec_weight", 1)
        self.total_weight = self.params.get("total_weight", 1)
        self.weight_range = self.params.get("weight_range", 1)
        self.weight_exp = self.params.get("weight_exp", 1)
        self.loss = self.params.get("loss", "poisson").lower()
        self.is_mlm = self.loss == "mlm"
        if self.loss != "mlm":
            self.loss_fn = parse_loss(
                self.loss,
                self.spec_weight,
                self.total_weight,
                self.weight_range,
                self.weight_exp,
            )

        # gene expression loss
        self.loss_gene = self.params.get("loss_gene", "poisson").lower()
        if self.loss_gene == "poisson":
            self.loss_gene_fn = metrics.GenePoissonLoss()
        elif self.loss_gene == "mse":
            self.loss_gene_fn = metrics.GeneMSELoss()
        else:
            raise ValueError(
                f"Gene loss function '{self.loss_gene}' not recognized. "
                "Use 'poisson' or 'mse'."
            )

        self.gene_weight = self.params.get("gene_weight", 1)

        # get species-specific loss coefficients
        self.species_weights = self.params.get(
            "species_weights", [1 for _ in range(self.num_datasets)]
        )

        # get track-specific loss coefficients
        if self.is_mlm and (
            self.params.get("loss_weight") or self.params.get("loss_weight_gene")
        ):
            # the MLM loss scores masked nucleotides, not tracks, so weights
            # would be silently dropped rather than applied
            raise ValueError("loss_weight is not supported with the mlm loss")
        self.target_weights = self._setup_target_weights()
        self.gene_target_weights = self._setup_target_weights(gene=True)

        # optimization
        self.optimizer = self.params.get("optimizer", "adam")
        if "learning_rate_min" in self.params:
            self.learning_rate = self.params["learning_rate_max"]
            self.learning_rate_min = self.params["learning_rate_min"]
        else:
            self.learning_rate = self.params["learning_rate"]
            self.learning_rate_min = self.learning_rate
        self.global_clipnorm = self.params.get("global_clipnorm", 1000.0)
        self.weight_decay = self.params.get("weight_decay", 0.0)

        # get weight decay group rules
        self.weight_decay_rules = self.params.get(
            "weight_decay_rules", [{"regex": "bias", "weight_decay": 0.0}]
        )

        # mixed precision
        self.mix_dtype = self.params.get("mix_dtype", torch.float32)
        if self.mix_dtype == "float16":
            self.mix_dtype = torch.float16
        elif self.mix_dtype == "bfloat16":
            self.mix_dtype = torch.bfloat16
        elif isinstance(self.mix_dtype, str):
            raise ValueError(f"Mixed precision dtype {self.mix_dtype} not recognized")

        # schedule
        self.schedule = self.params.get("schedule", "constant")
        self.warmup_steps = self.params.get("warmup_steps", 1)
        # per-dataset train-mode batches used to recompute BatchNorm running
        # stats before validation under the free schedule (see _eval_prep)
        self.eval_prep_batches = self.params.get("eval_prep_batches", 50)
        # total warmup batches, clamped to one pass over the train loader
        self.eval_prep_steps = min(
            self.eval_prep_batches * self.num_datasets, len(self.train_dataload)
        )
        self.patience = self.params.get("patience", self.train_epochs_max)
        self.reset_unimproved = self.params.get("reset_unimproved", False)

        # early stopping statistic: {metric: weight}, summed over datasets
        stop_stat = self.params.get("stop_stat", "weighted_r_r2")
        self.stop_stat = parse_stop_stat(stop_stat)
        self.stop_stat_loss_fallback = isinstance(stop_stat, str)
        self.spec_group_min = self.params.get("spec_group_min", 20)

        # MLM parameters (only loaded when loss == "mlm")
        if self.loss == "mlm":
            self.mask_rate = self.params.get("mask_rate", 0.15)
            # Loss scaling for exons/repeats - values < 1 downweight (e.g. 0.1 = 10%).
            # "non_*" default to 1.0 so setting only the exon/repeat scale works.
            self.exon_loss_scale = self.params.get("exon_loss_scale", None)
            self.non_exon_loss_scale = self.params.get("non_exon_loss_scale", 1.0)
            self.repeat_loss_scale = self.params.get("repeat_loss_scale", None)
            self.non_repeat_loss_scale = self.params.get("non_repeat_loss_scale", 1.0)
            # BERT-style masking: 80% [MASK], 10% random, 10% original
            self.use_bert = self.params.get("use_bert", False)
            # average multiple random masks at eval time
            self.repeat_eval = self.params.get("repeat_eval", 1)
            # RC augmentation is handled in SeqDatasetMLM.__getitem__, not here,
            # to avoid double-augmenting.

    def _setup_target_weights(self, gene: bool = False):
        """Per-dataset (1, T) loss weight tensors, or None where every weight is 1.

        Args:
            gene: weight the gene targets table under ``loss_weight_gene``,
              rather than the coverage one under ``loss_weight``.

        None selects the loss's unweighted reduction, so runs that configure no
        weights are numerically unchanged.
        """
        head = "gene" if gene else "coverage"
        rules = self.params.get("loss_weight_gene" if gene else "loss_weight", [])
        df_attr = "targets_gene_df" if gene else "targets_df"
        target_weights = []

        for di, data in enumerate(self.train_data):
            targets_df = getattr(data, df_attr, None)
            weights = None
            if targets_df is not None and not targets_df.empty:
                weights = parse_target_weights(targets_df, rules)

            if weights is None or np.allclose(weights, 1):
                target_weights.append(None)
                continue

            # gene targets are aggregated over a gene body, so they carry no strands
            if not gene and not np.allclose(weights, weights[data.strand_pair]):
                raise ValueError(
                    f"dataset {di} loss weights differ within a strand pair"
                )

            values, counts = np.unique(weights, return_counts=True)
            print(f"-- dataset {di} {head} loss weights --")
            print("\n".join(f"{v:8.4f} x {c}" for v, c in zip(values, counts)))

            target_weights.append(
                torch.tensor(
                    weights, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            )

        return target_weights

    @staticmethod
    def _collate_multi(batch):
        """Collate function for MultiDataset that returns (dataset_idx, BatchData).

        Static so DataLoader workers can pickle it under Python 3.14's forkserver
        start method; a bound method would pickle the whole Trainer.
        """
        dataset_indices = torch.tensor([item[0] for item in batch])
        batch_data = BatchData.collate([item[1] for item in batch])
        return dataset_indices, batch_data

    def _compute_loss(
        self,
        yh: ModelOutput,
        y: torch.Tensor | None,
        yg: torch.Tensor | None,
        gene_presence: torch.Tensor | None,
        species_weight: torch.Tensor,
        di: int,
    ) -> torch.Tensor:
        """Compute combined loss from model output."""
        loss = torch.tensor(0.0, device=self.device)

        if yh.has_coverage and y is not None:
            cov_loss = self.loss_fn(yh.coverage, y, self.target_weights[di])
            loss = loss + cov_loss * species_weight

        if yh.has_gene and yg is not None:
            loss = (
                loss
                + self.loss_gene_fn(
                    yh.gene, yg, gene_presence, self.gene_target_weights[di]
                )
                * species_weight
                * self.gene_weight
            )

        return loss

    def _compute_stop_stat(self, results_list):
        """Weighted sum of stop_stat metrics over datasets (higher is better).

        Named presets fall back to -loss for datasets without coverage r.
        Explicit weights apply to all datasets; missing metrics contribute 0
        but must be present in at least one dataset.
        """
        stat = 0
        scored_results = []
        for results in results_list:
            if self.stop_stat_loss_fallback and results["r"] is None:
                stat -= results["loss"]
            else:
                scored_results.append(results)
        for key, weight in self.stop_stat.items():
            values = [res[key] for res in scored_results if res.get(key) is not None]
            if scored_results and not values:
                raise ValueError(f"stop_stat metric '{key}' absent from all datasets")
            stat += weight * sum(values)
        return stat

    def _eval_prep(self):
        """Prepare for validation under a schedule-free optimizer.

        Schedule-free keeps the eval-time weights distinct from the training
        iterate, so BatchNorm running stats must be recomputed for those weights
        with a few training-mode forward passes before validation (model is still
        in train() mode here). The optimizer is always swapped to its eval point,
        but the warmup is skipped when the model has no running-stat buffers
        (e.g. a LayerNorm/RMSNorm model) since there is nothing to recompute.
        """
        if self.schedule != "free":
            return
        self.optimizer.eval()
        # Only BatchNorm-style running stats go stale at the eval-point weights.
        if not self._has_running_stats:
            return
        with torch.no_grad():
            evalprep_iter = iter(self.train_dataload)
            for _ in range(self.eval_prep_steps):
                di_tensor, batch_data = next(evalprep_iter)
                if self.is_mlm:
                    x, label, exon_mask, repeat_mask = self._mlm_unpack(batch_data)
                    species_id = self._mlm_species_id(label, "a batch")
                    mask_size = max(1, int(self.mask_rate * x.shape[2]))
                    x_masked, _, _, _ = self._mlm_prep(
                        x,
                        mask_size,
                        exon_mask=exon_mask,
                        repeat_mask=repeat_mask,
                        training=True,
                    )
                    with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                        self.model_eval(x_masked, hi=0, di=species_id)
                else:
                    di = di_tensor[0].item()
                    x, _, _, gene_presence, gene_out_mask = self._unpack_batch(
                        batch_data
                    )
                    hi = di if self.model_heads is None else self.model_heads
                    with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                        self.model_eval(
                            x,
                            hi,
                            di=self.model_di,
                            gene_out_mask=gene_out_mask,
                            gene_presence=gene_presence,
                        )
        torch.cuda.empty_cache()

    def fit(self, seqnn_model):
        """Fit the model, dispatching the task-specific parts by ``self.is_mlm``.

        Coverage-supervised and masked-language-model (MLM) training share one
        epoch loop and all checkpoint / early-stop / logging scaffolding. The
        parts that genuinely differ — metrics, the per-batch step, validation,
        and per-epoch logging — live in the ``*_cov`` / ``*_mlm`` helpers.

        Args:
            seqnn_model: SeqNN
        """
        self.seqnn_model = seqnn_model
        self.model = self.seqnn_model.model
        self._setup_optimization()
        self._init_metrics()
        epoch_start, valid_best, unimproved = self._load_checkpoint()
        self.loss_nan = False

        log_out = open(f"{self.out_dir}/log.txt", "a")
        self.tqdm_total = (
            self.train_batches_max
            if self.train_batches_max is not None
            else len(self.train_dataload)
        )

        # In GCP runs the tqdm bar (carriage-return) doesn't surface in Cloud
        # Logging; disable it and emit periodic newline progress lines instead.
        self.log_progress = os.environ.get("GCPRUNNER_OUTPUT_DIR_GCS") is not None

        # last_epoch tracks the last epoch whose body ran to completion, so the
        # terminal marker is correct even when the loop breaks at the top
        # (early-stop / loss_nan) or never runs (resuming a finished run).
        last_epoch = epoch_start - 1
        for ei in range(epoch_start, self.train_epochs_max):
            if self.loss_nan or (
                ei >= self.train_epochs_min and unimproved > self.patience
            ):
                break

            # train
            self.model.train()
            if self.schedule == "free":
                self.optimizer.train()
            t0 = time.time()
            self._train_epoch(ei)

            # evaluate
            self._eval_prep()
            self.model.eval()
            with torch.no_grad():
                if self.is_mlm:
                    self._validate_mlm()
                else:
                    n_eval_datasets = self._validate_cov(ei)
            gc.collect()

            # log, select the best model, checkpoint
            self._log_epoch_header(log_out, ei, t0)
            if self.is_mlm:
                early_stop_stat = self._log_epoch_mlm(log_out)
            else:
                early_stop_stat = self._log_epoch_cov(log_out, ei, n_eval_datasets)
            valid_best, unimproved = self._save_epoch(
                ei, early_stop_stat, valid_best, unimproved, log_out
            )
            self._reset_metrics()
            last_epoch = ei

        log_out.close()

        # terminal marker: COMPLETE on a clean finish, FAILED on loss_nan.
        # loss_nan returns normally (process exits 0) so the GCP backend treats
        # it as terminal-bad rather than a retryable crash.
        if self.loss_nan:
            with open(f"{self.out_dir}/FAILED", "w") as marker_out:
                json.dump({"reason": "loss_nan", "epoch": last_epoch}, marker_out)
        else:
            reason = "early_stop" if unimproved > self.patience else "max_epochs"
            with open(f"{self.out_dir}/COMPLETE", "w") as marker_out:
                json.dump(
                    {
                        "epoch": last_epoch,
                        "valid_best": float(valid_best),
                        "reason": reason,
                    },
                    marker_out,
                )

        # final hook so the terminal marker (and last state) reaches GCS
        if self.checkpoint_callback is not None:
            self.checkpoint_callback()

    def _init_metrics(self):
        """Build the metric accumulators for the active task."""
        if self.is_mlm:
            self.train_loss = metrics.Loss()
            self.valid_loss = metrics.Loss()
            return

        model_head_cov = self.model.heads_cov is not None
        model_head_gene = self.model.heads_gene is not None

        self.train_metrics = []
        self.valid_metrics = []
        for di in range(self.num_datasets):
            has_coverage = self.train_data[di].has_coverage and model_head_cov
            has_gene = self.train_data[di].has_genes and model_head_gene

            if has_coverage:
                num_targets = (
                    self.seqnn_model.output_depth(self.model_heads)
                    if self.model_heads == -1
                    else self.train_data[di].num_targets
                )
            else:
                num_targets = None

            num_gene_targets = (
                self.train_data[di].num_gene_targets if has_gene else None
            )

            train_metrics = metrics.DatasetMetrics(
                has_coverage,
                has_gene,
                num_targets,
                num_gene_targets,
                self.device,
                targets_df=getattr(self.train_data[di], "targets_df", None),
                target_hist=getattr(self.train_data[di], "target_hist", None),
                spec_group_min=self.spec_group_min,
            )
            self.train_metrics.append(train_metrics)
            # valid shares train's spec tables (same targets)
            self.valid_metrics.append(
                metrics.DatasetMetrics(
                    has_coverage,
                    has_gene,
                    num_targets,
                    num_gene_targets,
                    self.device,
                    targets_df=getattr(self.train_data[di], "targets_df", None),
                    spec_group_min=self.spec_group_min,
                    spec_like=train_metrics.spec,
                )
            )

        # Validate specificity keys against the datasets used for stopping.
        stop_metrics = (
            self.valid_metrics[: min(self.early_stop_datasets, len(self.eval_data))]
            if self.eval_data
            else self.train_metrics
        )
        spec_keys = set()
        for m in stop_metrics:
            if m.spec is not None and m.spec.enabled and m.spec.groups:
                spec_keys.add("spec")
                spec_keys.update(f"spec/{g}" for g in m.spec.groups)
        for key in self.stop_stat:
            if (key == "spec" or key.startswith("spec/")) and key not in spec_keys:
                raise ValueError(f"stop_stat metric '{key}' absent from all datasets")

    def _load_checkpoint(self):
        """Seed saved-best bookkeeping and resume from checkpoint if present.

        Returns:
            (epoch_start, valid_best, unimproved)
        """
        # track saved best model files (seed from disk for resume)
        self.saved_best_files = sorted(
            [
                f
                for f in os.listdir(self.out_dir)
                if re.match(r"model_best\d+\.pth$", f)
            ],
            key=lambda f: int(re.search(r"(\d+)", f).group(1)),
        )

        epoch_start = 0
        valid_best = -np.inf
        unimproved = 0
        if os.path.exists(f"{self.out_dir}/checkpoint.pth"):
            checkpoint = torch.load(f"{self.out_dir}/checkpoint.pth")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.schedule == "cosine":
                # Accept checkpoints from the previous CosineAnnealingLR phase.
                cosine_state = checkpoint["scheduler"]["_schedulers"][1]
                cosine_state.setdefault(
                    "lr_lambdas", [None] * len(self.optimizer.param_groups)
                )
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            valid_best = checkpoint["valid_best"]
            epoch_start = checkpoint["epoch"]
            unimproved = 0 if self.reset_unimproved else checkpoint["unimproved"]
        return epoch_start, valid_best, unimproved

    def _log_epoch_cov(self, log_out, ei, n_eval_datasets):
        """Log per-dataset coverage metrics, append all of them to metrics.tsv,
        and return the early-stop stat."""
        stop_results = []
        table_rows = []
        for di in range(self.num_datasets):
            log_out.write(f"\n  Data {di}")

            # training metrics
            train_results = self.train_metrics[di].compute()
            log_out.write(
                f" - {self.train_metrics[di].format_log('train', train_results)}"
            )
            self.loss_nan |= math.isnan(train_results["loss"])
            split_results = [("train", train_results)]

            # validation metrics
            if len(self.eval_data) > 0 and di < n_eval_datasets:
                valid_results = self.valid_metrics[di].compute()
                log_out.write(
                    f" - {self.valid_metrics[di].format_log('valid', valid_results)}"
                )
                if di < self.early_stop_datasets:
                    stop_results.append(valid_results)
                self.loss_nan |= math.isnan(valid_results["loss"])
                split_results.append(("valid", valid_results))
            elif len(self.eval_data) == 0:
                stop_results.append(train_results)

            for split, results in split_results:
                for key, value in results.items():
                    if value is not None:
                        metric, _, group = key.partition("/")
                        table_rows.append(
                            f"{ei}\t{split}\t{di}\t{group or 'all'}\t{metric}\t{value:.6g}\n"
                        )

        table_file = f"{self.out_dir}/metrics.tsv"
        write_header = not os.path.exists(table_file)
        with open(table_file, "a") as table_out:
            if write_header:
                table_out.write("epoch\tsplit\tdataset\tgroup\tmetric\tvalue\n")
            table_out.writelines(table_rows)

        return self._compute_stop_stat(stop_results)

    def _log_epoch_header(self, log_out, ei, t0):
        """Write the shared ``Epoch N - HH:MM:SS[, lr: ...]`` prefix."""
        epoch_time = time.time() - t0
        format_time = str(datetime.timedelta(seconds=epoch_time)).split(".")[0]
        if self.schedule == "free":
            log_out.write(f"Epoch {ei} - {format_time}")
        else:
            # report the base lr; group 0 may carry an lr_mult
            lr_mult = self.optimizer.param_groups[0].get("lr_mult") or 1
            lr = self.scheduler.get_last_lr()[0] / lr_mult
            log_out.write(f"Epoch {ei} - {format_time}, lr: {lr:.6f}")

    def _log_epoch_mlm(self, log_out):
        """Log scalar MLM train/valid loss; return ``-valid_loss`` (higher is better)."""
        train_loss_epoch = self.train_loss.compute()
        log_out.write(f" - train_loss: {train_loss_epoch:.5f}")
        self.loss_nan |= math.isnan(train_loss_epoch)

        valid_loss_epoch = self.valid_loss.compute()
        log_out.write(f" - valid_loss: {valid_loss_epoch:.5f}")
        self.loss_nan |= math.isnan(valid_loss_epoch)

        # for MLM, lower loss is better
        return -valid_loss_epoch

    def _make_optimizer(self):
        """Create optimizer with regularization over ``self.model`` parameters."""
        # get all named parameters
        named_params = list(self.model.named_parameters())

        # get partitioned parameters sets
        weight_decay_groups = self._partition_weight_decay_groups(named_params)

        # apply optional per-group learning-rate multipliers (relative to the base lr,
        # so cosine/schedule-free scaling stays proportional through training)
        for group in weight_decay_groups:
            if group["lr_mult"] is not None:
                group["lr"] = self.learning_rate * group["lr_mult"]

        beta1 = self.params.get("beta1", 0.9)
        beta2 = self.params.get("beta2", 0.999)
        momentum = self.params.get("momentum", 0.9)

        # choose optimizer
        if self.optimizer == "adam":
            optimizer_method = torch.optim.Adam
            optimizer_params = {"betas": (beta1, beta2)}
        elif self.optimizer == "adamw":
            optimizer_params = {"betas": (beta1, beta2)}
            if self.schedule == "free":
                optimizer_method = AdamWScheduleFree
                optimizer_params["warmup_steps"] = self.warmup_steps
            else:
                optimizer_method = torch.optim.AdamW
        elif self.optimizer == "sgd":
            optimizer_method = torch.optim.SGD
            optimizer_params = {"momentum": momentum}
        else:
            raise ValueError(f"Optimizer {self.optimizer} not recognized")

        # define optimizer w/ regularization
        self.optimizer = optimizer_method(
            weight_decay_groups,
            lr=self.learning_rate,
            **optimizer_params,
        )

    def _make_scheduler(self):
        """Create learning rate scheduler."""

        if self.schedule == "free":
            # dummy placeholder, because schedule is within optimizer
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)

        else:
            # Create linear warmup scheduler starting from 0
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )

            if self.schedule == "constant":
                self.scheduler = warmup_scheduler

            elif self.schedule == "cosine":
                # Create cosine annealing scheduler
                train_epoch_steps = (
                    self.train_batches_max
                    if self.train_batches_max is not None
                    else len(self.train_dataload)
                )
                max_steps = self.train_epochs_max * train_epoch_steps
                decay_steps = max_steps - self.warmup_steps
                min_factor = self.learning_rate_min / self.learning_rate

                def cosine_factor(step):
                    return (
                        min_factor
                        + (1 - min_factor)
                        * (1 + math.cos(math.pi * step / decay_steps))
                        / 2
                    )

                cosine_scheduler = LambdaLR(self.optimizer, lr_lambda=cosine_factor)

                # Combine schedulers sequentially
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_steps],
                )
            else:
                raise ValueError(f"Scheduler {self.schedule} not recognized")

    def _mlm_loss_per_sample(self, x_pred, x_orig, mask_idx, pos_weights):
        """Weighted cross-entropy at BERT-masked positions, averaged per sample.

        Returns a (batch,) tensor.
        """
        # gather predictions and targets at BERT-masked positions only
        gather_idx = mask_idx.unsqueeze(1).expand(-1, 4, -1)  # (batch, 4, mask_size)
        pred_at_mask = x_pred.gather(2, gather_idx).permute(0, 2, 1)  # (batch, mask, 4)
        true_at_mask = x_orig.gather(2, gather_idx).permute(0, 2, 1)

        ce = -(true_at_mask * torch.log(pred_at_mask.clamp(min=1e-7))).sum(-1)
        if pos_weights is not None:
            ce = ce * pos_weights.gather(1, mask_idx)
        return ce.mean(-1)  # (batch,)

    def _mlm_prep(
        self,
        x: torch.Tensor,
        mask_size: int,
        exon_mask: torch.Tensor = None,
        repeat_mask: torch.Tensor = None,
        training: bool = False,
    ):
        """Prepare masked language model inputs.

        Masks mask_rate fraction of positions in each sample.

        Args:
            x: Input sequence tensor (batch, 4, seq_length).
            mask_size: Number of positions to mask.
            exon_mask: Optional exon mask tensor (batch, seq_length). Binary 0/1.
            repeat_mask: Optional repeat mask tensor (batch, seq_length). Binary 0/1.
            training: Whether in training mode (enables BERT-style masking).

        Returns:
            x_masked: Masked input tensor (batch, 4, seq_length). DNA channels are
                zeroed at BERT-masked positions; unmasked positions unchanged.
            x_orig: Original sequence tensor (batch, 4, seq_length).
            mask_idx: Indices of BERT-masked positions (batch, mask_size).
            pos_weights: Per-position loss weights (batch, seq_length) or None.

        Note:
            RC augmentation happens in SeqDatasetMLM.__getitem__, not here.
        """
        batch_size = x.shape[0]
        seq_length = x.shape[2]
        device = x.device

        # values < 1 downweight regions (e.g., exon_loss_scale=0.1 → 10% weight on exons)
        pos_weights = None
        if exon_mask is not None and self.exon_loss_scale is not None:
            pos_weights = (
                exon_mask * self.exon_loss_scale
                + (1 - exon_mask) * self.non_exon_loss_scale
            )
        if repeat_mask is not None and self.repeat_loss_scale is not None:
            repeat_weights = (
                repeat_mask * self.repeat_loss_scale
                + (1 - repeat_mask) * self.non_repeat_loss_scale
            )
            pos_weights = (
                repeat_weights if pos_weights is None else pos_weights * repeat_weights
            )

        mask_idx = torch.stack(
            [
                torch.randperm(seq_length, device=device)[:mask_size]
                for _ in range(batch_size)
            ]
        )  # (batch, mask_size)

        bert_mask = torch.zeros((batch_size, seq_length), device=device)
        bert_mask.scatter_(1, mask_idx, 1.0)
        bert_mask = bert_mask.unsqueeze(1)

        x_orig = x

        if self.use_bert and training:
            # per position: 10% keep original, 10% random nucleotide, 80% → all-zero
            random_nucs = torch.randint(0, 4, (batch_size, seq_length), device=device)
            x_random = (
                torch.nn.functional.one_hot(random_nucs, num_classes=4)
                .permute(0, 2, 1)
                .float()
            )

            sub_probs = torch.tensor(
                [0.1, 0.1, 0.8], device=device
            )  # keep, random, zero
            sub_type_idx = torch.multinomial(
                sub_probs.expand(batch_size, -1),
                seq_length,
                replacement=True,
            )  # (batch, seq_length)
            sub_type = (
                torch.nn.functional.one_hot(sub_type_idx, num_classes=3)
                .permute(0, 2, 1)
                .float()
            )
            # sub_type: (batch, 3, seq_length) — channels: [keep, random, zero]

            x_masked = (
                x * (1 - bert_mask)
                + x * bert_mask * sub_type[:, 0:1, :]  # keep original
                + x_random
                * bert_mask
                * sub_type[:, 1:2, :]  # random nucleotide (80% stays zero)
            )
        else:
            # zero out DNA channels at BERT-masked positions
            x_masked = x * (1 - bert_mask)

        return x_masked, x_orig, mask_idx, pos_weights

    def _mlm_species_id(self, label, context):
        """Return the batch's single species index, erroring on mixed species.

        MLM uses one trunk-norm index (di) per batch, so each data_dir must hold a
        single species.
        """
        species_id = int(label[0, 0].argmax())
        if not (label[:, 0].argmax(-1) == species_id).all():
            raise ValueError(
                f"MLM training requires single-species batches, but found mixed "
                f"species labels in {context}. Each data_dir must contain a single "
                f"species."
            )
        return species_id

    def _mlm_unpack(self, batch):
        """Pull sequence / species / masks off a BatchData and move to device."""
        x = batch.sequence.to(self.device)
        label = batch.species_label
        exon_mask = batch.exon_mask
        repeat_mask = batch.repeat_mask
        if exon_mask is not None:
            exon_mask = exon_mask.to(self.device)
        if repeat_mask is not None:
            repeat_mask = repeat_mask.to(self.device)
        return x, label, exon_mask, repeat_mask

    def _optimizer_step(self, loss):
        """Backward (outside autocast), gradient clip, optimizer + scheduler step."""
        if self.mix_dtype == torch.float16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.global_clipnorm
        )
        if self.mix_dtype == torch.float16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.schedule != "free":
            self.scheduler.step()

    def _partition_weight_decay_groups(self, named_params):
        """Partition model parameters into sets depending on weight decay rules.

        Args:
            named_params: collection of named model parameters.
        """

        # define weight decay parameter groups and compile regular expressions
        weight_decay_groups = [
            {
                "names": [],
                "params": [],
                "weight_decay": group["weight_decay"],
                "lr_mult": group.get("lr_mult", None),
                "regex": re.compile(group["regex"]),
            }
            for group in self.weight_decay_rules
        ] + [
            {
                "params": [],
                "names": [],
                "weight_decay": self.weight_decay,
                "lr_mult": None,
            }
        ]

        # loop over named parameters and partition weight decay groups
        for name, param in named_params:
            match_i = -1

            # loop over group rules
            for group_i, group in enumerate(weight_decay_groups[:-1]):
                if "regex" in group and re.search(group["regex"], name):
                    match_i = group_i
                    break

            # add parameter to matched group
            if match_i != -1:
                weight_decay_groups[match_i]["names"].append(name)
                weight_decay_groups[match_i]["params"].append(param)
            else:  # add to default group
                weight_decay_groups[-1]["names"].append(name)
                weight_decay_groups[-1]["params"].append(param)

        # finally remove regexes
        for group_i in range(len(weight_decay_groups)):
            if "regex" in weight_decay_groups[group_i]:
                del weight_decay_groups[group_i]["regex"]

        # print weight decay groups and warn about uncaught parameters
        print("")
        n_params_caught = 0

        # loop over groups
        for group_i in range(len(weight_decay_groups)):
            print("-- parameter group " + str(group_i) + " --")
            print(
                " => weight_decay = "
                + str(weight_decay_groups[group_i]["weight_decay"])
            )
            if weight_decay_groups[group_i].get("lr_mult") is not None:
                print(" => lr_mult = " + str(weight_decay_groups[group_i]["lr_mult"]))

            # print parameter names in group
            print("\n".join(weight_decay_groups[group_i]["names"]))
            if group_i < len(weight_decay_groups) - 1:
                print("")

            n_params_caught += len(weight_decay_groups[group_i]["params"])

        # check if all parameters were caught
        if n_params_caught != len(named_params):
            print("[Warning] Not all named parameters caught in weight decay groups.")

        return weight_decay_groups

    def _reset_metrics(self):
        """Reset the active task's metric accumulators between epochs."""
        if self.is_mlm:
            self.train_loss.reset()
            self.valid_loss.reset()
            return
        for di in range(self.num_datasets):
            self.train_metrics[di].reset()
            if len(self.eval_data) > 0:
                self.valid_metrics[di].reset()

    def _save_epoch(self, ei, early_stop_stat, valid_best, unimproved, log_out):
        """Select/save the best model, write periodic + resume checkpoints.

        Returns:
            (valid_best, unimproved)
        """
        # check overall best
        model_best_file = f"{self.out_dir}/model_best.pth"
        if early_stop_stat > valid_best or not os.path.exists(model_best_file):
            log_out.write(" - best!")
            unimproved = 0
            valid_best = early_stop_stat
            torch.save(self.model.state_dict(), model_best_file)
            torch.save(self.model.state_dict(), f"{self.out_dir}/model_best{ei}.pth")
            self.saved_best_files.append(f"model_best{ei}.pth")
            while len(self.saved_best_files) > self.save_best_max:
                pruned = self.saved_best_files.pop(0)
                os.remove(os.path.join(self.out_dir, pruned))
                if self.prune_callback is not None:
                    self.prune_callback(pruned)
        else:
            unimproved += 1
        log_out.write("\n")
        log_out.flush()

        # periodic check snapshots
        if ei >= self.save_check_after_epoch and (
            self.save_check_after_epoch2 is None or ei < self.save_check_after_epoch2
        ):
            # check less frequently in the beginning of training
            if (
                self.save_check_epoch_rate is not None
                and ei % self.save_check_epoch_rate == 0
            ):
                torch.save(
                    self.model.state_dict(), f"{self.out_dir}/model_check{ei}.pth"
                )
        elif (
            self.save_check_after_epoch2 is not None
            and ei >= self.save_check_after_epoch2
        ):
            # check more frequently later in training
            if (
                self.save_check_epoch_rate2 is not None
                and ei % self.save_check_epoch_rate2 == 0
            ):
                torch.save(
                    self.model.state_dict(), f"{self.out_dir}/model_check{ei}.pth"
                )

        # resume checkpoint
        if not self.loss_nan:
            # atomic: write tmp then replace, so a concurrent GCS sync never
            # sees a half-written file
            checkpoint_file = f"{self.out_dir}/checkpoint.pth"
            tmp_file = f"{checkpoint_file}.tmp"
            torch.save(
                {
                    "epoch": ei + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "valid_best": valid_best,
                    "unimproved": unimproved,
                },
                tmp_file,
            )
            os.replace(tmp_file, checkpoint_file)

            # tiny progress marker — lets an orchestrator read the current
            # epoch without downloading the full checkpoint
            with open(f"{self.out_dir}/progress.json", "w") as progress_out:
                json.dump(
                    {"epoch": ei + 1, "valid_best": float(valid_best)},
                    progress_out,
                )

            # per-epoch hook (e.g. sync out_dir to GCS)
            if self.checkpoint_callback is not None:
                self.checkpoint_callback()
        return valid_best, unimproved

    def _setup_optimization(self):
        """Init optimizer, scheduler, AMP scaler; compile ``self.model`` if enabled."""
        self._make_optimizer()
        self._make_scheduler()
        if self.mix_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # detect running-stat modules before compile wraps the module
        self._has_running_stats = _module_has_running_stats(self.model)

        if self.params.get("compile", False):
            self.model = torch.compile(self.model)

        # Eager handle for eval forwards: schedule-free optimizer.eval() weight swaps
        # and BatchNorm running-stat recompute don't propagate through the compiled
        # graph on some torch versions. Falls back to self.model when not compiled.
        self.model_eval = getattr(self.model, "_orig_mod", self.model)

    def _train_batch_cov(self, di_tensor, batch_data):
        """Coverage-supervised training step."""
        di = di_tensor[0].item()
        x, y, yg, gene_presence, gene_out_mask = self._unpack_batch(batch_data)
        sw = torch.tensor(self.species_weights[di]).to(
            device=self.device, dtype=torch.float32
        )
        hi = di if self.model_heads is None else self.model_heads

        with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
            yh = self.model(
                x,
                hi,
                di=self.model_di,
                gene_out_mask=gene_out_mask,
                gene_presence=gene_presence,
            )
            loss = self._compute_loss(yh, y, yg, gene_presence, sw, di)

        self.train_metrics[di].update(yh, y, yg, gene_presence, loss, x.shape[0])
        return loss

    def _train_batch_mlm(self, batch):
        """Masked-language-model training step."""
        x, label, exon_mask, repeat_mask = self._mlm_unpack(batch)
        species_id = self._mlm_species_id(label, "a batch")

        seq_length = x.shape[2]
        mask_size = max(1, int(self.mask_rate * seq_length))
        x_masked, x_orig, mask_idx, pos_weights = self._mlm_prep(
            x, mask_size, exon_mask=exon_mask, repeat_mask=repeat_mask, training=True
        )

        with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
            # hi=0: single shared MLM head; di=species_id: per-species trunk norm
            x_pred = self.model(x_masked, hi=0, di=species_id).coverage
            loss_per_sample = self._mlm_loss_per_sample(
                x_pred, x_orig, mask_idx, pos_weights
            )
            loss = loss_per_sample.mean()

        self.train_loss.update(loss_per_sample.sum().item(), loss_per_sample.shape[0])
        return loss

    def _train_epoch(self, ei):
        """Run one training epoch over ``train_dataload``."""
        train_batch_i = 0
        t0 = time.time()
        last_progress = t0
        for di_tensor, batch_data in tqdm(
            self.train_dataload,
            total=self.tqdm_total,
            desc="Training",
            disable=self.log_progress,
        ):
            self.optimizer.zero_grad()
            if self.is_mlm:
                loss = self._train_batch_mlm(batch_data)
            else:
                loss = self._train_batch_cov(di_tensor, batch_data)
            self._optimizer_step(loss)

            train_batch_i += 1
            if self.log_progress:
                last_progress = _emit_progress(
                    "train", ei, train_batch_i, self.tqdm_total, t0, last_progress
                )

            if (
                self.train_batches_max is not None
                and train_batch_i >= self.train_batches_max
            ):
                break
        torch.cuda.empty_cache()

    def _unpack_batch(self, batch_data) -> tuple:
        """Unpack batch data and move tensors to device.

        Handles both BatchData objects and legacy tuple format (for distillation).

        Returns:
            (x, y, yg, gene_presence, gene_out_mask)
        """
        # Handle legacy tuple format from SeqDatasetTeacher
        if isinstance(batch_data, tuple):
            x, y = batch_data
            x = x.to(self.device)
            y = y.to(self.device).float()
            return x, y, None, None, None

        # Handle BatchData format
        x = batch_data.sequence.to(self.device)
        y = batch_data.coverage_targets
        if y is not None:
            y = y.to(self.device).float()

        yg = gene_presence = gene_out_mask = None
        if batch_data.has_gene:
            yg = batch_data.gene_targets.to(self.device).float()
            gene_presence = batch_data.gene_presence.to(self.device)
            gene_out_mask = batch_data.gene_out_mask.to(self.device)

        return x, y, yg, gene_presence, gene_out_mask

    def _validate_cov(self, ei):
        """Coverage validation over (a subset of) the eval datasets.

        Returns:
            n_eval_datasets: how many datasets were validated this epoch (passed
            on to ``_log_epoch_cov``).
        """
        # most epochs validate only the early-stop datasets; periodically all
        n_eval_datasets = self.early_stop_datasets
        if ei % self.full_validation_epoch_rate == 0:
            n_eval_datasets = len(self.eval_dataload)

        for di in range(n_eval_datasets):
            hi = di if self.model_heads is None else self.model_heads
            val_t0 = time.time()
            last_progress = val_t0
            val_batch_i = 0
            val_total = len(self.eval_dataload[di])
            for batch_data in tqdm(
                self.eval_dataload[di], desc="Validation", disable=self.log_progress
            ):
                x, y, yg, gene_presence, gene_out_mask = self._unpack_batch(batch_data)
                sw = torch.tensor(self.species_weights[di]).to(
                    device=self.device, dtype=torch.float32
                )
                with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                    yh = self.model_eval(
                        x,
                        hi,
                        di=self.model_di,
                        gene_out_mask=gene_out_mask,
                        gene_presence=gene_presence,
                    )
                    loss = self._compute_loss(yh, y, yg, gene_presence, sw, di)
                self.valid_metrics[di].update(
                    yh, y, yg, gene_presence, loss, x.shape[0]
                )

                val_batch_i += 1
                if self.log_progress:
                    last_progress = _emit_progress(
                        f"valid[{di}]",
                        ei,
                        val_batch_i,
                        val_total,
                        val_t0,
                        last_progress,
                    )
            torch.cuda.empty_cache()
        return n_eval_datasets

    def _validate_mlm(self):
        """MLM validation over all eval datasets, averaging ``repeat_eval`` masks."""
        for dataset_i in range(len(self.eval_dataload)):
            for batch in tqdm(self.eval_dataload[dataset_i], desc="Validation"):
                x, label, exon_mask, repeat_mask = self._mlm_unpack(batch)
                species_id = self._mlm_species_id(label, "a validation batch")

                seq_length = x.shape[2]
                mask_size = max(1, int(self.mask_rate * seq_length))

                # average the loss over repeat_eval independent random masks
                eval_losses = []
                for _ in range(self.repeat_eval):
                    x_masked, x_orig, mask_idx, pos_weights = self._mlm_prep(
                        x,
                        mask_size,
                        exon_mask=exon_mask,
                        repeat_mask=repeat_mask,
                        training=False,
                    )
                    with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                        x_pred = self.model_eval(x_masked, hi=0, di=species_id).coverage
                        eval_losses.append(
                            self._mlm_loss_per_sample(
                                x_pred, x_orig, mask_idx, pos_weights
                            )
                        )

                avg_per_sample = torch.stack(eval_losses, dim=0).mean(dim=0)  # (batch,)
                self.valid_loss.update(
                    avg_per_sample.sum().item(), avg_per_sample.shape[0]
                )
            torch.cuda.empty_cache()


def _module_has_running_stats(module) -> bool:
    """True if any submodule caches running statistics (BatchNorm-style).

    These caches are recorded at the schedule-free training iterate and go stale
    at the eval-point weights, so they need a warmup recompute before validation
    (see Trainer._eval_prep). LayerNorm/RMSNorm/Identity have no such buffers.
    """
    return any(
        getattr(m, "track_running_stats", False)
        and getattr(m, "running_mean", None) is not None
        for m in module.modules()
    )


def parse_loss(
    loss_label,
    spec_weight: float = 1,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 1,
):
    """Parse loss function from label, strategy, and fitting method.

    Args:
        loss_label (str): Loss function label.
        spec_weight (float): Specificity weight for PoissonKL.
        total_weight (float): Total weight for PoissionMultinomial.
        weight_range (float): Weight range for PoissionMultinomial.
        weight_exp (int): Weight exponent for PoissionMultinomial.

    Returns:
      loss_fn: Torch loss
    """
    if loss_label == "poisson":
        loss_fn = metrics.PoissonLoss()
    elif loss_label == "poisson_mn":
        loss_fn = metrics.PoissonMultinomialLoss(
            total_weight=total_weight, weight_range=weight_range, weight_exp=weight_exp
        )
    else:
        raise ValueError(f"Loss function {loss_label} not recognized")

    return loss_fn


def parse_target_weights(targets_df, rules):
    """Resolve per-track loss weights from the targets table and params rules.

    The targets table's ``weight`` column states what the dataset considers a
    track worth; it travels with the data and is the same in every run. Each
    rule selects tracks by equality on other targets columns and multiplies
    their weight, expressing a single run's deviation from that baseline:

        [{"group": "H3K9me3", "weight": 0.1}, {"assay": "chip", "weight": 0.2}]

    The first matching rule wins, so list specific rules before general ones.

    Args:
        targets_df: targets table DataFrame.
        rules: list of dicts, each a set of column equalities plus "weight".

    Returns:
        np.ndarray of float weights, length ``len(targets_df)``.
    """
    if "weight" in targets_df.columns:
        weights = targets_df["weight"].to_numpy(dtype=np.float64, copy=True)
    else:
        weights = np.ones(targets_df.shape[0])

    matched = np.zeros(targets_df.shape[0], dtype=bool)
    for rule in rules:
        select = {k: v for k, v in rule.items() if k != "weight"}
        if not select or "weight" not in rule:
            raise ValueError(f"loss_weight rule needs a weight and a selector: {rule}")

        missing = [c for c in select if c not in targets_df.columns]
        if missing:
            raise ValueError(
                f"loss_weight rule columns {missing} not in targets: {rule}"
            )

        select_match = np.ones(targets_df.shape[0], dtype=bool)
        for column, value in select.items():
            select_match &= (targets_df[column] == value).to_numpy()

        # warn on the selector alone: a rule shadowed by an earlier one is the
        # documented specific-before-general idiom, not a typo
        if not select_match.any():
            print(f"[Warning] loss_weight rule matched no targets: {rule}")

        rule_match = select_match & ~matched
        weights[rule_match] *= rule["weight"]
        matched |= rule_match

    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Loss weights must be finite and nonnegative")
    if not (weights > 0).any():
        raise ValueError("Loss weights must have a positive total")

    return weights
