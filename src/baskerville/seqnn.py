import os

from natsort import natsorted
import numpy as np
import pdb
import torch
import torch.nn as nn
from tqdm import tqdm
import zarr

from baskerville import blocks
from baskerville.dataset import get_untransform_func
from baskerville import dna
from baskerville import metrics
from baskerville.types import BatchData, ModelOutput


def _float_output(out: ModelOutput, keep_gradients: bool) -> ModelOutput:
    """Cast predictions to float32 (autocast may leave them reduced), detaching unless kept."""

    def cast(x):
        if x is None:
            return None
        x = x.float()
        return x if keep_gradients else x.detach()

    return ModelOutput(coverage=cast(out.coverage), gene=cast(out.gene))


class SeqNN:
    """Sequence neural network model.

    Args:
      params (dict): Model specification and parameters.
    """

    def __init__(self, params: dict, output_slice=None):
        # defaults
        self.verbose = True
        self.ensemble_rc = False
        self.ensemble_shifts = [0]
        self.strand_pair = None
        self.seq_length = params.get("seq_length", None)
        self.mix_dtype = torch.float32
        self.output_slice = output_slice

        # set params
        for key, value in params.items():
            self.__setattr__(key, value)

        self.untransform_fn = get_untransform_func(params)

        # build model
        self.build_model()
        self.set_device()

        # seed w/ pretrained weights
        if hasattr(self, "pretrained_model"):
            strict = True
            if hasattr(self, "pretrained_model_strict"):
                strict = self.pretrained_model_strict

            trunk_only = True
            if hasattr(self, "pretrained_model_trunk_only"):
                trunk_only = self.pretrained_model_trunk_only

            self.restore(self.pretrained_model, trunk_only=trunk_only, strict=strict)

    def build_model(self, save_reprs: bool = True):
        # Support new architecture (heads_cov, heads_gene) and legacy (head0, head1, ...)
        heads_cov = None
        heads_gene = None

        if hasattr(self, "heads_cov"):
            heads_cov = self.heads_cov
        if hasattr(self, "heads_gene"):
            heads_gene = self.heads_gene

        # Legacy support: convert old head0, head1, ... format
        if heads_cov is None and heads_gene is None:
            head_keys = natsorted([v for v in vars(self) if v.startswith("head")])
            if head_keys:
                # Old format: flat list of heads
                legacy_heads = [getattr(self, hk) for hk in head_keys]
                # For backward compatibility, treat all as coverage heads
                heads_cov = legacy_heads
                heads_gene = None

        global_names = [
            "act_func",
            "norm_type",
            "num_species",
        ]
        global_vars = {}
        for gv in global_names:
            if hasattr(self, gv):
                global_vars[gv] = getattr(self, gv)

        self.model = SeqNNMod(
            self.trunk,
            heads_cov,
            heads_gene,
            self.output_slice,
            global_vars,
            self.seq_length,
        )

        if self.verbose:
            print(self.model)
            param_count = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"Model parameters: {param_count}")

    def __call__(
        self,
        x,
        hi=0,
        keep_gradients=False,
        gene_out_mask=None,
        gene_presence=None,
        di=None,
    ) -> ModelOutput:
        """Predict on data, assuming model is set.

        Args:
            x: Input data.
            hi: Head index or indices. Can be:
                - int: Use single head (e.g., hi=0)
                - -1: Use all heads concatenated
                - list[int]: Use specified heads concatenated (e.g., hi=[0, 2])
            keep_gradients: If True, preserve gradients (don't detach).
            gene_out_mask: Optional boolean bin mask (B, max_genes, target_bins)
            gene_presence: Optional gene mask (B, max_genes)

        Returns:
            ModelOutput with coverage and/or gene predictions.
        """
        if self.ensemble_rc:
            strand_pair_hi = self._make_strand_pair(hi)

        outputs = []
        for shift in self.ensemble_shifts:
            # shift
            xs = dna.torch_shift(x, shift)

            # forward
            eff_di = di if di is not None else getattr(self, "model_di", None)
            with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                out = self.model(
                    xs,
                    hi,
                    di=eff_di,
                    gene_out_mask=gene_out_mask,
                    gene_presence=gene_presence,
                )
            outputs.append(_float_output(out, keep_gradients))

            # reverse complement
            if self.ensemble_rc:
                xsr = dna.torch_rc(xs)

                # Flip bin mask for RC coordinates
                rc_gene_out_mask = gene_out_mask
                if gene_out_mask is not None:
                    rc_gene_out_mask = torch.flip(gene_out_mask, [-1])

                with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                    out_rc = self.model(
                        xsr,
                        hi,
                        di=eff_di,
                        gene_out_mask=rc_gene_out_mask,
                        gene_presence=gene_presence,
                    )
                out_rc = _float_output(out_rc, keep_gradients)

                # Process coverage: flip and apply strand pairing
                coverage_rc = None
                if out_rc.has_coverage:
                    coverage_rc = torch.flip(out_rc.coverage, [2])
                    if strand_pair_hi is not None:
                        coverage_rc = coverage_rc[:, strand_pair_hi, :]

                # Gene predictions: slices were transformed above, no post-hoc flipping needed
                gene_rc = out_rc.gene

                outputs.append(ModelOutput(coverage=coverage_rc, gene=gene_rc))

        # average ensemble predictions
        if len(outputs) == 1:
            return outputs[0]

        coverage_avg = None
        gene_avg = None

        if outputs[0].has_coverage:
            coverage_avg = torch.stack([o.coverage for o in outputs]).mean(0)
        if outputs[0].has_gene:
            gene_avg = torch.stack([o.gene for o in outputs]).mean(0)

        return ModelOutput(coverage=coverage_avg, gene=gene_avg)

    def compile(self):
        """Compile model for faster inference; call after restore.

        Default mode: reduce-overhead's CUDA graphs overwrite outputs that
        callers hold across calls (e.g. ref predictions).
        """
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] < 7:
            print("Warning: CUDA device capability < 7.0, skipping compilation.")
        else:
            self.model = torch.compile(self.model)

    def eval(
        self,
        data,
        hi=0,
        batch_size=1,
        return_values=False,
        step=1,
        zarr_store: str | None = None,
        target_chunk: int | None = 128,
        seq_chunk: int | None = None,
    ):
        """Evaluate model on data.

        Args:
            data: SeqDataset to evaluate.
            hi: Head index.
            batch_size: Batch size.
            return_values (bool): Return values.
            step (int): Step.
            zarr_store: Optional zarr store path for predictions.
            target_chunk: zarr chunk size along the target axis
            seq_chunk: zarr chunk size along the seq axis for the coverage store

        Returns:
            dict with keys:
                - 'coverage': dict with 'r', 'r2', 'spec', 'preds', 'targets' (if
                  coverage head); 'spec' is per-track specificity Pearson (NaN
                  outside scored groups), or None without data.target_hist
                - 'gene': dict with 'r', 'r2', 'preds', 'targets', 'masks' (if gene head)
        """
        self.model.eval()

        if target_chunk is not None and target_chunk < 1:
            raise ValueError(f"target_chunk must be >= 1, got {target_chunk}")
        if seq_chunk is not None and seq_chunk < 1:
            raise ValueError(f"seq_chunk must be >= 1, got {seq_chunk}")

        # detect what heads/data we have
        has_coverage = data.has_coverage and self.model.heads_cov is not None
        has_gene = (
            hasattr(data, "has_genes")
            and data.has_genes
            and self.model.heads_gene is not None
        )

        # prepare data
        data_load = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=4,
            drop_last=False,
            pin_memory=self.device != "cpu",
            collate_fn=BatchData.collate,
        )
        # Fork DataLoader workers before any zarr.open() calls below.
        # zarr.open() spawns asyncio threads that deadlock forked workers.
        data_load_iter = iter(data_load)

        # prepare coverage metrics
        eval_metric_r = eval_metric_r2 = None
        if has_coverage:
            if self.output_slice is None:
                num_targets = data.num_targets
            else:
                num_targets = len(self.output_slice)
            eval_metric_r = metrics.PearsonCorrCoef(num_targets, average=False)
            eval_metric_r.to(self.device)
            eval_metric_r2 = metrics.R2Score(average=False)
            eval_metric_r2.to(self.device)

            # specificity, given the dataset's target histograms
            eval_metric_spec = None
            target_hist = getattr(data, "target_hist", None)
            if target_hist is not None:
                targets_df = data.targets_df
                if self.output_slice is not None:
                    targets_df = targets_df.loc[self.output_slice]
                    target_hist = target_hist[self.output_slice]
                # metadata must describe every scored track
                if len(targets_df) == num_targets:
                    eval_metric_spec = metrics.SpecPearsonCorrCoef(
                        targets_df, target_hist
                    )
                    eval_metric_spec.to(self.device)

        # prepare gene metrics
        eval_metric_r_gene = eval_metric_r2_gene = None
        if has_gene:
            num_gene_targets = data.num_gene_targets
            eval_metric_r_gene = metrics.GenePearsonCorrCoef(
                num_gene_targets, average=False
            )
            eval_metric_r_gene.to(self.device)
            eval_metric_r2_gene = metrics.GeneR2Score(average=False)
            eval_metric_r2_gene.to(self.device)

        # initialize preds/targets storage
        preds = targets = gene_preds = gene_targets = gene_presence = None
        if return_values:
            num_seqs = len(data)
            compressors = zarr.codecs.BloscCodec(cname="zstd", clevel=1)

            if zarr_store is not None:
                os.makedirs(zarr_store, exist_ok=True)
                root = zarr.open_group(zarr_store, mode="w")
            else:
                root = None

            def _create_array(name, shape, chunks, dtype):
                if root is not None:
                    return root.create_array(
                        name,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        compressors=compressors,
                    )
                return np.empty(shape, dtype=dtype)

            # on-disk coverage seq chunk (None -> batch_size), clamped to num_seqs
            cov_seq_chunk = (
                min(seq_chunk, num_seqs) if (root is not None and seq_chunk) else None
            )

            # coverage storage
            if has_coverage:
                if data.target_length % step != 0:
                    raise ValueError(
                        f"step ({step}) must evenly divide target_length ({data.target_length})"
                    )
                targets_length = data.target_length // step
                vshape = (num_seqs, num_targets, targets_length)
                tchunk = (
                    num_targets
                    if target_chunk is None
                    else min(target_chunk, num_targets)
                )
                cshape = (cov_seq_chunk or batch_size, tchunk, targets_length)
                preds = _create_array("preds", vshape, cshape, "float16")
                targets = _create_array("targets", vshape, cshape, "float16")
                w_preds = _RowWriter(preds, cov_seq_chunk)
                w_targets = _RowWriter(targets, cov_seq_chunk)

            # gene storage
            if has_gene:
                if data.zarr_data is None:
                    data._open_zarr()
                max_genes = data.zarr_data[0]["gene_presence"].shape[1]
                vshape = (num_seqs, num_gene_targets, max_genes)
                cshape = (batch_size, num_gene_targets, max_genes)
                gene_preds = _create_array("gene_preds", vshape, cshape, "float16")
                gene_targets = _create_array("gene_targets", vshape, cshape, "float16")
                gene_presence = _create_array(
                    "gene_presence",
                    (num_seqs, max_genes),
                    (batch_size, max_genes),
                    "bool",
                )

        si = 0
        with torch.no_grad():
            for batch_data in tqdm(data_load_iter, desc="Test", total=len(data_load)):
                x = batch_data.sequence
                sb = si + x.shape[0]
                x = x.to(self.device).float()

                # extract coverage targets
                y = None
                if has_coverage and batch_data.coverage_targets is not None:
                    y = batch_data.coverage_targets
                    if self.output_slice is not None:
                        y = y[:, self.output_slice, :]
                    if return_values:
                        y_np = y.detach().numpy()
                        if step != 1:
                            y_np = y_np[:, :, ::step]
                        w_targets.add(si, y_np)
                    y = y.to(self.device).float()

                # extract gene data
                yg = batch_gene_presence = batch_gene_out_mask = None
                if has_gene and batch_data.has_gene:
                    yg = batch_data.gene_targets.to(self.device).float()
                    batch_gene_presence = batch_data.gene_presence.to(self.device)
                    batch_gene_out_mask = batch_data.gene_out_mask.to(self.device)
                    if return_values:
                        gene_targets[si:sb] = batch_data.gene_targets.numpy()
                        gene_presence[si:sb] = batch_data.gene_presence.numpy()

                # predict
                with torch.autocast(device_type=self.device, dtype=self.mix_dtype):
                    yh = self(
                        x,
                        hi,
                        gene_out_mask=batch_gene_out_mask,
                        gene_presence=batch_gene_presence,
                    )

                # update coverage metrics
                if has_coverage and y is not None:
                    eval_metric_r.update(yh.coverage, y)
                    eval_metric_r2.update(yh.coverage, y)
                    if eval_metric_spec is not None:
                        eval_metric_spec.update(yh.coverage, y)
                    if return_values:
                        yhs = yh.coverage if step == 1 else yh.coverage[:, :, ::step]
                        w_preds.add(si, yhs.to(torch.float16).detach().cpu().numpy())

                # update gene metrics
                if has_gene and yg is not None and yh.has_gene:
                    eval_metric_r_gene.update(yh.gene, yg, batch_gene_presence)
                    eval_metric_r2_gene.update(yh.gene, yg, batch_gene_presence)
                    if return_values:
                        gene_preds[si:sb] = (
                            yh.gene.to(torch.float16).detach().cpu().numpy()
                        )

                si = sb

        # flush any buffered coverage rows to the store
        if return_values and has_coverage:
            w_preds.flush()
            w_targets.flush()

        # finalize and return results
        results = {}

        if has_coverage:
            eval_r = eval_metric_r.compute().cpu().detach().numpy()
            eval_r2 = eval_metric_r2.compute().cpu().detach().numpy()
            eval_spec = None
            if eval_metric_spec is not None:
                eval_spec = eval_metric_spec.compute().numpy()
            results["coverage"] = {
                "r": eval_r,
                "r2": eval_r2,
                "spec": eval_spec,
                "preds": preds,
                "targets": targets,
            }

        if has_gene:
            eval_r_gene = eval_metric_r_gene.compute().cpu().detach().numpy()
            eval_r2_gene = eval_metric_r2_gene.compute().cpu().detach().numpy()
            results["gene"] = {
                "r": eval_r_gene,
                "r2": eval_r2_gene,
                "preds": gene_preds,
                "targets": gene_targets,
                "masks": gene_presence,
            }

        return results

    def gradients(
        self,
        x,
        hi=0,
        spatial_slice=None,
        task_slice=None,
        untransform_targets_df=None,
        log_transform=False,
        agg_fn=None,
        agg_fn_kwargs=None,
    ):
        """Compute gradients of model predictions with respect to input sequence.

        This method computes the gradient of aggregated predictions, which can be used
        for sequence interpretation and attribution analysis. Supports both default
        slice+sum-based aggregation and custom aggregation functions.

        Args:
            x: Input sequence tensor of shape (channels, seq_length).
            hi: Head index to use for predictions.
            spatial_slice: Boolean or integer slice for spatial axis (length L).
                          If None, defaults to all True (full spatial axis).
                          Ignored if agg_fn is provided.
            task_slice: Boolean or integer slice for task axis (length T).
                       If None, defaults to all True (full task axis).
                       Ignored if agg_fn is provided.
            untransform_targets_df: DataFrame with target transformation information.
                                  If provided, applies inverse transformations to predictions before aggregation.
            log_transform (bool): If True, applies log transformation to the aggregated predictions before computing gradients.
                                Ignored if agg_fn is provided.
            agg_fn: Custom aggregation function that takes yh tensor and returns a scalar.
                   If provided, spatial_slice, task_slice, and log_transform are ignored.
            agg_fn_kwargs: Dictionary of keyword arguments to pass to agg_fn.

        Returns:
            gradients: Gradients tensor of same shape as input x, containing
                      gradients of aggregated predictions with respect to input sequence.
        """
        self.model.eval()

        # Verify slices are on device
        if spatial_slice is not None and isinstance(spatial_slice, torch.Tensor):
            spatial_slice = spatial_slice.to(self.device)
        if task_slice is not None and isinstance(task_slice, torch.Tensor):
            task_slice = task_slice.to(self.device)

        # Add batch dimension and ensure input requires gradients
        xb = x.unsqueeze(0).to(self.device)
        if not xb.requires_grad:
            xb.requires_grad_(True)

        # Compute predictions with gradients preserved
        output = self(xb, hi, keep_gradients=True)

        # Extract coverage predictions and remove batch dimension
        yh = output.coverage.squeeze(0)

        # Apply inverse transformations if requested
        if untransform_targets_df is not None:
            yh = self.untransform_fn(yh, untransform_targets_df)

        if agg_fn is not None:
            # Custom aggregation function
            if agg_fn_kwargs is None:
                agg_fn_kwargs = {}
            prediction_agg = agg_fn(yh, **agg_fn_kwargs)
        else:
            # Sum-based aggregation with spatial and task slicing

            # Slice bins
            if spatial_slice is not None:
                yh = yh[:, spatial_slice]

            # Slice tasks
            if task_slice is not None:
                yh = yh[task_slice, :]

            # Aggregate across spatial and task dimensions
            prediction_agg = yh.sum(dim=1).mean(dim=0)

            # Log transform, if specified
            if log_transform:
                prediction_agg = torch.log(prediction_agg + 1e-6)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=prediction_agg,
            inputs=xb,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        # Remove batch dimension from gradients to match input shape
        return gradients.squeeze(0)

    def output_crop_bp(self):
        """Return crop length."""
        return self.model.output_crop_bp

    def output_depth(self, hi=0, head_type="cov"):
        """Return output channel depth.

        Args:
            hi: Head index or indices (species index). Can be:
                - int: Return channels for single species (e.g., hi=0)
                - -1: Return sum of channels across all species
                - list[int]: Return sum of channels for specified species
            head_type: "cov" for coverage heads or "gene" for gene heads

        Returns:
            int: Number of output channels.
        """
        # Get the appropriate heads definition (not the module)
        heads_def = (
            self.model.heads_cov_def
            if head_type == "cov"
            else self.model.heads_gene_def
        )

        if heads_def is None:
            return 0

        # Determine which species to query
        if isinstance(hi, list):
            species_indices = hi
        elif hi == -1:
            species_indices = list(range(len(heads_def)))
        else:
            species_indices = [hi]

        # Sum output channels from selected species
        total = 0
        for species_i in species_indices:
            if species_i >= len(heads_def) or heads_def[species_i] is None:
                continue

            head_def = heads_def[species_i]
            blocks = head_def if isinstance(head_def, list) else [head_def]
            for block in reversed(blocks):
                depth = self._block_output_depth(block)
                if depth > 0:
                    total += depth
                    break

        return total

    @staticmethod
    def _block_output_depth(block_def):
        """Get output channel count from a block definition."""
        for key in ("num_targets", "num_gene_targets", "out_channels"):
            if key in block_def:
                return block_def[key]
        return 0

    def output_length(self):
        """Return output length."""
        dna_length = self.seq_length - 2 * self.model.output_crop_bp
        return dna_length // self.model.output_stride

    def output_stride(self):
        """Return output stride."""
        return self.model.output_stride

    def restore(self, model_path: str, trunk_only: bool = False, strict=True):
        """Restore model from file.

        Args:
        model_path (str): Path to model file.
        trunk_only (bool): If True, only load trunk parameters (ignore heads).
        strict (bool): If true, fail on any weight/layer mismatch.
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # check compilation
        is_compiled = any("_orig_mod." in k for k in checkpoint.keys())

        if trunk_only:
            # skip head parameters
            if is_compiled:
                clean_state_dict = {
                    k.replace("_orig_mod.", ""): v
                    for k, v in checkpoint.items()
                    if not k.replace("_orig_mod.", "").startswith("heads")
                }
            else:
                clean_state_dict = {
                    k: v for k, v in checkpoint.items() if not k.startswith("heads")
                }

            # load the filtered state dict with strict=False
            self.model.load_state_dict(clean_state_dict, strict=False)
            print(f"Model trunk restored from {model_path}")
        else:
            # load full model
            if is_compiled:
                clean_state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()
                }
            else:
                clean_state_dict = checkpoint

            # Backward compatibility: translate old "heads." keys to new format
            # Check if checkpoint uses old format (has "heads." keys) and model uses new format
            has_old_heads = any(k.startswith("heads.") for k in clean_state_dict.keys())
            has_new_heads_cov = any(
                k.startswith("heads_cov.") for k in clean_state_dict.keys()
            )
            has_new_heads_gene = any(
                k.startswith("heads_gene.") for k in clean_state_dict.keys()
            )

            # If checkpoint has old format and current model has new format, translate
            if has_old_heads and not (has_new_heads_cov or has_new_heads_gene):
                translated_state_dict = {}
                for k, v in clean_state_dict.items():
                    if k.startswith("heads."):
                        # Translate old "heads." to "heads_cov." (assume coverage heads)
                        new_key = k.replace("heads.", "heads_cov.", 1)
                        translated_state_dict[new_key] = v
                    else:
                        translated_state_dict[k] = v
                clean_state_dict = translated_state_dict

            self.model.load_state_dict(clean_state_dict, strict=strict)
            print(f"Model restored from {model_path}")

    def set_device(self, device=None):
        """Set device.

        Args:
          device (str): Device.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model.to(device)

    def _make_strand_pair(self, hi):
        """Create strand_pair array for given head index or indices.

        Args:
            hi: Head index or indices. Can be:
                - int: Use single head (e.g., hi=0)
                - -1: Use all heads concatenated
                - list[int]: Use specified heads concatenated (e.g., hi=[0, 2])

        Returns:
            strand_pair array for the specified head(s), or None if no strand pairing.
        """
        if self.strand_pair is None:
            return None

        elif not isinstance(self.strand_pair, list):
            # Single strand_pair array for all heads (backward compatibility)
            return self.strand_pair

        else:
            # Handle list of strand_pairs (one per head)

            # Determine which heads to use
            if isinstance(hi, list):
                head_indices = hi
            elif hi == -1:
                head_indices = list(range(len(self.strand_pair)))
            else:
                head_indices = [hi]

            # Collect strand_pairs for selected heads
            strand_pairs_selected = [self.strand_pair[idx] for idx in head_indices]

            # If single head, return its strand_pair directly
            if len(strand_pairs_selected) == 1:
                return strand_pairs_selected[0]
            else:
                # For multiple heads, concatenate with proper offsets
                concatenated = []
                offset = 0
                for i, sp in enumerate(strand_pairs_selected):
                    if sp is None:
                        raise ValueError(
                            f"Cannot use reverse complement ensemble with multiple heads when "
                            f"head {head_indices[i]} has no strand_pair information. "
                            f"Either provide strand_pair for all heads or disable reverse complement ensemble (ensemble_rc=False)."
                        )
                    concatenated.append(sp + offset)
                    offset += len(sp)

                return np.concatenate(concatenated) if concatenated else None


class SeqNNMod(nn.Module):
    def __init__(
        self,
        trunk_def,
        heads_cov_def=None,
        heads_gene_def=None,
        output_slice=None,
        global_vars={},
        seq_length=None,
    ):
        super(SeqNNMod, self).__init__()
        self.trunk_def = trunk_def
        self.heads_cov_def = heads_cov_def
        self.heads_gene_def = heads_gene_def
        self.output_stride = 1
        self.output_crop_bp = 0
        # index tensor (not e.g. a pandas Index) so torch.compile can trace the slice
        if output_slice is not None:
            output_slice = torch.as_tensor(np.asarray(output_slice), dtype=torch.long)
        self.register_buffer("output_slice", output_slice, persistent=False)
        global_vars = global_vars or {}
        self.global_vars = global_vars
        self.seq_length = seq_length

        # model trunk
        self.trunk = torch.nn.ModuleList()
        for block_params in trunk_def:
            self.trunk.append(self.build_block(block_params, global_vars))

        # Build heads
        self.heads_cov = self._build_heads(heads_cov_def, global_vars)
        self.heads_gene = self._build_heads(heads_gene_def, global_vars)

    def _build_heads(self, heads_def, global_vars):
        """Build a ModuleList of head modules from definitions.

        Args:
            heads_def: List of head definitions (one per species), or None.
            global_vars: Global variables for block construction.

        Returns:
            ModuleList of head modules, or None if heads_def is None.
        """
        if heads_def is None:
            return None
        heads = torch.nn.ModuleList()
        for head_def in heads_def:
            if head_def is None:
                heads.append(None)
            else:
                if not isinstance(head_def, list):
                    head_def = [head_def]
                head_blocks = [self.build_block(bp, global_vars) for bp in head_def]
                heads.append(torch.nn.Sequential(*head_blocks))
        return heads

    def build_block(self, block_params, global_vars=None):
        """Construct a SeqNN block.

        Args:
            block_params (dict): Block parameters.
        Returns:
            block_module: Block module.
        """
        global_vars = global_vars or {}
        # switch for block
        block_name = block_params["name"]
        if block_name in blocks.torch_module:
            block_module = blocks.torch_module[block_name]
            block_signature = ""
        else:
            block_module = blocks.name_module[block_name]
            block_signature = str(blocks.name_init[block_name])

            # add signature for internal block used by tower
            if block_name[-5:] == "Tower":
                child_name = block_name[:-5] + "Block"
                if child_name in blocks.name_module:
                    block_signature += " and " + str(blocks.name_init[child_name])
                elif (
                    block_name[:-5] + "EnformerBlock" in blocks.name_module
                ):  # special case for enformer block
                    block_signature += " and " + str(
                        blocks.name_init[block_name[:-5] + "EnformerBlock"]
                    )

        block_args = {}

        # set global defaults
        for gv in global_vars:
            gv_value = global_vars[gv]
            if gv not in block_params and gv in block_signature:
                block_args[gv] = gv_value

        # set remaining params
        block_args.update(block_params)
        del block_args["name"]
        # Remove species key if present (used for organization, not block construction)
        if "species" in block_args:
            del block_args["species"]

        # auto-inject seq_len (current sequence length at this stage)
        if (
            self.seq_length is not None
            and "seq_len" in block_signature
            and "seq_len" not in block_args
        ):
            block_args["seq_len"] = (
                self.seq_length - 2 * self.output_crop_bp
            ) // self.output_stride

        # track output stride
        if "pool_size" in block_args:
            block_pool = block_args["pool_size"] ** block_args.get("repeat", 1)
            self.output_stride *= block_pool
        elif "stride" in block_args:
            block_stride = block_args["stride"] ** block_args.get("repeat", 1)
            self.output_stride *= block_stride
        elif "Unet" in block_name:
            block_upsample = 2 ** block_args.get("repeat", 1)
            self.output_stride //= block_upsample

        # track output crop
        if "crop_size" in block_args:
            self.output_crop_bp += self.output_stride * block_args["crop_size"]

        return block_module(**block_args)

    def forward(
        self, x, hi=0, di=None, gene_out_mask=None, gene_presence=None
    ) -> ModelOutput:
        """Forward pass with species-specific heads.

        Args:
            x: Input tensor.
            hi: Head/species index for selecting output heads. Can be:
                - int: Use single species (e.g., hi=0 uses heads_cov[0] and heads_gene[0])
                - -1: Use all species concatenated
                - list[int]: Use specified species concatenated (e.g., hi=[0, 2])
            di: Species normalization index for trunk blocks with SPECIES_ARG_FLAG.
                Overrides hi for trunk normalization. Useful when the head index (hi)
                differs from the species used for batch norm (e.g., MLM with a shared
                head but species-specific normalization). Defaults to species_indices[0].
            gene_out_mask: Optional boolean bin mask (B, max_genes, target_bins)
            gene_presence: Optional gene mask (B, max_genes)

        Returns:
            ModelOutput with coverage and/or gene predictions (None if head not present).
        """
        # Determine which species to use
        if isinstance(hi, list):
            species_indices = hi
        elif hi == -1:
            # Use all species that have at least one head
            max_species = 0
            if self.heads_cov is not None:
                max_species = max(max_species, len(self.heads_cov))
            if self.heads_gene is not None:
                max_species = max(max_species, len(self.heads_gene))
            species_indices = list(range(max_species))
        else:
            species_indices = [hi]

        # Species normalization index for trunk blocks
        norm_idx = di if di is not None else species_indices[0]

        # Process trunk
        conv_reprs = []
        ui = 1
        for bi, block in enumerate(self.trunk):
            block_name = self.trunk_def[bi]["name"]
            block_flags = blocks.name_flag[block_name]

            if (
                blocks.SPECIES_ARG_FLAG in block_flags
                and blocks.CONV_RET_FLAG in block_flags
            ):
                # conv tower
                x, crs = block(x, norm_idx)
                conv_reprs += crs
            elif (
                blocks.SPECIES_ARG_FLAG in block_flags
                and blocks.CONV_REP_FLAG in block_flags
            ):
                # unet block
                x = block(x, conv_reprs[-ui], norm_idx)
                ui += 1
            elif (
                blocks.SPECIES_ARG_FLAG in block_flags
                and blocks.CONV_REPS_FLAG in block_flags
            ):
                # unet tower
                x = block(x, conv_reprs, norm_idx)
            elif blocks.SPECIES_ARG_FLAG in block_flags:
                # species arg
                x = block(x, norm_idx)
            else:
                x = block(x)

        # Process heads for each species
        coverage_preds = []
        gene_preds = []

        for species_i in species_indices:
            # Process coverage head if exists for this species
            if self.heads_cov is not None and species_i < len(self.heads_cov):
                head = self.heads_cov[species_i]
                if head is not None:
                    y_cov = head(x)
                    if self.output_slice is not None:
                        y_cov = y_cov[:, self.output_slice, :]
                    coverage_preds.append(y_cov)

            # Process gene head if exists for this species
            if self.heads_gene is not None and species_i < len(self.heads_gene):
                head = self.heads_gene[species_i]
                if head is not None and gene_out_mask is not None:
                    # Gene heads need gene metadata, call directly (not through Sequential)
                    if len(head) == 1:
                        y_gene = head[0](x, gene_out_mask, gene_presence)
                    else:
                        # Multi-block head: process all but last, then call last with gene args
                        h = x
                        for block_idx, block in enumerate(head):
                            if block_idx == len(head) - 1:
                                y_gene = block(h, gene_out_mask, gene_presence)
                            else:
                                h = block(h)
                    gene_preds.append(y_gene)

        # Combine results
        coverage_pred = None
        gene_pred = None

        if coverage_preds:
            coverage_pred = (
                torch.cat(coverage_preds, dim=1)
                if len(coverage_preds) > 1
                else coverage_preds[0]
            )

        if gene_preds:
            gene_pred = (
                torch.cat(gene_preds, dim=1) if len(gene_preds) > 1 else gene_preds[0]
            )

        return ModelOutput(coverage=coverage_pred, gene=gene_pred)


class _RowWriter:
    """Write batch rows along a store's seq axis, buffering full ``seq_chunk``
    blocks so partial-chunk writes don't force zarr to read-modify-write each
    chunk. ``seq_chunk=None`` writes through directly (numpy/in-RAM path).
    """

    def __init__(self, arr, seq_chunk):
        self.arr = arr
        self.chunk = seq_chunk
        self.buf = (
            np.empty((seq_chunk, *arr.shape[1:]), dtype=arr.dtype)
            if seq_chunk
            else None
        )
        self.base = 0  # store row index of buf[0]
        self.fill = 0  # rows currently buffered

    def add(self, si, rows):
        if self.buf is None:
            self.arr[si : si + rows.shape[0]] = rows
            return
        if si != self.base + self.fill:  # a skipped seq: realign, keep chunks clean
            self.flush()
            self.base = si
        n, off = rows.shape[0], 0
        while off < n:
            take = min(self.chunk - self.fill, n - off)
            self.buf[self.fill : self.fill + take] = rows[off : off + take]
            self.fill += take
            off += take
            if self.fill == self.chunk:
                self.arr[self.base : self.base + self.chunk] = self.buf
                self.base += self.chunk
                self.fill = 0

    def flush(self):
        if self.buf is not None and self.fill:
            self.arr[self.base : self.base + self.fill] = self.buf[: self.fill]
            self.base += self.fill
            self.fill = 0
