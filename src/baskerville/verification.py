# Copyright 2024 Calico Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""Forward-pass regression / download-integrity verification for published models.

One mechanism, two layers (forward on CPU; on CUDA for Hydra models, whose scan
is a Triton kernel):

* Layer A (synthetic weights): every parameter is filled deterministically from a
  fixed seed, so no trained checkpoint is needed. This runs in CI and guards
  against code changes that silently alter the forward numerics of a published
  published architecture.
* Layer B (real weights): a downloaded ``model_best.pth`` is loaded instead.
  Higher fidelity (real weight distribution); also the end-user "did my download
  work" integrity check.

A compact fingerprint (output shape + global stats + a fixed-seed sample of
output values) is committed per config under ``releases/<family>/verify/``.
Verification recomputes the fingerprint and passes iff the shape matches exactly
and the Pearson r of the sampled values against the committed golden is
>= R_THRESHOLD.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted

# Deterministic seeds / sizes. Changing any of these invalidates committed
# goldens and requires regenerating them (hound_verify --generate).
SEQ_SEED = 42
WEIGHT_SEED = 7
SAMPLE_SEED = 1234
N_SAMPLES = 4096
R_THRESHOLD = 0.9999
# Pearson r is scale/shift invariant, so it alone misses a uniform affine drift
# (e.g. a changed norm epsilon or constant). Also gate on relative RMSE of the
# raw sampled values. Same-env reproduction is bit-identical (rmse 0); this
# tolerance only absorbs cross-BLAS/torch-build float noise, while still
# catching systematic drift above ~0.1%.
RMSE_THRESHOLD = 1e-3

FAMILIES = ("borzoi", "borzoi_prime", "cerberus")
SPECIES = ("human", "mouse")

# src/baskerville/verification.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def releases_dir(family: str) -> Path:
    return _REPO_ROOT / "releases" / family


def verify_dir(family: str) -> Path:
    return releases_dir(family) / "verify"


def is_joint(family: str) -> bool:
    """One multi-species model per fold (params.json) vs one model per species."""
    return (releases_dir(family) / "params.json").exists()


def params_path(family: str, species: str) -> Path:
    if is_joint(family):
        return releases_dir(family) / "params.json"
    return releases_dir(family) / f"params_{species}.json"


def models_dir(family: str, species: str) -> Path:
    sub = "models" if is_joint(family) else f"models_{species}"
    return releases_dir(family) / sub


def synth_ref_path(family: str, species: str) -> Path:
    return verify_dir(family) / f"ref_synth_{species}.npz"


def real_ref_path(family: str, species: str, fold: int) -> Path:
    return verify_dir(family) / f"ref_real_{species}_f{fold}.npz"


_FOLD_RE = re.compile(r"^f(\d+)c0$")


def convention_weights(family: str, species: str, fold: int) -> Path | None:
    """Trained weights at the single shared convention path.

    This is exactly where ``releases/<family>/download.sh`` writes when run from
    the release directory.
    """
    p = models_dir(family, species) / f"f{fold}c0" / "model_best.pth"
    return p if p.exists() else None


def discover_real_folds(family: str, species: str) -> list[int]:
    """Folds whose trained weights are present at the convention path.

    Empty list -> no downloaded weights (run the synthetic-weight check).
    """
    base = models_dir(family, species)
    folds = []
    if base.is_dir():
        for d in base.glob("f*c0"):
            m = _FOLD_RE.match(d.name)
            if m and (d / "model_best.pth").exists():
                folds.append(int(m.group(1)))
    return sorted(folds)


def source_tree(models_root: Path, family: str, species: str):
    """(params, {fold: weights}) in a known-good source tree (for --generate).

    Joint families use the bucket layout <root>/{params.json,f<n>c0/model_best.pth};
    per-species families the training layout <root>/models_<species>/f<n>c0/train/.
    """
    root = Path(models_root)
    if is_joint(family):
        base, params, rel = root, root / "params.json", "model_best.pth"
    else:
        base = root / f"models_{species}"
        params, rel = base / "f0c0" / "train" / "params.json", "train/model_best.pth"
    weights = {}
    if base.is_dir():
        for d in base.glob("f*c0"):
            m = _FOLD_RE.match(d.name)
            if m and (d / rel).exists():
                weights[int(m.group(1))] = d / rel
    return params, dict(sorted(weights.items()))


def synthetic_one_hot(seq_length: int, seed: int = SEQ_SEED) -> np.ndarray:
    """Deterministic (4, L) float32 one-hot input.

    Uses numpy's PCG64 (default_rng), which is stable across numpy versions and
    platforms, so the input is byte-identical everywhere.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=seq_length)
    oh = np.zeros((4, seq_length), dtype=np.float32)
    oh[idx, np.arange(seq_length)] = 1.0
    return oh


def fill_synthetic_weights(model: torch.nn.Module, seed: int = WEIGHT_SEED) -> None:
    """Deterministically fill every tensor of ``model``'s state_dict.

    Weights get a Kaiming-style ``N(0, 1/sqrt(fan_in))`` scale (keeps a deep
    524k-length forward through the softplus head numerically bounded), biases
    zero, and normalization layers are set to an identity transform
    (gamma=1, beta=0, running_mean=0, running_var=1). Reproducible across
    machines/torch versions because values come from numpy's PCG64, not torch
    init.
    """
    rng = np.random.default_rng(seed)
    sd = model.state_dict()
    new_sd = {}
    for name in sorted(sd):
        t = sd[name]
        if not torch.is_floating_point(t):
            # integer buffers, e.g. BatchNorm num_batches_tracked
            new_sd[name] = torch.zeros_like(t)
        elif name.endswith("running_mean") or name.endswith(".bias"):
            new_sd[name] = torch.zeros_like(t)
        elif name.endswith("running_var"):
            new_sd[name] = torch.ones_like(t)
        elif name.endswith(".weight") and t.ndim == 1:
            # normalization affine weight (BatchNorm/LayerNorm gamma)
            new_sd[name] = torch.ones_like(t)
        else:
            shape = tuple(t.shape)
            fan_in = int(np.prod(shape[1:])) if t.ndim >= 2 else 1
            scale = 1.0 / math.sqrt(max(fan_in, 1))
            vals = rng.standard_normal(size=shape).astype(np.float32) * scale
            new_sd[name] = torch.from_numpy(vals).to(t.dtype)
    model.load_state_dict(new_sd, strict=True)


def needs_cuda(params_json) -> bool:
    with open(params_json) as f:
        trunk = json.load(f)["model"]["trunk"]
    return any(b["name"].startswith("Hydra") for b in trunk)


def head_index(params_json, species: str) -> int:
    with open(params_json) as f:
        heads = natsorted(k for k in json.load(f)["model"] if k.startswith("head"))
    return heads.index(f"head_{species}")


def run_forward(params_json, weights_path, species: str) -> np.ndarray:
    """Run a deterministic fp32 forward, returning the (C, T) coverage array.

    weights_path is None -> Layer A (synthetic weights); otherwise Layer B.
    """
    from baskerville.seqnn import SeqNN

    with open(params_json) as f:
        params = json.load(f)
    device = "cuda" if needs_cuda(params_json) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.allow_tf32 = False
    # SeqNN.__init__ prints the full module; keep CLI/test output clean.
    with contextlib.redirect_stdout(io.StringIO()):
        snn = SeqNN(params["model"])
    snn.set_device(device)
    if weights_path is not None:
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Deliberately not routed through SeqNN.restore: that drops
        # weights_only=True, loads onto self.device, and remaps legacy
        # heads. keys -- key remapping could mask genuine architecture
        # drift and defeat this guard, which requires an exact strict
        # match against the pinned architecture. Only the safe
        # torch.compile _orig_mod. prefix strip is applied, so an
        # end-user's compiled checkpoint still verifies.
        if any("_orig_mod." in k for k in sd):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        snn.model.load_state_dict(sd, strict=True)
    else:
        fill_synthetic_weights(snn.model)
    snn.model.eval()
    snn.model.to(device)

    oh = synthetic_one_hot(snn.seq_length)
    x = torch.from_numpy(oh)[None].to(device)  # (1, 4, L)
    with torch.no_grad():
        out = snn.model(x, head_index(params_json, species))
    return out.coverage[0].float().cpu().numpy()  # (C, T)


def _sample_indices(n_total: int, seed: int = SAMPLE_SEED, n: int = N_SAMPLES):
    rng = np.random.default_rng(seed)
    n = min(n, n_total)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def fingerprint(output: np.ndarray) -> dict:
    """Compact, git-committable summary of a (C, T) forward output."""
    flat = output.reshape(-1).astype(np.float64)
    idx = _sample_indices(flat.size)
    return {
        "shape": np.asarray(output.shape, dtype=np.int64),
        "stats": np.asarray(
            [flat.mean(), flat.std(), flat.min(), flat.max(), flat.sum()],
            dtype=np.float64,
        ),
        "samples": flat[idx].astype(np.float32),
    }


@dataclass
class CompareResult:
    passed: bool
    shape_ok: bool
    r: float
    max_abs: float
    rmse_rel: float
    ref_shape: tuple
    cur_shape: tuple


def compare(
    ref: dict,
    output: np.ndarray,
    r_threshold: float = R_THRESHOLD,
    rmse_threshold: float = RMSE_THRESHOLD,
) -> CompareResult:
    ref_shape = tuple(int(v) for v in ref["shape"])
    cur_shape = tuple(int(v) for v in output.shape)
    shape_ok = ref_shape == cur_shape
    if not shape_ok:
        return CompareResult(
            False,
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            ref_shape,
            cur_shape,
        )

    flat = output.reshape(-1).astype(np.float64)
    cur = flat[_sample_indices(flat.size)]
    refv = np.asarray(ref["samples"], dtype=np.float64)

    a = cur - cur.mean()
    b = refv - refv.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        r = float(np.dot(a, b) / denom)
    else:
        r = 1.0 if np.allclose(cur, refv) else 0.0
    max_abs = float(np.abs(cur - refv).max())
    rmse = float(np.sqrt(np.mean((cur - refv) ** 2)))
    rmse_rel = rmse / (float(np.sqrt(np.mean(refv**2))) + 1e-12)
    passed = shape_ok and r >= r_threshold and rmse_rel <= rmse_threshold
    return CompareResult(passed, shape_ok, r, max_abs, rmse_rel, ref_shape, cur_shape)


def save_fingerprint(path: Path, fp: dict, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        shape=fp["shape"],
        stats=fp["stats"],
        samples=fp["samples"],
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
    )


def load_fingerprint(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {
        "shape": d["shape"],
        "stats": d["stats"],
        "samples": d["samples"],
        "meta": json.loads(str(d["meta"])),
    }


def make_meta(family: str, species: str, fold, kind: str, output: np.ndarray) -> dict:
    """kind is 'synthetic' (Layer A) or 'trained' (Layer B)."""
    return {
        "family": family,
        "species": species,
        "fold": fold,
        "kind": kind,
        "output_shape": [int(v) for v in output.shape],
        "seq_seed": SEQ_SEED,
        "weight_seed": WEIGHT_SEED if kind == "synthetic" else None,
        "sample_seed": SAMPLE_SEED,
        "n_samples": N_SAMPLES,
        "r_threshold": R_THRESHOLD,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def verify(family: str, species: str, r_threshold: float = R_THRESHOLD):
    """Verify one config, auto-detecting downloaded weights.

    Uses the first discovered trained fold if any are present, otherwise the
    synthetic-weight check. Never skips. Returns (fold_or_None, CompareResult).
    """
    pp = params_path(family, species)
    folds = discover_real_folds(family, species)
    if folds:
        fold = folds[0]
        out = run_forward(pp, convention_weights(family, species, fold), species)
        ref = load_fingerprint(real_ref_path(family, species, fold))
    else:
        fold = None
        out = run_forward(pp, None, species)
        ref = load_fingerprint(synth_ref_path(family, species))
    return fold, compare(ref, out, r_threshold)
