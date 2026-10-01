#!/usr/bin/env python
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
"""
hound_verify

Verify that downloaded model weights + the cloned code reproduce the published
forward pass, and guard against code changes that regress a published
architecture.

Verify (default) -- for every downloaded fold, run a deterministic forward (CPU;
CUDA for Hydra models such as Cerberus) and compare a compact fingerprint
against the committed reference. If no weights have been downloaded it instead
checks the architecture itself:

    python -m baskerville.scripts.hound_verify
    python -m baskerville.scripts.hound_verify --family borzoi --species human

Generate -- (maintainer, run once) recompute and overwrite the pinned params and
reference fingerprints from a known-good model tree. Per-species families read
<root>/models_<species>/f<n>c0/train/{params.json,model_best.pth}. Joint
families (one multi-species model per fold, e.g. Cerberus) read the bucket
layout <root>/{params.json,f<n>c0/model_best.pth} and need
releases/<family>/params.json in place first:

    python -m baskerville.scripts.hound_verify --generate \\
        --family borzoi --models-root /path/to/4-17/borzoi
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from baskerville import verification as V


def _families(arg):
    return list(V.FAMILIES) if arg == "all" else [arg]


def _species(arg):
    return list(V.SPECIES) if arg == "all" else [arg]


def _seq_length(params_json: Path) -> int:
    with open(params_json) as f:
        return int(json.load(f)["model"]["seq_length"])


def run_generate(args):
    family = args.family
    root = Path(args.models_root)

    # plan all forwards up front so the progress bar has a total
    plan = []  # (species, fold|None)  fold None == synthetic
    src_params, src_weights = {}, {}
    for species in _species(args.species):
        sp, weights = V.source_tree(root, family, species)
        if not sp.exists():
            print(f"skip {family}/{species}: no params at {sp}")
            continue
        src_params[species], src_weights[species] = sp, weights
        plan.append((species, None))
        plan += [(species, f) for f in weights]
    if not plan:
        print("nothing to generate")
        return 1

    seq_len = _seq_length(next(iter(src_params.values())))
    device = "CUDA" if V.needs_cuda(next(iter(src_params.values()))) else "CPU"
    print(
        f"{family}: {len(plan)} forward(s), 1 sequence x {seq_len} bp each ({device})"
    )
    for species, fold in tqdm(plan, desc=f"generate {family}", unit="fwd"):
        pinned = V.params_path(family, species)
        if fold is None:
            pinned.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_params[species], pinned)
            out = V.run_forward(pinned, None, species)
            assert out.std() > 0, "degenerate synthetic output (zero variance)"
            V.save_fingerprint(
                V.synth_ref_path(family, species),
                V.fingerprint(out),
                V.make_meta(family, species, None, "synthetic", out),
            )
        else:
            out = V.run_forward(pinned, src_weights[species][fold], species)
            V.save_fingerprint(
                V.real_ref_path(family, species, fold),
                V.fingerprint(out),
                V.make_meta(family, species, fold, "trained", out),
            )
    print(f"wrote references -> {V.verify_dir(family)}")
    return 0


def run_verify(args):
    # plan: one forward per downloaded fold; if a config has no weights,
    # one architecture check instead. Folds are auto-detected, not assumed.
    jobs = []  # (family, species, fold|None)
    for family in _families(args.family):
        for species in _species(args.species):
            if not V.params_path(family, species).exists():
                print(f"skip {family}/{species}: no pinned reference")
                continue
            folds = V.discover_real_folds(family, species)
            if folds:
                jobs += [(family, species, f) for f in folds]
            else:
                jobs.append((family, species, None))
    if not jobs:
        print("no configs to verify")
        return 1

    rows = []
    all_ok = True
    for family, species, fold in tqdm(jobs, desc="verify", unit="fwd"):
        try:
            if fold is not None:
                out = V.run_forward(
                    V.params_path(family, species),
                    V.convention_weights(family, species, fold),
                    species,
                )
                ref = V.load_fingerprint(V.real_ref_path(family, species, fold))
            else:
                out = V.run_forward(V.params_path(family, species), None, species)
                ref = V.load_fingerprint(V.synth_ref_path(family, species))
            res = V.compare(ref, out, args.r_threshold)
            all_ok &= res.passed
            rows.append((family, species, fold, res, None))
        except Exception as e:  # corrupt weights / missing reference / etc.
            all_ok = False
            rows.append(
                (
                    family,
                    species,
                    fold,
                    None,
                    f"{type(e).__name__}: {e}".splitlines()[0][:80],
                )
            )

    print(f"\n{'family':<13}{'species':<8}{'fold':<6}{'r':>10}{'rmse_rel':>12}  result")
    print("-" * 55)
    for family, species, fold, res, err in rows:
        fold_label = "-" if fold is None else f"f{fold}"
        if err is not None:
            print(
                f"{family:<13}{species:<8}{fold_label:<6}"
                f"{'-':>10}{'-':>12}  ERROR  {err}"
            )
            continue
        print(
            f"{family:<13}{species:<8}{fold_label:<6}"
            f"{res.r:>10.6f}{res.rmse_rel:>12.3e}  "
            f"{'PASS' if res.passed else 'FAIL'}"
        )
    print("-" * 55)
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["all", *V.FAMILIES], default="all")
    p.add_argument("--species", choices=["all", *V.SPECIES], default="all")
    p.add_argument(
        "--r-threshold",
        type=float,
        default=V.R_THRESHOLD,
        help=f"Pearson r pass threshold (default {V.R_THRESHOLD})",
    )
    p.add_argument(
        "--generate",
        action="store_true",
        help="(maintainer) regenerate pinned params + golden fingerprints",
    )
    p.add_argument(
        "--models-root",
        help="for --generate: known-good model tree (layout per family, see above)",
    )
    args = p.parse_args()

    if args.generate:
        if not args.models_root:
            p.error("--generate requires --models-root")
        if args.family == "all":
            p.error("--generate requires an explicit --family")
        sys.exit(run_generate(args))
    sys.exit(run_verify(args))


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
