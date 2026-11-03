#!/usr/bin/env python
# Copyright 2026 Calico Life Sciences LLC
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
import argparse
import json

import numpy as np
import torch

from baskerville import dataset, metrics, seqnn
from baskerville.trainer import parse_loss, parse_target_weights

"""
hound_qmap_calibrate

Calibrate qmap_weight: per dataset, R = |grad_h qmap term| / |grad_h base loss| at the
trunk output h of a trained model, over random train batches; lambda = push / median R.
"""


def gradient_ratios(
    model, data, di, base_fn, qmap_fn, batches, batch_size, target_weights=None, seed=0
):
    """Per batch, R for dataset di's head; model is a SeqNN on its device."""
    net = model.model
    head = net.heads_cov[di]
    device = next(net.parameters()).device
    captured = {}
    hook = head.register_forward_pre_hook(lambda m, inp: captured.update(h=inp[0]))
    rng = np.random.default_rng(seed)
    n = min(len(data), batches * batch_size)
    idx = rng.choice(len(data), n, replace=False)
    autocast = device.type == "cuda" and model.mix_dtype != torch.float32
    ratios = []
    try:
        for bi in range(0, n, batch_size):
            items = [data[int(j)] for j in idx[bi : bi + batch_size]]
            x = torch.stack([it.sequence for it in items]).to(device).float()
            y = torch.stack([it.coverage_targets for it in items]).to(device).float()
            with (
                torch.no_grad(),
                torch.autocast(device.type, dtype=model.mix_dtype, enabled=autocast),
            ):
                net(x, di)
            h = captured["h"].float().requires_grad_()
            yh = head(h).float()
            if net.output_slice is not None:
                yh = yh[:, net.output_slice, :]
            base = base_fn(yh, y, target_weights)
            (g_base,) = torch.autograd.grad(base, h, retain_graph=True)
            (g_qmap,) = torch.autograd.grad(qmap_fn(yh, y), h)
            ratios.append((g_qmap.norm() / g_base.norm()).item())
    finally:
        hook.remove()
    return np.array(ratios)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate qmap_weight per dataset.")
    parser.add_argument(
        "--batches",
        type=int,
        default=16,
        help="Train batches per dataset [Default: %(default)s]",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size [Default: %(default)s]",
    )
    parser.add_argument(
        "--cross",
        type=int,
        default=0,
        help="Cross replicate index (folds data) [Default: %(default)s]",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold held out as test (folds data) [Default: %(default)s]",
    )
    parser.add_argument(
        "--form",
        default=None,
        help="qmap_form [Default: params train.qmap_form, else mse]",
    )
    parser.add_argument(
        "--push",
        type=float,
        default=0.1,
        help="Target ratio of qmap to base gradient at the trunk [Default: %(default)s]",
    )
    parser.add_argument("params_file", help="JSON file with model parameters")
    parser.add_argument("model_file", help="Trained model file")
    parser.add_argument("data_dirs", nargs="+", help="Data directories, one per head")
    args = parser.parse_args(argv)

    with open(args.params_file) as params_open:
        params = json.load(params_open)
    t = params["train"]
    form = args.form or t.get("qmap_form", "mse")

    model = seqnn.SeqNN(params["model"])
    model.restore(args.model_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device).eval()
    model.mix_dtype = getattr(torch, t.get("mix_dtype", "float32"))

    base_fn = parse_loss(
        t.get("loss", "poisson").lower(),
        t.get("spec_weight", 1),
        t.get("total_weight", 1),
        t.get("weight_range", 1),
        t.get("weight_exp", 1),
        t.get("split_exp", 0),
        t.get("split_knee", 1),
    )

    lambdas = []
    for di, data_dir in enumerate(args.data_dirs):
        fold = args.fold if dataset.discover_num_folds(data_dir) is not None else None
        data = dataset.SeqDataset(
            data_dir, "train", mode="eval", fold=fold, cross=args.cross
        )
        qmap_fn = metrics.QmapCrossTrackLoss(
            data.targets_df,
            data.target_hist,
            t.get("weight_range", 1),
            t.get("weight_exp", 1),
            t.get("spec_group_min", 20),
            form=form,
        ).to(device)
        tw = parse_target_weights(data.targets_df, t.get("loss_weight", []))
        tw = (
            None
            if np.allclose(tw, 1)
            else torch.tensor(tw, dtype=torch.float32, device=device)[None]
        )

        r = gradient_ratios(
            model, data, di, base_fn, qmap_fn, args.batches, args.batch_size, tw
        )
        lam = args.push / np.median(r)
        q25, q75 = np.quantile(r, [0.25, 0.75])
        print(
            f"{data_dir}: R median {np.median(r):.4g} (IQR {q25:.4g}-{q75:.4g}), "
            f"lambda {lam:.4g}"
        )
        lambdas.append(float(f"{lam:.4g}"))

    print(json.dumps({"qmap_form": form, "qmap_weight": lambdas}))


if __name__ == "__main__":
    main()
