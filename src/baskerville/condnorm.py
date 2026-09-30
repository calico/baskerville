from einops import rearrange, repeat
import math
import torch
from torch import Tensor, einsum, nn
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedBuffer, UninitializedParameter
from typing import Any, Optional


class _CondNormBase(nn.Module):
    """Common base of _CondInstanceNorm and _CondBatchNorm.
    Code adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/batchnorm.py
    """

    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "num_classes",
        "num_features",
        "affine",
    ]
    num_classes: int
    num_features: int
    eps: float
    momentum: Optional[float]
    affine: bool
    track_running_stats: bool

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # initialize affine parameters
        if self.affine:
            self.weight = Parameter(
                torch.empty(num_classes, num_features, **factory_kwargs)
            )
            self.bias = Parameter(
                torch.empty(num_classes, num_features, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # register running stats buffers
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_classes, num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_var", torch.ones(num_classes, num_features, **factory_kwargs)
            )
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]

            # register buffer for batch counter
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    [0] * num_classes,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # reset running stats
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:

        # reset buffers
        self.reset_running_stats()

        # optionally reset parameters
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_classes}, {num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = (
                    self.num_batches_tracked
                    if self.num_batches_tracked is not None
                    and self.num_batches_tracked.device != torch.device("meta")
                    else torch.tensor([0] * self.num_classes, dtype=torch.long)
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _CondBatchNorm(_CondNormBase):
    def __init__(
        self,
        num_classes: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_classes,
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            **factory_kwargs,
        )

    def forward(self, input: Tensor, di: int) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # get species index as singleton tensor on target device
        di_t = None
        if not isinstance(di, Tensor):
            di_t = torch.tensor(di)
        else:
            di_t = di

        if di_t.device != input.device:
            di_t = di_t.to(device=input.device)

        # update batch counter
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                # add 1 counter at species index
                self.num_batches_tracked.index_add_(
                    0,
                    di_t,
                    torch.tensor(
                        [1],
                        dtype=self.num_batches_tracked.dtype,
                        device=self.num_batches_tracked.device,
                    ),
                )

                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        torch.index_select(self.num_batches_tracked, 0, di_t).item()
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        exponential_average_factor = self.momentum  # delete in future

        # get views of running stats buffers for current species index
        _running_mean = self.running_mean.index_select(0, di_t).view(
            [self.num_features]
        )
        _running_var = self.running_var.index_select(0, di_t).view([self.num_features])

        # update running stats if in training mode
        if self.training:
            bn_training = True

            # update running stats
            dims = [0] + [2, 3, 4][: (input.dim() - 2)]
            mean = input.mean(dims)
            var = input.var(dims, correction=1)

            with torch.no_grad():
                # calculate running mean
                _upd_running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * _running_mean
                )
                # calculate running var
                _upd_running_var = (
                    exponential_average_factor * var
                    + (1 - exponential_average_factor) * _running_var
                )

                # update values
                self.running_mean = self.running_mean.index_put_(
                    (di_t,), _upd_running_mean
                )
                self.running_var = self.running_var.index_put_(
                    (di_t,), _upd_running_var
                )

        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # call batch_norm functional
        out = F.batch_norm(
            input,
            _running_mean if not self.training else None,
            _running_var if not self.training else None,
            None,
            None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        # optionally apply affine transforms (adapted from https://github.com/pytorch/pytorch/issues/8985)
        if self.weight is not None and self.bias is not None:
            # get affine parameters for target species index
            di_t_1hot = (
                F.one_hot(di_t, num_classes=self.num_classes)
                .unsqueeze(0)
                .to(dtype=input.dtype)
            )
            _weight = (di_t_1hot @ self.weight).unsqueeze(-1)
            _bias = (di_t_1hot @ self.bias).unsqueeze(-1)

            # apply transforms
            out = out * _weight + _bias

        return out


class CondBatchNorm1d(_CondBatchNorm):
    r"""Applies Class-conditional Batch Normalization."""

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")
