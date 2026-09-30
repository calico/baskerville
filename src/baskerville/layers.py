from einops import rearrange, repeat
import math
import torch
from torch import Tensor, einsum, nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except (ImportError, RuntimeError):
    RMSNormGated = None
    mamba_chunk_scan_combined = None

# separate: mamba_ssm builds predating Mamba-3 still provide the Mamba-2 ops above
try:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
except (ImportError, RuntimeError):
    mamba3_siso_combined = None

from baskerville.position import (
    get_central_mask,
    relative_shift_orig,
    relative_shift_hing,
)

from baskerville.deprecated.layers import *
from baskerville.condnorm import CondBatchNorm1d


class Activation(nn.Module):
    """
    A nonlinear activation layer.

    Args:
        func: The type of activation function. Supported values are 'relu',
            'elu', 'softplus', 'gelu', 'exp', and 'silu'. If None, will return
            nn.Identity.

    Raises:
        NotImplementedError: If 'func' is not a supported activation function.
    """

    def __init__(self, func: str) -> None:
        super().__init__()

        if func == "relu":
            self.layer = nn.ReLU()
        elif func == "elu":
            self.layer = nn.ELU()
        elif func == "gelu":
            self.layer = nn.GELU()
        elif func == "softplus":
            self.layer = nn.Softplus()
        elif func == "exp":
            self.layer = torch.exp
        elif func == "silu":
            self.layer = nn.SiLU()
        elif func is None:
            self.layer = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input tensor

        Returns:
            Output tensor
        """
        return self.layer(x)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Precomputes cos/sin tables and rotates all dimensions of Q and K.

    Args:
        dim: Head dimension (must be even).
        base: Base for frequency computation.
        max_seq_len: Maximum sequence length for precomputation.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple:
        """Apply rotary embedding to q and k.

        Args:
            q, k: (batch, heads, seq_len, head_dim)

        Returns:
            Rotated q, k of same shape.
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )


class AttentionSDP(nn.Module):
    """Multi-head attention using PyTorch's scaled_dot_product_attention.

    Supports RoPE, ALiBi, or no positional encoding. Automatically dispatches
    to FlashAttention-2, Memory-Efficient, or Math backend based on hardware.

    Args:
        channels: Number of input/output channels (embed_dim).
        heads: Number of attention heads.
        dropout: Attention dropout probability.
        pos_embed: Positional encoding type: "rope", "alibi", or None.
        seq_len: Required when pos_embed is set.
        gated: Sigmoid gating after SDPA output (Qiu et al., 2025).
            "elementwise" for per-dimension gating, "headwise" for per-head scalar gating.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float = 0.0,
        pos_embed: str = None,
        seq_len: int = None,
        gated: str = None,
    ):
        super().__init__()
        self.heads = heads
        assert channels % heads == 0, (
            f"channels ({channels}) must be divisible by heads ({heads})"
        )
        self.head_dim = channels // heads
        self.dropout = dropout
        self.pos_embed = pos_embed
        self.gated = gated

        self.Wqkv = nn.Linear(channels, 3 * channels, bias=True)
        self.out_proj = nn.Linear(channels, channels, bias=True)

        if self.gated == "elementwise":
            self.gate_proj = nn.Linear(channels, channels, bias=False)
        elif self.gated == "headwise":
            self.gate_proj = nn.Linear(channels, heads, bias=False)
        elif self.gated is not None:
            raise ValueError(
                f"gated must be 'elementwise', 'headwise', or None, got '{gated}'"
            )

        if pos_embed == "rope":
            assert self.head_dim % 2 == 0, (
                f"RoPE head_dim ({self.head_dim}) must be even"
            )
            assert seq_len is not None, "seq_len required for RoPE"
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=seq_len)
        elif pos_embed == "alibi":
            self.register_buffer("alibi_slopes", self._alibi_slopes(heads))

    @staticmethod
    def _alibi_slopes(heads: int) -> Tensor:
        """Compute ALiBi slope per head. Shape: (heads, 1, 1)."""
        ratio = (2**8) ** (1 / heads)
        return torch.tensor([1 / ratio ** (i + 1) for i in range(heads)]).view(
            heads, 1, 1
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (N, L, C)

        Returns:
            Output tensor of shape (N, L, C)
        """
        B, L, C = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.reshape(B, L, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if self.pos_embed == "rope":
            q, k = self.rotary(q, k)
        elif self.pos_embed == "alibi":
            positions = torch.arange(L, device=x.device)
            rel_pos = torch.abs(positions[None, :] - positions[:, None])
            attn_mask = -self.alibi_slopes * rel_pos

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        if self.gated is not None:
            gate = torch.sigmoid(self.gate_proj(x))
            if self.gated == "headwise":
                # (B, L, heads) -> (B, heads, L, 1)
                gate = gate.unsqueeze(-1).transpose(1, 2)
            else:
                # (B, L, C) -> (B, heads, L, head_dim)
                gate = gate.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
            out = out * gate

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out_proj(out)


class AttentionEnformer(nn.Module):
    def __init__(
        self,
        channels: int,
        key_len: int,
        heads: int,
        pos_features: int,
        pos_dropout: float = 0,
        attn_dropout: float = 0,
        shift_method: str = "original",
        gated: str = None,
    ):
        """
        Multi-head Attention (MHA) layer. Modified from
        https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py

        This code is adapted from gReLU
        Source: https://github.com/Genentech/gReLU
        License: MIT (https://opensource.org/licenses/MIT)

        Args:
            channels: Number of input/output channels
            key_len: Length of the key vectors
            heads: Number of attention heads
            pos_features: Number of positional embedding features
            pos_dropout: Dropout probability in the positional embeddings
            attn_dropout: Dropout probability in the output layer
            shift_method: Method for shifting positional embeddings.
            gated: Sigmoid gating applied to the attention output (Qiu et al., 2025).
                "elementwise" for per-dimension gating, "headwise" for per-head scalar gating.
        """
        super().__init__()

        # Save params
        self.channels = channels
        self.key_len = key_len
        self.heads = heads
        if channels % heads != 0:
            raise ValueError(
                f"AttentionEnformer channels ({channels}) must be divisible by heads ({heads})"
            )
        self.head_dim = channels // heads
        self.pos_features = pos_features
        self.shift_method = shift_method
        self.gated = gated

        # Create linear layers
        self.q_proj = nn.Linear(self.channels, self.key_len * self.heads, bias=False)
        self.k_proj = nn.Linear(self.channels, self.key_len * self.heads, bias=False)
        self.v_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.out_proj = nn.Linear(channels, self.channels)

        # regularization / initialization

        # relative positional encoding
        self.positional_embed = get_central_mask
        self.to_pos_k = nn.Linear(
            self.pos_features, self.key_len * self.heads, bias=False
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, self.heads, 1, self.key_len)
        )
        self.rel_pos_bias = nn.Parameter(torch.randn(1, self.heads, 1, self.key_len))

        # dropouts
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # gated attention
        if gated == "elementwise":
            self.gate_proj = nn.Linear(channels, channels, bias=False)
        elif gated == "headwise":
            self.gate_proj = nn.Linear(channels, heads, bias=False)
        elif gated is not None:
            raise ValueError(
                f"gated must be 'elementwise', 'headwise', or None, got '{gated}'"
            )

    def _get_pos_k(self, x):
        positions = self.positional_embed(x, out_channels=self.pos_features)
        positions = self.pos_dropout(positions)
        pos_k = self.to_pos_k(positions)
        pos_k = rearrange(pos_k, "n (h d) -> h n d", h=self.heads)
        return pos_k

    def get_attn_scores(self, x, return_v=False):
        # Q, K, V
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Get content embeddings
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        q = q / (self.key_len**0.5)

        # Content logits
        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        # Positional embeddings
        pos_k = self._get_pos_k(x)

        # Positional logits
        if self.shift_method == "original":
            pos_logits = einsum(
                "b h i d, h j d -> b h i j", q + self.rel_pos_bias, pos_k
            )
            pos_logits = relative_shift_orig(pos_logits)
        elif self.shift_method == "hingerl":
            pos_logits = relative_shift_hing(q + self.rel_pos_bias, pos_k)
        else:
            raise ValueError(f"Invalid shift method: {self.shift_method}")

        # Add content and positional embeddings
        logits = content_logits + pos_logits

        # Softmax
        attn = logits.softmax(dim=-1)

        if return_v:
            return self.attn_dropout(attn), v
        else:
            return self.attn_dropout(attn)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        # Get attention scores
        attn, v = self.get_attn_scores(x, return_v=True)

        # Output
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        if self.gated is not None:
            B, _, L, _ = out.shape
            gate = torch.sigmoid(self.gate_proj(x))
            if self.gated == "headwise":
                gate = gate.unsqueeze(-1).transpose(1, 2)
            else:
                gate = gate.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
            out = out * gate

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out_proj(out)


class Hydra(nn.Module):
    """
    Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.
    Base code from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        act_func="silu",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.act_func = act_func
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = (
            2 * self.d_inner + 2 * (2 * self.ngroups * self.d_state) + 2 * self.nheads
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * (2 * self.ngroups * self.d_state)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        if self.act_func == "silu":
            self.act = nn.SiLU()
        elif self.act_func == "gelu":
            self.act = nn.GELU()
        else:
            raise NotImplementedError(
                f"Activation function {self.act_func} not implemented"
            )

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs
        )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        initial_states = (
            repeat(self.init_states, "... -> b ...", b=2 * batch)
            if self.learnable_init_states
            else None
        )
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * (2 * self.ngroups * self.d_state),
                2 * self.nheads,
            ],
            dim=-1,
        )

        dt = torch.cat(
            (dt[:, :, : self.nheads], torch.flip(dt[:, :, self.nheads :], (1,))), dim=0
        )
        dt = F.softplus(dt + self.dt_bias)  # (2 * B, L, nheads)

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * (2 * ngroups * d_state))

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, BC = torch.split(
            xBC, [self.d_inner, 2 * (2 * self.ngroups * self.d_state)], dim=-1
        )
        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        BC = torch.cat(
            (
                BC[:, :, : 2 * self.ngroups * self.d_state],
                torch.flip(BC[:, :, 2 * self.ngroups * self.d_state :], (1,)),
            ),
            dim=0,
        )
        B, C = torch.split(
            BC, [self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y = (
            y_fw
            + y_bw
            + x_og
            * repeat(
                F.linear(x_og, self.fc_D.weight, bias=self.D),
                "b l h -> b l (h p)",
                p=self.headdim,
            )
        )

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)

        return out


def heavy_tail_activation(x: Tensor) -> Tensor:
    """
    Heavy-tail activation for Mamba-3's data-dependent A.

        f(x) = 1 + x        if x >= 0
             = 1 / (1 - x)   if x < 0

    Positive, continuous, and differentiable at x = 0. Improves stability at
    higher learning rates. Copied from mamba_ssm.modules.mamba3 to avoid importing
    that module (which eagerly pulls optional TileLang/CuTe backends).
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


class Hydra3(nn.Module):
    """
    Bidirectional Mamba-3, built with Hydra's quasiseparable matrix-mixer construction.

    Hydra (Hwang et al. 2024, Prop 3.7) turns any causal SSM scan SS(.) into a
    bidirectional quasiseparable mixer:

        QS(X) = shift(SS(X)) + flip(shift(SS(flip(X)))) + D X

    where shift(.) rolls the sequence right by one (zeroing position 0) and D is a
    free diagonal. The construction is SSM-agnostic; here SS(.) is the Mamba-3 SISO
    scan (Lahoti et al. 2026: exponential-trapezoidal discretization + data-dependent
    rotary/complex state updates), replacing Mamba-2/SSD used by the original Hydra.

    This mirrors the structure of the Hydra block above 1:1 (shared input projection,
    roll/flip combine, separate data-dependent diagonal, gated RMSNorm), only swapping
    the scan kernel and its preprocessing. There is no short conv1d (Mamba-3's
    trapezoidal rule is itself an in-recurrence width-2 convolution on the state-input).
    MIMO is a decode-efficiency feature and is intentionally not used (SISO only).
    """

    def __init__(
        self,
        d_model,
        d_state=32,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        rope_rate=1.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        dyn_init_scale=1.0,
        chunk_size=64,
        act_func="silu",  # absorbed for config compatibility (gate uses silu)
        dropout=0.0,  # absorbed (dropout lives in HydraBlock)
        layer_idx=None,  # absorbed for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        self.A_floor = A_floor
        # r > 1 sends atanh(r tanh(angle)) to NaN wherever |r tanh(angle)| >= 1
        assert 0 <= rope_rate <= 1.0, f"rope_rate must be in [0, 1], got {rope_rate}"
        self.rope_rate = rope_rate
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        assert mamba3_siso_combined is not None, (
            "mamba3_siso_combined is unavailable; install mamba_ssm with Mamba-3 "
            "(see the recipe in dockerfiles/baskerville.Dockerfile)."
        )

        # RoPE sizing (mirror mamba_ssm.modules.mamba3). d_state is the rotary head
        # dim: it must be even (the kernel asserts headdim_qk % 2 == 0) so that there
        # is at least one rotary angle. d_state=1 (Mamba-2 Hydra setting) is invalid.
        assert rope_fraction in (0.5, 1.0)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        assert self.num_rope_angles > 0, (
            f"d_state={d_state} with rope_fraction={rope_fraction} yields no rotary "
            "angles; use an even power-of-two d_state >= 4 (e.g. 32)."
        )

        # Input projection. z and x are shared across directions; B and C are duplicated
        # (forward + backward). The data-dependent dynamics (dd_dt, dd_A, trap, angle)
        # live in a SEPARATE projection (in_proj_dyn) so they can be init-scaled and
        # given their own learning rate: these drive Mamba-3's rotational/recurrent state
        # and dominate training instability. Splitting one matmul into two is numerically
        # equivalent at dyn_init_scale=1.0 (identical fan-in init distribution). Order:
        #   in_proj:     [z, x, B(2), C(2)]
        #   in_proj_dyn: [dd_dt(2), dd_A(2), trap(2), angle(2)]
        bc = self.num_bc_heads * self.d_state
        d_in_proj = 2 * self.d_inner + 2 * (2 * bc)
        d_in_proj_dyn = 2 * (3 * self.nheads + self.num_rope_angles)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        self.in_proj_dyn = nn.Linear(
            self.d_model, d_in_proj_dyn, bias=False, **factory_kwargs
        )
        # gentle the data-dependent dynamics at init (1.0 = no change, backwards-compatible)
        self.dyn_init_scale = dyn_init_scale
        if dyn_init_scale != 1.0:
            with torch.no_grad():
                self.in_proj_dyn.weight.mul_(dyn_init_scale)

        # dt bias (shared across directions), softplus-inverse init.
        dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # B/C biases and norms (shared across directions). SISO uses mimo_rank=1.
        self.B_bias = nn.Parameter(
            1
            + torch.zeros(
                (self.nheads, 1, self.d_state), dtype=torch.float32, device=device
            )
        )
        self.C_bias = nn.Parameter(
            1
            + torch.zeros(
                (self.nheads, 1, self.d_state), dtype=torch.float32, device=device
            )
        )
        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        # Free quasiseparable diagonal (delta_i), data-dependent per head as in Hydra.
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

        # Gated normalization before the output projection (applied outside the kernel).
        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs
        )
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=False, **factory_kwargs
        )

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, _ = u.shape
        bc = self.num_bc_heads * self.d_state

        proj = self.in_proj(u)
        z, x, B, C = torch.split(
            proj,
            [self.d_inner, self.d_inner, 2 * bc, 2 * bc],
            dim=-1,
        )
        proj_dyn = self.in_proj_dyn(u)
        dd_dt, dd_A, trap, angles = torch.split(
            proj_dyn,
            [
                2 * self.nheads,
                2 * self.nheads,
                2 * self.nheads,
                2 * self.num_rope_angles,
            ],
            dim=-1,
        )

        # Build the doubled batch [forward ; flipped-backward] along dim 0. x and z are
        # shared: x is duplicated by flipping, z is consumed once (gate) after combine.
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        x_og = rearrange(x, "b l h p -> b l (h p)")  # forward x, for the diagonal term
        V = torch.cat((x, torch.flip(x, (1,))), dim=0)

        def combine(t, size):
            fw, bw = t[..., :size], t[..., size:]
            return torch.cat((fw, torch.flip(bw, (1,))), dim=0)

        B = combine(B, bc)
        C = combine(C, bc)
        dd_dt = combine(dd_dt, self.nheads)
        dd_A = combine(dd_A, self.nheads)
        trap = combine(trap, self.nheads)
        angles = combine(angles, self.num_rope_angles)

        # Preprocess exactly like Mamba3.forward (now on the 2*batch tensors).
        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)  # (2B, L, nheads), fp32
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        trap = rearrange(trap, "b l h -> b h l")
        angles = angles.to(torch.float32)
        if self.rope_rate != 1.0:
            # Cap the phase rate. The kernel advances phase at pi tanh(angle) dt, so
            # atanh(r tanh(angle)) makes rope_rate an exact multiplier on that rate and
            # a hard ceiling of r*pi -- unlike scaling `angle`, which the network can
            # undo by growing it. Bounded for r < 1, so the gradient is well behaved.
            angles = torch.atanh(self.rope_rate * torch.tanh(angles))
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1)

        B = self.B_norm(rearrange(B, "b l (g n) -> b l g n", g=self.num_bc_heads))
        C = self.C_norm(rearrange(C, "b l (g n) -> b l g n", g=self.num_bc_heads))

        # Mamba-3 SISO scan. D=None and Z=None: the diagonal skip and the gate are
        # applied once, outside the kernel (so they are not double-counted across the
        # two stacked directions) — exactly as the Mamba-2 Hydra block does.
        y = mamba3_siso_combined(
            Q=C,
            K=B,
            V=V,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=None,
            Z=None,
            chunk_size=self.chunk_size,
            Input_States=None,
            return_final_states=False,
            cu_seqlens=None,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # shift(.) : roll right by one, zero position 0 -> strictly off-diagonal.
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y = (
            y_fw
            + y_bw
            + x_og
            * repeat(
                F.linear(x_og, self.fc_D.weight, bias=self.D),
                "b l h -> b l (h p)",
                p=self.headdim,
            )
        )

        # Multiply "gate" branch and apply extra normalization layer.
        y = self.norm(y, z)
        out = self.out_proj(y.to(u.dtype))

        return out


class Norm(nn.Module):
    """
    A flexible normalization layer that can implement various normalization schemes.

    Args:
        func: Type of normalization function. Supported values are 'batch',
            'batch_sp', 'batch_sp_v2', 'layer', 'layer_sp', and 'rms'.
            If None or '', returns nn.Identity.
        in_dim: Number of features in the input tensor.
        num_species: Number of species.
        **kwargs: Additional arguments to pass to the normalization function.
    """

    def __init__(
        self,
        func: Optional[str] = None,
        in_dim: Optional[int] = None,
        num_species: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        eps = 1e-4
        self.layer = None
        self.layers = None

        def _require_in_dim() -> None:
            if in_dim is None:
                raise ValueError("Number of input features must be provided.")

        def _require_species_args() -> None:
            if in_dim is None or num_species is None:
                raise ValueError(
                    "Number of input features and number of species must be provided."
                )

        if func == "batch":
            _require_in_dim()
            self.layer = nn.BatchNorm1d(in_dim, eps=eps, **kwargs)

        elif func == "batch_sp":
            _require_species_args()
            self.layers = nn.ModuleList(
                [nn.BatchNorm1d(in_dim, eps=eps, **kwargs) for _ in range(num_species)]
            )

        elif func == "batch_sp_v2":
            _require_species_args()
            self.layer = CondBatchNorm1d(num_species, in_dim, eps=eps, **kwargs)

        elif func == "layer":
            _require_in_dim()
            self.layer = nn.LayerNorm(in_dim, eps=eps, **kwargs)

        elif func == "layer_sp":
            _require_species_args()
            self.layers = nn.ModuleList(
                [nn.LayerNorm(in_dim, eps=eps, **kwargs) for _ in range(num_species)]
            )

        elif func == "rms":
            _require_in_dim()
            self.layer = nn.RMSNorm(in_dim, eps=eps, **kwargs)

        elif func is None or func == "":
            self.layer = nn.Identity()

        else:
            raise NotImplementedError(f"Normalization function '{func}' not supported.")

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L) or (N, L, C) for layer norm.
            di : Input species index for selecting normalization layer instance.

        Returns:
            Output tensor
        """
        if self.layers is not None:
            if di is None:
                raise ValueError(
                    "Species index 'di' must be provided for species-specific normalization."
                )
            return self.layers[di](x)

        if isinstance(self.layer, CondBatchNorm1d):
            if di is None:
                raise ValueError(
                    "Species index 'di' must be provided for CondBatchNorm1d normalization."
                )
            return self.layer(x, di)

        return self.layer(x)


class Scale(nn.Module):
    """
    A learnable scaling layer.

    Args:
        init: Initial scale value.
    """

    def __init__(
        self,
        init: Optional[int] = 1.0,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor.

        Returns:
            Output tensor
        """
        return x * self.scale


class SpeciesEmbedding(nn.Module):
    """Augment tensor with a learnable species embedding.

    Args:
        num_species (int): Number of species / datasets
        channels (int): Number of channels
        axis (int): length axis
        scale (bool): Use learnable scale parameters
    """

    def __init__(
        self,
        channels,
        num_species=2,
        axis=-1,
        scale=False,
    ):
        super().__init__()
        self.num_species = num_species
        self.channels = channels
        self.axis = axis
        self.scale = scale

        self.species_bias = nn.Embedding(self.num_species, self.channels)
        torch.nn.init.zeros_(self.species_bias.weight)

        # optionally use trainable scale
        if self.scale:
            self.species_weight = nn.Embedding(self.num_species, self.channels)
            torch.nn.init.ones_(self.species_weight.weight)

    def forward(self, x, di):
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)
            di : Species integer index

        Returns:
            Output tensor
        """
        dit = torch.tensor([di]).to(device=x.device)

        offset_ = torch.unsqueeze(self.species_bias(dit), self.axis)
        coef_ = None
        if self.scale:
            coef_ = torch.unsqueeze(self.species_weight(dit), self.axis)

        if self.scale:
            x = x * coef_ + offset_
        else:
            x = x + offset_
        return x
