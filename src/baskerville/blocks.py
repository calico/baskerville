from einops import rearrange
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    import borzoi_pytorch
except ImportError:
    pass

from baskerville.layers import *
from baskerville.deprecated.blocks import *

from inspect import signature


class ConvDNA(nn.Module):
    """Initial convolutional layer for DNA sequences.

    Args:
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        in_channels (int): Number of input channels.
        pool_size (int): Size of the pooling window.
        residual (bool): Use residual connections.
        act_func (str): Name of the activation function.
        norm_type (str): Type of normalization.
        dropout (float): Dropout probability.
        num_species (int): Number of species / datasets.
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        in_channels=4,
        pool_size=1,
        residual=False,
        act_func="gelu",
        norm_type="batch",
        dropout=0,
        num_species=2,
        grad_checkpoint=False,
    ):
        super().__init__()
        self.residual = residual
        self.grad_checkpoint = grad_checkpoint
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=True,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        if self.residual:
            self.norm = Norm(norm_type, in_dim=out_channels, num_species=num_species)
            self.act = Activation(act_func)
            self.conv2 = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
            )
            nn.init.kaiming_normal_(
                self.conv2.weight, mode="fan_in", nonlinearity="relu"
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, padding=0, ceil_mode=True)

    def forward(self, x, di=None):
        if self.grad_checkpoint:
            x = checkpoint(self.conv, x, use_reentrant=False)
        else:
            x = self.conv(x)
        if self.residual:
            xi = x
            x = self.norm(x, di)
            x = self.act(x)
            x = self.conv2(x)
            x = self.dropout(x)
            x = torch.add(xi, x)
        if self.grad_checkpoint:
            x = checkpoint(self.pool, x, use_reentrant=False)
        else:
            x = self.pool(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block for 1D sequences.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        pool_size (int): Size of the pooling window.
        act_func (str): Name of the activation function.
        norm_type (str): Type of normalization.
        residual (bool): Use residual connections.
        residual_norm_type (str): Type of normalization for residual connection.
        residual_kernel_size (int): Kernel width for residual convolution.
        residual_pad (bool): Use padded residual connections for initial conv.
        return_initial (bool): Additionally return output form initial conv.
        dropout (float): Residual dropout probability.
        num_species (int): Number of species / datasets.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        pool_size=1,
        act_func="gelu",
        norm_type="batch",
        residual=False,
        residual_norm_type="batch",
        residual_kernel_size=1,
        residual_pad=False,
        return_initial=False,
        dropout=0,
        num_species=2,
        conv_bias=False,
    ):
        super().__init__()
        self.residual = residual
        self.return_initial = return_initial
        self.residual_pad = residual_pad
        self.norm = Norm(norm_type, in_dim=in_channels, num_species=num_species)
        self.act = Activation(act_func)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=conv_bias,
        )
        self.pad = out_channels - in_channels
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        if residual:
            self.norm2 = Norm(
                residual_norm_type, in_dim=out_channels, num_species=num_species
            )
            self.act2 = Activation(act_func)
            self.conv2 = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=residual_kernel_size,
                padding="same",
                bias=True,
            )
            nn.init.kaiming_normal_(
                self.conv2.weight, mode="fan_in", nonlinearity="relu"
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if pool_size > 1:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, padding=0, ceil_mode=True)
        else:
            self.pool = None

    def forward(self, x, di=None):
        if self.residual_pad:
            xi = nn.functional.pad(x, (0, 0, 0, self.pad, 0, 0), "constant", 0)
        x = self.norm(x, di)
        x = self.act(x)
        x = self.conv(x)
        if self.residual_pad:
            x = torch.add(xi, x)
        if self.residual:
            xii = x
            x = self.norm2(x, di)
            x = self.act2(x)
            x = self.conv2(x)
            x = self.dropout(x)
            x = torch.add(xii, x)
        if self.pool is not None:
            x = self.pool(x)
        if self.residual and self.return_initial:
            return x, xii
        return x


class MBConvBlock(nn.Module):
    """Inverted-residual (MBConv) block for 1D sequences.

    Mirrors the Hydra macro-block with the selective scan removed: a
    pre-norm residual around ``expand -> depthwise conv -> project``. With
    ``gated=True`` the expansion also produces a gate that modulates the conv
    output SwiGLU-style (``value * act(gate)``; gate-only activation via ``act_func``).

    Runs channels-last internally, transposing only
    around the depthwise conv, so ``norm_type`` must be a channels-last norm
    ('rms' or 'layer'); 'batch' is not supported here.

    Args:
        channels (int): Number of input/output channels.
        expand (float): Expansion factor for the inner width.
        kernel_size (int): Depthwise convolution kernel width.
        gated (bool): Add a SwiGLU-style gate branch to the expansion.
        norm_type (str): Channels-last normalization ('rms' or 'layer').
        act_func (str): Name of the activation function.
        dropout (float): Residual dropout probability.
        layer_scale (float): LayerScale init; 0 disables it.
        num_species (int): Number of species / datasets.
    """

    def __init__(
        self,
        channels,
        expand=2.0,
        kernel_size=7,
        gated=False,
        norm_type="rms",
        act_func="silu",
        dropout=0,
        layer_scale=1e-6,
        num_species=2,
    ):
        super().__init__()
        if norm_type not in {"rms", "layer"}:
            raise ValueError(
                f"MBConvBlock requires a channels-last norm_type ('rms' or 'layer'); got {norm_type!r}."
            )
        inner = int(round(channels * expand))
        if inner < 1:
            raise ValueError(
                f"MBConvBlock inner width must be >= 1; got {inner} (channels={channels}, expand={expand})."
            )
        self.gated = gated
        self.norm = Norm(norm_type, in_dim=channels, num_species=num_species)
        self.act = Activation(act_func)
        self.in_proj = nn.Linear(channels, inner * (2 if gated else 1), bias=False)
        self.conv = nn.Conv1d(
            in_channels=inner,
            out_channels=inner,
            kernel_size=kernel_size,
            padding="same",
            groups=inner,
            bias=True,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        self.out_proj = nn.Linear(inner, channels, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if layer_scale:
            self.gamma = nn.Parameter(layer_scale * torch.ones(channels))
        else:
            self.gamma = None

    def forward(self, x, di=None):
        # (N, C, L) -> channels-last for norm / pointwise projections
        x = rearrange(x, "b c l -> b l c")
        xi = x
        x = self.norm(x, di)
        x = self.in_proj(x)
        if self.gated:
            x, z = x.chunk(2, dim=-1)
        # depthwise conv in channels-first
        x = rearrange(x, "b l c -> b c l")
        x = self.conv(x)
        x = rearrange(x, "b c l -> b l c")
        if self.gated:
            x = x * self.act(z)
        else:
            x = self.act(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = torch.add(xi, x)
        return rearrange(x, "b l c -> b c l")


class ConvTower(nn.Module):
    """Convolutional tower for 1D sequences.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        divisible_by (int): Force channels to be divisible by.
        pool_size (int): Size of the pooling window.
        repeat (int): Number of repetitions.
        ... ConvBlock arguments ...
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        divisible_by=1,
        repeat=1,
        pool_size=1,
        grad_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.repeat = repeat
        self.layers = nn.ModuleList()
        self.grad_checkpoint = grad_checkpoint
        self.residual_return_initial = kwargs.get("return_initial", False)

        # channel helpers
        scale_channels = np.exp(np.log(out_channels / in_channels) / repeat)

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        # initialize filters
        rep_out = in_channels

        for ri in range(self.repeat):
            # update filters
            rep_in = rep_out
            rep_out = rep_in * scale_channels

            # initial
            self.layers.append(
                ConvBlock(
                    in_channels=_round(rep_in),
                    out_channels=_round(rep_out),
                    **kwargs,
                )
            )

            # pool
            self.layers.append(
                nn.MaxPool1d(kernel_size=pool_size, padding=0, ceil_mode=True)
            )

    def forward(self, x, di=None):
        """
        Args:
            x : Input tensor of shape (N, C, L)
            di: Input species index

        Returns:
            Output tensor
            Intermediate representations for U-net.
        """
        crs = []
        xi = None
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool1d):
                if xi is not None:
                    crs.append(xi)
                else:
                    crs.append(x)
                if self.grad_checkpoint:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            else:
                if self.grad_checkpoint:
                    x = checkpoint(layer, x, di, use_reentrant=False)
                else:
                    x = layer(x, di)
                if self.residual_return_initial:
                    x, xi = x
        return x, crs


class Crop(nn.Module):
    """
    Crop layer for 1D data.

    Args:
        crop_size (int): Number of elements to crop from each side.
    """

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[..., self.crop_size : -self.crop_size]


class FeedForwardBlock(nn.Module):
    """
    2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

    Args:
        channels: Number of channels in the input/output sequence
        expansion: Expansion factor for the hidden layer
        dropout: Dropout probability
        act_func: Name of the activation function
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2,
        dropout: float = 0,
        act_func: str = "relu",
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        expansion_channels = int(channels * expansion)
        self.dense1 = LinearBlock(
            channels,
            expansion_channels,
            norm=norm_type,
            dropout=dropout,
            act_func=act_func,
            bias=True,
        )
        self.dense2 = LinearBlock(
            expansion_channels,
            channels,
            norm=None,
            dropout=0,
            act_func=None,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class Final(nn.Module):
    """Final dense layer with configurable output activation.

    Args:
        in_channels (int): Number of input channels.
        num_targets (int): Number of targets.
        act_func (str): Name of the activation function.
        norm_type (str): Type of normalization.
        output_act (str): Output activation. One of "softplus" (default), "softmax",
            "sigmoid", or "linear".
    """

    def __init__(
        self,
        in_channels,
        num_targets,
        act_func="gelu",
        norm_type="batch",
        output_act="softplus",
    ):
        super().__init__()
        self.num_targets = num_targets
        self.output_act_name = output_act
        self.norm = Norm(norm_type, in_dim=in_channels)
        self.act = Activation(act_func)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_targets,
            kernel_size=1,
            padding="same",
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        if output_act == "softplus":
            self.output_act = nn.Softplus()
        elif output_act == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif output_act == "softmax":
            self.output_act = nn.Softmax(dim=1)
        elif output_act == "linear":
            self.output_act = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported output_act={output_act!r}; expected "
                "'softplus', 'softmax', 'sigmoid', or 'linear'."
            )

    def forward(self, x):
        """
        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.output_act(x)
        return x


class FinalBorzoi(nn.Module):
    """Legacy Borzoi head: conv_nac expansion followed by activated projection.

    Mirrors the TF pair of `conv_nac` (Norm→Act→Conv→Dropout, expanding
    to ``hidden_channels``) followed here by an additional activation,
    a final 1x1 projection, and ``Softplus``. In TF the expansion is shared
    across heads; here each head carries its own copy.

    Args:
        in_channels (int): Number of input channels (trunk output).
        num_targets (int): Number of output targets.
        hidden_channels (int): Expansion channels for the conv_nac stage.
        act_func (str): Name of the activation function.
        norm_type (str): Type of normalization.
        dropout (float): Post-conv_nac dropout probability.
    """

    def __init__(
        self,
        in_channels,
        num_targets,
        hidden_channels=1920,
        act_func="gelu",
        norm_type="batch",
        dropout=0.1,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.norm = Norm(norm_type, in_dim=in_channels)
        self.act = Activation(act_func)
        self.conv_nac = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            padding="same",
        )
        nn.init.kaiming_normal_(
            self.conv_nac.weight, mode="fan_in", nonlinearity="relu"
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act_final = Activation(act_func)
        self.conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=num_targets,
            kernel_size=1,
            padding="same",
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_nac(x)
        x = self.dropout(x)
        x = self.act_final(x)
        x = self.conv(x)
        x = self.softplus(x)
        return x


class FinalGene(nn.Module):
    """Final layer for gene expression prediction.

    Aggregates trunk outputs over gene body positions to predict gene expression.

    Args:
        in_channels (int): Number of input channels from trunk.
        num_targets (int): Number of gene expression targets (samples).
        act_func (str): Name of the activation function.
        norm_type (str): Type of normalization.
        aggregation (str): Aggregation method ('sum' or 'mean').
        output_act (str): Per-bin output activation, 'softplus' or 'linear'.
    """

    def __init__(
        self,
        in_channels,
        num_targets,
        act_func="gelu",
        norm_type="batch",
        aggregation="sum",
        output_act="softplus",
    ):
        super().__init__()
        self.num_targets = num_targets
        self.aggregation = aggregation
        self.norm = Norm(norm_type, in_dim=in_channels)
        self.act = Activation(act_func)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_targets,
            kernel_size=1,
            padding="same",
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        if output_act == "softplus":
            self.output_act = nn.Softplus()
        elif output_act == "linear":
            self.output_act = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported output_act={output_act!r}; expected "
                "'softplus' or 'linear'."
            )

    def forward(self, x, gene_out_mask, gene_presence):
        """
        Args:
            x: Input tensor of shape (N, C, L) from trunk
            gene_out_mask: Boolean tensor of shape (N, max_genes, L) indicating
                which output bins overlap gene exons
            gene_presence: Boolean tensor of shape (N, max_genes) indicating valid genes

        Returns:
            Output tensor of shape (N, num_targets, max_genes)
        """
        # Apply normalization, activation, and convolution
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)  # (N, num_targets, L)
        x = self.output_act(x)

        # Combine bin mask with gene validity mask: (N, G, L)
        pos_mask = gene_out_mask & gene_presence.unsqueeze(-1)

        # Expand x for broadcasting: (N, T, 1, L)
        x_expanded = x.unsqueeze(2)

        # Expand mask for broadcasting: (N, 1, G, L)
        pos_mask_expanded = pos_mask.unsqueeze(1).to(x.dtype)

        # Compute weighted values: (N, T, G, L)
        weighted = x_expanded * pos_mask_expanded

        if self.aggregation == "sum":
            # Sum over sequence length: (N, T, G)
            gene_preds = weighted.sum(dim=-1)
        else:  # mean
            # Count valid positions per gene: (N, G)
            counts = pos_mask.sum(dim=-1).clamp(min=1)
            # Sum and divide by counts: (N, T, G)
            gene_preds = weighted.sum(dim=-1) / counts.unsqueeze(1)

        return gene_preds


class LinearBlock(nn.Module):
    """
    Linear layer followed by optional normalization,
    activation and dropout.

    gReLU

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        act_func: Name of activation function
        dropout: Dropout probability
        norm: Type of normalization
        bias: If True, include bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_func: str = "relu",
        dropout: float = 0.0,
        norm: str = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.norm = Norm(norm, in_dim=in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = Activation(act_func)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm residual connections.

    Expects (N, L, C) input and output layout. TransformerTower transposes
    from (N, C, L) before calling this block.

    Supports RoPE, ALiBi, or no positional encoding via PyTorch SDPA,
    which automatically dispatches to the best available attention backend
    (FlashAttention-2, Memory-Efficient, or Math).

    Args:
        channels: Number of input/output channels
        heads: Number of attention heads
        attn_dropout: Dropout probability in the attention layer
        ff_dropout: Dropout probability in the linear feed-forward layers
        expansion: Expansion factor for the FFN hidden layer
        pos_embed: Positional encoding: "rope", "alibi", or None
        seq_len: Required when pos_embed is set
        act_func: Activation function name
        norm_type: Normalization type
        extra_norm: Extra post-attention/FFN normalization
        gated: Sigmoid gating after SDPA output (Qiu et al., 2025).
            "elementwise" for per-dimension gating, "headwise" for per-head scalar gating.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        attn_dropout: float = 0,
        ff_dropout: float = 0,
        expansion: float = 2,
        pos_embed: str = None,
        seq_len: int = None,
        act_func="silu",
        norm_type="layer",
        extra_norm=False,
        gated: str = None,
    ) -> None:
        super().__init__()
        self.norm = Norm(norm_type, channels)
        self.mha = AttentionSDP(
            channels=channels,
            heads=heads,
            dropout=attn_dropout,
            pos_embed=pos_embed,
            seq_len=seq_len,
            gated=gated,
        )
        self.normr1 = Norm(norm_type, channels) if extra_norm else nn.Identity()
        self.dropout = nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity()
        self.ffn = FeedForwardBlock(
            channels=channels,
            expansion=expansion,
            dropout=ff_dropout,
            act_func=act_func,
            norm_type=norm_type,
        )
        self.normr2 = Norm(norm_type, channels) if extra_norm else nn.Identity()
        self.dropout2 = nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.normr1(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x)
        x = self.normr2(x)
        x = self.dropout2(x)
        x = torch.add(ffn_input, x)
        return x


class TransformerEnformerBlock(nn.Module):
    """
    A block containing a multi-head attention layer followed by a feed-forward
    network and residual connections.

    Args:
        channels: Number of input/output channels
        heads: Number of attention heads
        pos_features: Number of positional embedding features
        key_len: Length of the key vectors
        pos_dropout: Dropout probability in the positional embeddings
        attn_dropout: Dropout probability in the output layer
        ff_droppout: Dropout probability in the linear feed-forward layers
        expansion: Expansion factor for the FFN hidden layer
        shift_method: Method for relative positional embeddings
        gated: Sigmoid gating applied to the attention output (Qiu et al., 2025).
            "elementwise" for per-dimension gating, "headwise" for per-head scalar gating.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        pos_features: int = 32,
        key_len: int = 32,
        pos_dropout: float = 0,
        attn_dropout: float = 0,
        ff_dropout: float = 0,
        expansion: float = 2,
        act_func="relu",
        norm_type="layer",
        shift_method="original",
        extra_norm=False,
        seq_len=None,
        gated: str = None,
    ) -> None:
        super().__init__()
        self.norm = Norm(norm_type, channels)
        self.mha = AttentionEnformer(
            channels=channels,
            heads=heads,
            pos_features=pos_features,
            key_len=key_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            shift_method=shift_method,
            gated=gated,
        )
        self.normr1 = Norm(norm_type, channels) if extra_norm else nn.Identity()
        self.dropout = nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity()
        self.ffn = FeedForwardBlock(
            channels=channels,
            expansion=expansion,
            dropout=ff_dropout,
            act_func=act_func,
            norm_type=norm_type,
        )
        self.normr2 = Norm(norm_type, channels) if extra_norm else nn.Identity()
        self.dropout2 = nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.mha(x)
        x = self.normr1(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x)
        x = self.normr2(x)
        x = self.dropout2(x)
        x = torch.add(ffn_input, x)
        return x


class DropPath(nn.Module):
    """Stochastic depth: randomly zero the residual branch per sample in training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}.")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # per-sample mask
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class HydraBlock(nn.Module):
    """
    Residual Hydra block.

    Args:
        layer_scale (float): LayerScale init; 0 (default) disables it.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0,
        version: int = 2,
        drop_path: float = 0,
        layer_scale: float = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm = Norm("rms", channels)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale else None
        )
        # version 2 -> bidirectional Mamba-2 (Hydra); version 3 -> bidirectional Mamba-3
        mixer_cls = Hydra3 if version == 3 else Hydra
        self.hydra = mixer_cls(
            d_model=channels,
            **kwargs,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.hydra(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = self.dropout(x)
        x = self.drop_path(x)
        x = torch.add(x_input, x)
        return x


class HydraTower(nn.Module):
    """
    Multiple stacked Hydra layers.

    Args:
        repeat: Number of HydraBlock layers
        drop_path: Stochastic-depth drop probability for the residual branch
        drop_path_ramp: Ramp drop_path linearly from 0 at the first block to
            drop_path at the last, versus applying drop_path to every block
    """

    def __init__(
        self,
        repeat: int,
        grad_checkpoint: bool = False,
        drop_path: float = 0.0,
        drop_path_ramp: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if drop_path_ramp and repeat > 1:
            dp_rates = [drop_path * i / (repeat - 1) for i in range(repeat)]
        else:
            dp_rates = [drop_path] * repeat
        self.blocks = nn.ModuleList(
            [HydraBlock(drop_path=dp, **kwargs) for dp in dp_rates]
        )
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        for block in self.blocks:
            if self.grad_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = rearrange(x, "b l t -> b t l")
        return x


class TransformerTower(nn.Module):
    """
    Multiple stacked transformer encoder layers.

    Args:
        repeat: Number of TransformerBlock layers
    """

    def __init__(
        self,
        repeat: int,
        type: str = "",
        grad_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if type == "enformer":
            transformer_block = TransformerEnformerBlock
        else:
            transformer_block = TransformerBlock
        self.blocks = nn.ModuleList(
            [transformer_block(**kwargs) for _ in range(repeat)]
        )
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        for block in self.blocks:
            if self.grad_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = rearrange(x, "b l t -> b t l")
        return x


class UnetBorzoiBlock(nn.Module):
    """
    Legacy Borzoi-style upsampling U-net block.

    Args:
        channels: Number of input/output channels.
        unet_channels: Number of channels in the U-net connection.
        norm_type: Type of normalization.
        act_func: Name of the activation function.
        kernel_size: Size of the convolutional kernel.
        dropout: Dropout probability.
        num_species (int): Number of species / datasets.
    """

    def __init__(
        self,
        channels: int,
        unet_channels: int,
        norm_type: str = "batch",
        act_func: str = "gelu",
        kernel_size: int = 3,
        dropout: float = 0,
        skip_match: bool = False,
        upsample_conv: bool = False,
        num_species=2,
        **_,
    ):
        super().__init__()
        self.normx = Norm(norm_type, in_dim=channels, num_species=num_species)
        self.normu = Norm(norm_type, in_dim=unet_channels, num_species=num_species)
        self.act = Activation(act_func)
        if upsample_conv:
            self.convx = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                padding="same",
                bias=True,
            )
            nn.init.kaiming_normal_(
                self.convx.weight, mode="fan_in", nonlinearity="relu"
            )
        else:
            self.convx = nn.Identity()
        if skip_match and channels == unet_channels:
            self.convu = nn.Identity()
        else:
            self.convu = nn.Conv1d(
                in_channels=unet_channels,
                out_channels=channels,
                kernel_size=1,
                padding="same",
                bias=True,
            )
            nn.init.kaiming_normal_(
                self.convu.weight, mode="fan_in", nonlinearity="relu"
            )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_depth = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.conv_depth.weight, mode="fan_in", nonlinearity="relu"
        )
        self.conv_point = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=True,
        )
        nn.init.kaiming_normal_(
            self.conv_point.weight, mode="fan_in", nonlinearity="relu"
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, u: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Args:
            x : Input tensor of shape (N, C, L)
            u : Higher-resolution tensor of shape (N, C, 2L)
            di : Input species index

        Returns:
            Output tensor
        """
        # normalize/activate
        x = self.act(self.normx(x, di))
        u = self.act(self.normu(u, di))

        # align shapes
        x = self.convx(x)
        x = self.upsample(x)
        u = self.convu(u)

        # add
        x = torch.add(x, u)

        # convolution
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.dropout(x)

        return x


class UnetTower(nn.Module):
    """
    Upsampling U-net tower.

    Args:
        channels: Number of input/output channels.
        unet_channels: List of number of channels in the U-net connection, per U-net block.
        norm_type: Type of normalization.
        act_func: Name of the activation function.
        kernel_size: Size of the convolutional kernel.
        dropout: Dropout probability.
        num_species (int): Number of species / datasets.
        repeat (int): Number of repetitions.
        grad_checkpoint (bool): Use gradient checkpointing.
        type: U-net implementation to use. ``"v2"`` is the preferred default;
            ``"borzoi"`` selects the legacy implementation.
        **block_kwargs: Forwarded to the block constructor. ``v2`` options:
            ``x_residual``, ``x_scale``, ``unet_residual``, ``final_residual``.
            ``borzoi/v1`` options: ``upsample_conv``, ``skip_match`` (applied
            only to the first block).
    """

    def __init__(
        self,
        channels: int,
        unet_channels: [int],
        norm_type: str = "batch",
        act_func: str = "gelu",
        kernel_size: int = 3,
        dropout: float = 0,
        num_species=2,
        repeat=1,
        grad_checkpoint=False,
        type: str = "",
        **block_kwargs,
    ):
        super().__init__()
        self.repeat = repeat
        self.layers = nn.ModuleList()
        self.grad_checkpoint = grad_checkpoint
        tower_type = type.lower()

        shared = dict(
            channels=channels,
            norm_type=norm_type,
            act_func=act_func,
            kernel_size=kernel_size,
            dropout=dropout,
            num_species=num_species,
        )

        for ri in range(self.repeat):
            if tower_type in {"", "v2"}:
                self.layers.append(
                    UnetBlock(unet_channels=unet_channels[ri], **shared, **block_kwargs)
                )
                continue

            if tower_type not in {"borzoi", "v1"}:
                raise ValueError(
                    f"Unknown U-net tower type '{type}'. Expected 'v2' or 'borzoi/v1'."
                )

            skip_match = block_kwargs.get("skip_match", True)
            other_kw = {k: v for k, v in block_kwargs.items() if k != "skip_match"}
            self.layers.append(
                UnetBorzoiBlock(
                    unet_channels=unet_channels[ri],
                    **shared,
                    **other_kw,
                    skip_match=skip_match and ri == 0,
                )
            )

    def forward(self, x: Tensor, us: [Tensor], di: Optional[int] = None) -> Tensor:
        """
        Args:
            x : Input tensor of shape (N, C, L)
            us : Higher-resolution tensors of shape (N, C, ... x L), ..., (N, C, 2 x L)
            di : Input species index

        Returns:
            Output tensor
        """
        ui = 1
        for layer in self.layers:
            if self.grad_checkpoint:
                x = checkpoint(layer, x, us[-ui], di, use_reentrant=False)
            else:
                x = layer(x, us[-ui], di)
            ui += 1

        return x


class UnetBlock(nn.Module):
    """
    Preferred upsampling U-net block.

    Args:
        channels: Number of input/output channels.
        unet_channels: Number of channels in the U-net connection.
        norm_type: Type of normalization.
        act_func: Name of the activation function.
        kernel_size: Size of the convolutional kernel.
        dropout: Dropout probability.
        num_species (int): Number of species / datasets.
        x_residual (bool): Residual conv block for the main x.
        x_scale (bool): Rescale main x.
        unet_residual (bool): Residual conv block for the U-net skip.
        final_residual (bool): Make final conv block residual.
    """

    def __init__(
        self,
        channels: int,
        unet_channels: int,
        norm_type: str = "batch",
        act_func: str = "silu",
        kernel_size: int = 3,
        dropout: float = 0,
        num_species: int = 2,
        x_residual: bool = False,
        x_scale: bool = True,
        unet_residual: bool = False,
        final_residual: bool = True,
        **_,
    ):
        super().__init__()
        self.x_residual = x_residual
        self.unet_residual = unet_residual
        self.final_residual = final_residual
        self.act = Activation(act_func)

        # optionally refine the low-resolution path before upsampling
        if self.x_residual:
            self.normx = Norm(norm_type, in_dim=channels, num_species=num_species)
            self.convx = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding="same",
                bias=True,
            )
            nn.init.kaiming_normal_(
                self.convx.weight, mode="fan_in", nonlinearity="relu"
            )
        self.scalex = Scale(0.9) if x_scale else nn.Identity()

        # project the skip connection into the working channel dimension
        self.normu = Norm(norm_type, in_dim=unet_channels, num_species=num_species)
        self.convu = nn.Conv1d(
            in_channels=unet_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=True,
        )
        nn.init.kaiming_normal_(self.convu.weight, mode="fan_in", nonlinearity="relu")
        self.pad = channels - unet_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # refine the merged representation with a depthwise-pointwise block
        self.normf = Norm(norm_type, in_dim=channels, num_species=num_species)
        self.convf_depth = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.convf_depth.weight, mode="fan_in", nonlinearity="relu"
        )
        self.convf_point = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=True,
        )
        nn.init.kaiming_normal_(
            self.convf_point.weight, mode="fan_in", nonlinearity="relu"
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, u: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Args:
            x : Input tensor of shape (N, C, L)
            u : Higher-resolution tensor of shape (N, C, 2L)
            di : Input species index

        Returns:
            Output tensor
        """
        if self.x_residual:
            xi = x
            x = self.convx(self.act(self.normx(x, di)))
            x = torch.add(x, xi)
        x = self.scalex(x)
        x = self.upsample(x)

        if self.unet_residual:
            ui = nn.functional.pad(u, (0, 0, 0, self.pad, 0, 0), "constant", 0)
        u = self.convu(self.act(self.normu(u, di)))
        if self.unet_residual:
            u = torch.add(u, ui)

        x = torch.add(x, u)

        if self.final_residual:
            xi = x
        x = self.act(self.normf(x, di))
        x = self.convf_depth(x)
        x = self.convf_point(x)
        x = self.dropout(x)
        if self.final_residual:
            x = torch.add(x, xi)

        return x


@torch._dynamo.disable
class BorzoiTrunk(nn.Module):
    """
    Borzoi trunk PyTorch implementation (with flash attention option) from Gagneur lab
    via https://github.com/johahi/borzoi-pytorch. Weights initialized to those from
    original Borzoi paper.

    Args:
        use_flash_attn: Use flash attention (requires flash_attn package)
        replicate_index: Which of the 4 replicate models to use [0,1,2,3]
    """

    def __init__(
        self,
        use_flash_attn: bool = False,
        replicate_index: int = 0,
        pool_size: int = 32,
        crop_size: int = 5120,
    ):
        super().__init__()

        # check borzoi_pytorch is installed
        if "borzoi_pytorch" not in globals():
            raise ImportError(
                "User trying to use BorzoiTrunk block which requires borzoi_pytorch. "
                "Please install the borzoi_pytorch package from "
                "https://github.com/johahi/borzoi-pytorch."
            )

        # check flash attention is installed if user wants to use it
        if use_flash_attn and "flash_attn" not in globals():
            raise ImportError(
                "User trying to use BorzoiTrunk with flash attention which requires flash_attn. "
                "Please install the flash_attn package. "
            )

        # assert the replicate index is in [0,1,2,3]
        assert replicate_index in [0, 1, 2, 3], "replicate_index must be in [0,1,2,3]"

        # in practice, pool size and crop size are fixed and have no use...just specified
        # for parsimony with seqnn.py
        assert pool_size == 32, "BorzoiTrunk requires pool_size=32"
        assert crop_size == 5120, "BorzoiTrunk requires crop_size=5120"

        # ** init instance variables **
        self.use_flash_attn = use_flash_attn
        self.replicate_index = replicate_index

        # ** load borzoi (human) **
        borzoi = borzoi_pytorch.Borzoi.from_pretrained(
            f"johahi/{'flashzoi' if use_flash_attn else 'borzoi'}-replicate-{replicate_index}"
        )

        # delete non trunk parameters / objects
        try:
            del borzoi.human_head
        except:
            pass
        try:
            del borzoi.final_softplus
        except:
            pass

        # set trunk
        self.borzoi = borzoi

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        # This entire class is excluded from compilation via @torch._dynamo.disable
        # to avoid CUDA memory issues with FlashAttention and complex operations
        x = self.borzoi.get_embs_after_crop(x)
        x = self.borzoi.final_joined_convs(x)
        return x


@torch._dynamo.disable
class BorzoiHead(nn.Module):
    """
    Borzoi head PyTorch implementation (with flash attention option) from Gagneur lab
    via https://github.com/johahi/borzoi-pytorch. Weights initialized to those from
    original Borzoi paper.

    Args:
        use_flash_attn: Use flash attention (requires flash_attn package)
        replicate_index: Which of the 4 replicate models to use [0,1,2,3]
        use_human: Boolean for whether or not to use the human or mouse head.
        in_channels: Number of input channels (enforced as 1920)
        out_channels: Number of output channels (enforced 7611 for human, 2608 for mouse)
    """

    def __init__(
        self,
        use_flash_attn: bool = False,
        replicate_index: int = 0,
        use_human: bool = True,
        in_channels: int = 1920,
        out_channels: int = 7611,  # 7611 for human, 2608 for mouse
    ):
        super().__init__()

        # check borzoi_pytorch is installed
        if "borzoi_pytorch" not in globals():
            raise ImportError(
                "User trying to use BorzoiHead block which requires borzoi_pytorch. "
                "Please install the borzoi_pytorch package from "
                "https://github.com/johahi/borzoi-pytorch."
            )

        # check flash attention is installed if user wants to use it
        if use_flash_attn and "flash_attn" not in globals():
            raise ImportError(
                "User trying to use BorzoiHead with flash attention which requires flash_attn. "
                "Please install the flash_attn package. "
            )

        # assert the replicate index is in [0,1,2,3]
        assert replicate_index in [0, 1, 2, 3], "replicate_index must be in [0,1,2,3]"

        # ** init instance variables **
        self.use_flash_attn = use_flash_attn
        self.replicate_index = replicate_index
        self.use_human = use_human
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ** toss error if in/out channels don't match pretrained model **
        assert self.in_channels == 1920, "BorzoiHead requires in_channels=1920"
        if self.use_human:
            assert self.out_channels == 7611, (
                "BorzoiHead with use_human=True requires out_channels=7611"
            )
        else:
            assert self.out_channels == 2608, (
                "BorzoiHead with use_human=False requires out_channels=2608"
            )

        # ** load borzoi **
        model_string = f"johahi/{'flashzoi' if use_flash_attn else 'borzoi'}-replicate-{replicate_index}"
        if not self.use_human:
            model_string += "-mouse"
        borzoi = borzoi_pytorch.Borzoi.from_pretrained(model_string)

        # ** get the head **
        import copy

        self.head = (
            copy.deepcopy(borzoi.human_head)
            if self.use_human
            else copy.deepcopy(borzoi.mouse_head)
        )
        self.final_softplus = copy.deepcopy(borzoi.final_softplus)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        # This entire class is excluded from compilation via @torch._dynamo.disable
        # to avoid CUDA memory issues with FlashAttention and complex operations
        x = self.final_softplus(self.head(x.float()))
        return x


############################################################
# Dictionary
############################################################
name_module = {
    "SpeciesEmbedding": SpeciesEmbedding,
    "BorzoiTrunk": BorzoiTrunk,
    "BorzoiHead": BorzoiHead,
    "ConvDNA": ConvDNA,
    "ConvBlock": ConvBlock,
    "ConvTower": ConvTower,
    "Crop": Crop,
    "Final": Final,
    "FinalBorzoi": FinalBorzoi,
    "FinalGene": FinalGene,
    "HydraBlock": HydraBlock,
    "HydraTower": HydraTower,
    "MBConvBlock": MBConvBlock,
    "LinearBlock": LinearBlock,
    "TransformerBlock": TransformerBlock,
    "TransformerTower": TransformerTower,
    "UnetBorzoiBlock": UnetBorzoiBlock,
    "UnetBlock": UnetBlock,
    "UnetTower": UnetTower,
    "UnetV2Block": UnetBlock,
    "UnetV2Tower": UnetTower,
}
# append deprecated modules
for deprec_name in deprec_name_module:
    name_module[deprec_name] = deprec_name_module[deprec_name]

# special block flags (e.g. to pass additional args in seqnn forward call)
SPECIES_ARG_FLAG = "species_arg"  # pass species index as additional arg
CONV_REP_FLAG = "conv_rep"  # pass conv rep as extra arg (unet block)
CONV_RET_FLAG = "conv_ret"  # get conv rep as extra return val (conv tower)
CONV_REPS_FLAG = "conv_reps"  # pass conv rep as extra arg (unet tower)
GENE_ARG_FLAG = (
    "gene_arg"  # pass gene metadata (gene_out_mask, gene_presence) as additional args
)

name_flag = {
    "ConvDNA": [SPECIES_ARG_FLAG],
    "SpeciesEmbedding": [SPECIES_ARG_FLAG],
    "ConvTower": [SPECIES_ARG_FLAG, CONV_RET_FLAG],
    "UnetBorzoiBlock": [SPECIES_ARG_FLAG, CONV_REP_FLAG],
    "UnetBlock": [SPECIES_ARG_FLAG, CONV_REP_FLAG],
    "UnetTower": [SPECIES_ARG_FLAG, CONV_REPS_FLAG],
    "UnetV2Block": [SPECIES_ARG_FLAG, CONV_REP_FLAG],
    "UnetV2Tower": [SPECIES_ARG_FLAG, CONV_REPS_FLAG],
    "FinalGene": [GENE_ARG_FLAG],
}
# append deprecated modules
for deprec_name in deprec_name_flag:
    name_flag[deprec_name] = deprec_name_flag[deprec_name]

# add defaults
for name in name_module:
    if name not in name_flag:
        name_flag[name] = []

# create module init signatures
name_init = {}
for name in name_module:
    name_init[name] = signature(name_module[name].__init__)

torch_module = {"Conv1D": nn.Conv1d, "Linear": nn.Linear}
