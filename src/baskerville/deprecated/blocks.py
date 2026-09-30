from einops import rearrange
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn.modules import mha as flash_attn
except ImportError:
    pass

from baskerville.layers import *


class AugmentSpecies(nn.Module):
    """Augment one-hot DNA sequence with species encoding.

    Args:
        num_species (int): Number of species / datasets
    """

    def __init__(
        self,
        num_species,
    ):
        super().__init__()
        self.num_species = num_species

    def forward(self, x, di):
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, 4, L)
            di : Species integer index

        Returns:
            Output tensor
        """
        enc = torch.tile(  # tile across length
            torch.unsqueeze(
                nn.functional.one_hot(
                    torch.tile(  # tile across batches
                        torch.tensor([di]), (x.shape[0],)
                    ),
                    num_classes=self.num_species,
                ),
                -1,
            ),
            (1, 1, x.shape[2]),
        ).to(dtype=x.dtype, device=x.device)
        x = torch.cat([x, enc], dim=1)
        return x


class UnetBlockGradCheck(nn.Module):
    """
    Upsampling U-net block (gradient checkpointed)

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
        num_species=2,
    ):
        super().__init__()
        from baskerville.blocks import UnetBorzoiBlock

        self.unet = UnetBorzoiBlock(
            channels,
            unet_channels,
            norm_type,
            act_func,
            kernel_size,
            dropout,
            skip_match,
            num_species,
        )

    def forward(self, x: Tensor, u: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Args:
            x : Input tensor of shape (N, C, L)
            u : Higher-resolution tensor of shape (N, C, 2L)
            di : Input species index

        Returns:
            Output tensor
        """
        x = checkpoint(self.unet, x, u, di, use_reentrant=False)

        return x


class FeedForwardSpBlock(nn.Module):
    """
    2-layer feed-forward network with species-specific affine transforms. Can be used to follow layers such as GRU and attention.

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
        num_species: int = 2,
    ) -> None:
        super().__init__()
        from baskerville.blocks import LinearBlock

        expansion_channels = int(channels * expansion)
        self.dense1 = LinearSpBlock(
            channels,
            expansion_channels,
            norm=norm_type,
            dropout=dropout,
            act_func=act_func,
            num_species=num_species,
            bias=True,
        )
        self.dense2 = LinearBlock(
            expansion_channels,
            channels,
            norm=None,
            dropout=dropout,
            act_func=None,
            bias=True,
        )

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        x = self.dense1(x, di)
        x = self.dense2(x)
        return x


class LinearSpBlock(nn.Module):
    """
    Linear layer followed by optional normalization,
    activation and dropout with species-specific affine transforms.

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
        num_species: int = 2,
    ) -> None:
        super().__init__()

        self.norm = Norm(norm, in_dim=in_channels, elementwise_affine=False)
        self.norm_affine = SpeciesEmbedding(
            in_channels, num_species=num_species, axis=-2, scale=True
        )
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = Activation(act_func)

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, L, C)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        x = self.norm_affine(x, di)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.act(x)
        return x


class TransformerSpEnformerBlock(nn.Module):
    """
    A block containing a multi-head attention layer followed by a feed-forward
    network and residual connections with species-specific affine transforms.

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
        num_species: int = 2,
    ) -> None:
        super().__init__()
        self.norm = Norm(norm_type, channels, elementwise_affine=False)
        self.norm_affine = SpeciesEmbedding(
            channels, num_species=num_species, axis=-2, scale=True
        )
        self.mha = AttentionEnformer(
            channels=channels,
            heads=heads,
            pos_features=pos_features,
            key_len=key_len,
            pos_dropout=pos_dropout,
            attn_dropout=attn_dropout,
            shift_method=shift_method,
        )
        self.dropout = nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity()
        self.ffn = FeedForwardSpBlock(
            channels=channels,
            expansion=expansion,
            dropout=ff_dropout,
            act_func=act_func,
            norm_type=norm_type,
            num_species=num_species,
        )

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.norm_affine(x, di)
        x = self.mha(x)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        ffn_input = x
        x = self.ffn(x, di)
        x = torch.add(ffn_input, x)
        return x


class TransformerSpTower(nn.Module):
    """
    Multiple stacked transformer encoder layers with species-specific affine transforms.

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
            transformer_block = TransformerSpEnformerBlock
        else:
            transformer_block = TransformerSpBlock
        self.blocks = nn.ModuleList(
            [transformer_block(**kwargs) for _ in range(repeat)]
        )
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)
            di : Input species index

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        for block in self.blocks:
            if self.grad_checkpoint:
                x = checkpoint(block, x, di, use_reentrant=False)
            else:
                x = block(x, di)
        x = rearrange(x, "b l t -> b t l")
        return x


class HydraSpBlock(nn.Module):
    """
    Residual Hydra block with species-specific affine transforms.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0,
        hydra_type: str = "hydra",
        num_species: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.norm = Norm("rms", channels, elementwise_affine=False)
        self.norm_affine = SpeciesEmbedding(
            channels, num_species=num_species, axis=-2, scale=True
        )
        self.hydra_type = hydra_type
        if self.hydra_type == "hydra":
            self.hydra = Hydra(
                d_model=channels,
                **kwargs,
            )
        elif self.hydra_type == "hydra_sp":
            self.hydra = HydraSp(
                d_model=channels,
                num_species=num_species,
                **kwargs,
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)
            di : Input species index

        Returns:
            Output tensor
        """
        x_input = x
        x = self.norm(x)
        x = self.norm_affine(x, di)
        if self.hydra_type == "hydra":
            x = self.hydra(x)
        elif self.hydra_type == "hydra_sp":
            x = self.hydra(x, di)
        x = self.dropout(x)
        x = torch.add(x_input, x)
        return x


class HydraSpTower(nn.Module):
    """
    Multiple stacked species-specific Hydra layers.

    Args:
        repeat: Number of HydraBlock layers
    """

    def __init__(
        self,
        repeat: int,
        grad_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([HydraSpBlock(**kwargs) for _ in range(repeat)])
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: Tensor, di: Optional[int] = None) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)
            di : Input species index

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        for block in self.blocks:
            if self.grad_checkpoint:
                x = checkpoint(block, x, di, use_reentrant=False)
            else:
                x = block(x, di)
        x = rearrange(x, "b l t -> b t l")
        return x


############################################################
# Dictionary
############################################################
deprec_name_module = {
    "AugmentSpecies": AugmentSpecies,
    "UnetBlockGradCheck": UnetBlockGradCheck,
    "HydraSpBlock": HydraSpBlock,
    "HydraSpTower": HydraSpTower,
    "TransformerSpTower": TransformerSpTower,
}

# special block flags (e.g. to pass additional args in seqnn forward call)
SPECIES_ARG_FLAG = "species_arg"  # pass species index as additional arg
CONV_REP_FLAG = "conv_rep"  # pass conv rep as extra arg (unet block)
CONV_RET_FLAG = "conv_ret"  # get conv rep as extra return val (conv tower)
CONV_REPS_FLAG = "conv_reps"  # pass conv rep as extra arg (unet tower)

deprec_name_flag = {
    "AugmentSpecies": [SPECIES_ARG_FLAG],
    "UnetBlockGradCheck": [SPECIES_ARG_FLAG, CONV_REP_FLAG],
    "HydraSpTower": [SPECIES_ARG_FLAG],
    "TransformerSpTower": [SPECIES_ARG_FLAG],
}

# add defaults
for name in deprec_name_module:
    if name not in deprec_name_flag:
        deprec_name_flag[name] = []
