from espnet2.asr.encoder.abs_encoder import AbsEncoder
import torch
import torch.nn as nn
from typing import Optional, Tuple
from typeguard import typechecked
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
## IMPORT MAMBA HERE
from espnet2.asr.encoder.mamba_encoder import MambaEncoder

## REPLACE BELOW WITH ACTUAL mamba_ssm stuff
from espnet2.asr.encoder.dynamic_chunking import DeChunkLayer, DeChunkState, ChunkLayer, RoutingModule, RoutingModuleOutput, RoutingModuleState
from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling, check_short_utt, TooShortUttError
from dataclasses import dataclass, field
from typing import Union, Optional
import optree

@dataclass
class IsotropicInferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

        optree.tree_map(
            lambda x: x.zero_() if isinstance(x, torch.Tensor) else x,
            self.key_value_memory_dict,
        )

@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)



class HNetEncoder(AbsEncoder):
    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        hidden_size: int = 256,
        d_model: int = 256,
        # Architecture depth configuration
        num_encoder_layers: int = 2,  # Depth of epsilon (E)
        num_main_layers: int = 2,     # Depth of Main (M)
        # Hierarchy configuration
        downsample_rate: int = 4,
        # Mamba configuration
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.1,
        transformer_ffn_dim: int = 1024,
        num_heads: int = 8,
        max_pos_emb_len: int = 5000,
        positional_dropout_rate: float = 0.1
    ):
        """
        Pipeline:
        encoder → chunking → main network → dechunking → decoder
        """
        super().__init__()
        self._output_size = output_size
        self.downsample_rate = downsample_rate
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.embed = Conv2dSubsampling(
            input_size,
            d_model,
            positional_dropout_rate,
            RelPositionalEncoding(d_model, positional_dropout_rate, max_pos_emb_len),
        )

        # self.encoder = Mamba(....)
        self.routing_module = RoutingModule(d_model=d_model)
        self.chunking_layer = ChunkLayer()

        # 4. Main Network (M) - Coarse-grained processing (Bottleneck)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
            activation="gelu",
            norm_first=True,
        )
        self.main_layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_main_layers
        )
        self.norm_main = nn.LayerNorm(hidden_size)
        self.dechunking_layer = DeChunkLayer(d_model=d_model)
        self.residual_proj = nn.Linear(
            self.d_model, self.d_model, dtype=torch.float32
        )
        nn.init.zeros_(self.residual_proj.weight)
        self.residual_proj.weight._no_reinit = True

        self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
    
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        ctc=None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = xs_pad.shape
        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]
        
        short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
        if short_status:
            raise TooShortUttError(
                f"has {xs_pad.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                xs_pad.size(1),
                limit_size,
            )

        # 1. Input embedding
        xs_pad, masks = self.embed(xs_pad, masks)
        xs_pad = xs_pad[0]
        # hs_all = []
        # CALL MAMBA ENCODER HERE
        #..........
        
        xs_pad_hs_for_residual = xs_pad.to(
            dtype=self.residual_proj.weight.dtype
        )
        xs_pad_residual = self.residual_proj(xs_pad_hs_for_residual)
        if masks is None:
            masks = (~make_pad_mask(olens)).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]
        # print(f"XS_PAD.............{xs_pad.shape}, MASKS..................{masks.shape}")
        bpred_output = self.routing_module(
            xs_pad,
            cu_seqlens=None,
            mask=masks,
            inference_params=inference_params.routing_module_state,
        )
        boundary_mask = bpred_output.boundary_mask.squeeze(1)
        xs_pad, next_cu_seqlens, next_max_seqlen, next_mask = self.chunking_layer(
            xs_pad, boundary_mask, None, mask=masks
        )
        # print(f"AFTER CHUNKING.............{xs_pad.shape}")
        xs_pad = self.main_layers(
            xs_pad,
            # src_key_padding_mask=next_mask  # Transformer expects False=keep, True=pad
        )
        xs_pad = self.norm_main(xs_pad)
        xs_pad = self.dechunking_layer(
            xs_pad,
            boundary_mask,
            bpred_output.boundary_prob,
            next_cu_seqlens,
            mask=masks,
            inference_params=inference_params.dechunk_state,
        )
        xs_pad = self.residual_func(
            xs_pad.to(dtype=xs_pad_residual.dtype), xs_pad_residual, bpred_output.selected_probs
        ).to(xs_pad.dtype)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
        
