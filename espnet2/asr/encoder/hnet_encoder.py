from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    ConvolutionalPositionalEmbedding,
    PositionalEncoding,
)
from hnet_modules.components import ChunkLayer, DeChunkLayer, RoutingModule
from hnet_modules.multilayer_perceptron import SwiGLU
from mamba_ssm import Mamba2

class HNetEncoder(AbsEncoder):
    @typechecked
    def __init__(
        self,
        dechunk_d_model: int = 256,
        routing_module_d_model: int = 256,
        main_network_d_model: int = 256,
        encoder_d_model: int = 256,
        encoder_d_state: int = 64,
        encoder_d_conv: int = 4,
        encoder_expand: int = 2,
        decoder_d_model: int = 256,
        decoder_d_state: int = 64,
        decoder_d_conv: int = 4,
        decoder_expand: int = 2,
        device=None,
        dtype=None
    ):
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(d_model=dechunk_d_model)
        self.main_network = torch.nn.Transformer(d_model=main_network_d_model)
        self.routing_module = RoutingModule(d_model=routing_module_d_model)
        self.encoder = Mamba2(d_model=encoder_d_model, d_state=encoder_d_state, d_conv=encoder_d_conv, expand=encoder_expand)
        self.decoder = Mamba2(d_model=decoder_d_model, d_state=decoder_d_state, d_conv=decoder_d_conv, expand=decoder_expand)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        ctc: CTC = None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]
        D = xs_pad.shape[-1]
        xs_encoded = self.encoder(xs_pad)
        bpred_output = self.routing_module(
            xs_encoded,
            cu_seqlens=ilens,
            mask=masks,
        )
        xs_chunked, next_cu_lens, next_max_seqlen, masks = self.chunk_layer.forward(xs_encoded, bpred_output.boundary_mask)
        xs_network = self.main_network(xs_chunked)
        xs_dechunked = self.dechunk_layer(xs_network, bpred_output.boundary_mask, bpred_output.boundary_prob)
        xs_decoded = self.decoder(xs_dechunked)
        xs_pad = xs_decoded[..., :D]
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None

        
        
