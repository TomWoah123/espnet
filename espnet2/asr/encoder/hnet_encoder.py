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
from hnet_modules.components import ChunkLayer, DeChunkLayer
from hnet_modules.multilayer_perceptron import SwiGLU
from mamba_ssm import Mamba

class HNetEncoder(AbsEncoder):
    @typechecked
    def __init__(
        self,
        device=None,
        dtype=None
    ):
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer()
        self.main_network = SwiGLU()
        self.encoder = Mamba()
        self.decoder = Mamba()

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