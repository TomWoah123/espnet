from espnet2.asr.encoder.abs_encoder import AbsEncoder
import torch
import torch.nn as nn
from typing import Optional, Tuple
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class HNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4)

    def forward(self, x, mask=None, prev_state=None):
        out = self.layer(x, src_key_padding_mask=mask)
        return out, prev_state


def chunk_sequence(xs: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, D = xs.shape
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        xs = torch.cat([xs, xs.new_zeros(B, pad_len, D)], dim=1)
    T2 = xs.shape[1]
    n_chunks = T2 // chunk_size
    return xs.view(B, n_chunks, chunk_size, D)


def dechunk_sequence(xs_chunked: torch.Tensor, original_length: int) -> torch.Tensor:
    B, N, S, D = xs_chunked.shape
    xs = xs_chunked.reshape(B, N * S, D)
    return xs[:, :original_length, :]


class HNetEncoder(AbsEncoder):
    @typechecked
    def __init__(
        self,
        idim: int = 80,
        enc_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        chunk_size: Optional[int] = None,
        out_dim: Optional[int] = None,
    ):
        """
        Pipeline:
        encoder → chunking → main network → dechunking → decoder
        """
        super().__init__()

        self.chunk_size = chunk_size

        self.input_linear = nn.Linear(idim, enc_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(20000, enc_dim)

        self.layers = nn.ModuleList([HNetBlock(enc_dim) for _ in range(num_layers)])

        if out_dim is None:
            out_dim = enc_dim
        self.decoder = nn.Linear(enc_dim, out_dim)
    
    def output_size(self) -> int:
        return self.out_dim

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
        original_T = T

        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]
        key_padding_mask = ~masks.squeeze(1)  # (B, T)

        # Positional encoding + linear
        xs = self.input_linear(xs_pad)
        xs = self.dropout(xs)

        pos_ids = torch.arange(T, device=xs.device).unsqueeze(0).expand(B, T)
        xs = xs + self.pos_emb(pos_ids)

        if self.chunk_size is not None:
            xs_chunked = chunk_sequence(xs, self.chunk_size)
            mask_chunked = chunk_sequence(
                key_padding_mask.unsqueeze(-1).float(), self.chunk_size
            ).squeeze(-1).bool()

            # Flatten (B*N, S, D)
            B, N, S, D = xs_chunked.shape
            xs_chunked = xs_chunked.reshape(B * N, S, D)
            mask_chunked = mask_chunked.reshape(B * N, S)
        else:
            xs_chunked = xs
            mask_chunked = key_padding_mask

        states = prev_states
        for layer in self.layers:
            xs_chunked, states = layer(xs_chunked, mask=mask_chunked, prev_state=states)

        if self.chunk_size is not None:
            xs_chunked = xs_chunked.reshape(B, N, S, D)
            xs = dechunk_sequence(xs_chunked, original_T)
        else:
            xs = xs_chunked
        xs = self.decoder(xs)

        return xs, key_padding_mask
