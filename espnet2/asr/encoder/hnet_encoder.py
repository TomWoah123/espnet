from espnet2.asr.encoder.abs_encoder import AbsEncoder
import torch
import torch.nn as nn
from typing import Optional, Tuple
from typeguard import typechecked
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.encoder.mamba import Mamba, ModelArgs


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
        input_size: int,
        output_size: int = 256,
        hidden_size: int = 256,
        # Architecture depth configuration
        num_encoder_layers: int = 2,  # Depth of epsilon (E)
        num_main_layers: int = 4,     # Depth of Main (M)
        num_decoder_layers: int = 2,  # Depth of Delta (D)
        # Hierarchy configuration
        downsample_rate: int = 4,
        # Mamba configuration
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.1,
    ):
        """
        Pipeline:
        encoder → chunking → main network → dechunking → decoder
        """
        super().__init__()
        self._output_size = output_size
        self.downsample_rate = downsample_rate
        self.hidden_size = hidden_size

        # 1. Input Projection
        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        # 2. Encoder (E) - Fine-grained processing
        model_args = ModelArgs(d_model=hidden_size, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.encoder_layers = nn.ModuleList([
            Mamba(model_args)
            for _ in range(num_encoder_layers)
        ])
        self.norm_enc = nn.LayerNorm(hidden_size)

        # 3. Chunking Layer (Downsampling)
        # Reduces Sequence Length by factor R
        self.chunking_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=downsample_rate, stride=downsample_rate),
            nn.LayerNorm(hidden_size)
        )

        # 4. Main Network (M) - Coarse-grained processing (Bottleneck)
        self.main_layers = nn.ModuleList([
            Mamba(model_args)
            for _ in range(num_main_layers)
        ])
        self.norm_main = nn.LayerNorm(hidden_size)

        # 5. Dechunking Layer (Upsampling)
        # Restores Sequence Length
        self.dechunking_proj = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=downsample_rate, stride=downsample_rate),
            nn.LayerNorm(hidden_size)
        )

        # 6. Decoder (D) - Fine-grained reconstruction
        self.decoder_layers = nn.ModuleList([
            Mamba(model_args)
            for _ in range(num_decoder_layers)
        ])
        self.norm_dec = nn.LayerNorm(hidden_size)

        # Final projection to output size
        self.output_proj = nn.Linear(hidden_size, output_size)
    
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
        if masks is None:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        else:
            masks = ~masks[:, None, :]
        
        x_emb = self.embed(xs_pad)  # (B, T, D)
        
        # --- 2. Encoder (E) ---
        x = x_emb
        for layer in self.encoder_layers:
            x = layer(x)
        x_enc = self.norm_enc(x) # (B, T, D)

        # --- 3. Chunking ---
        # Adjust lengths for padding
        B, T, D = x_enc.size()
        if T % self.downsample_rate != 0:
            pad_len = self.downsample_rate - (T % self.downsample_rate)
            x_enc = F.pad(x_enc, (0, 0, 0, pad_len)) # Pad time dimension
        
        # Permute for Conv1d (B, D, T)
        x_permuted = x_enc.transpose(1, 2)
        x_chunked = self.chunking_proj(x_permuted)
        x_chunked = x_chunked.transpose(1, 2) # Back to (B, T_sub, D)
        
        # Update lengths (integer division)
        ilens_sub = torch.ceil(ilens / self.downsample_rate).long()

        # --- 4. Main Network (M) ---
        x_main = x_chunked
        for layer in self.main_layers:
            x_main = layer(x_main)
        x_main = self.norm_main(x_main)

        # --- 5. Dechunking ---
        x_main_permuted = x_main.transpose(1, 2)
        x_dechunked = self.dechunking_proj(x_main_permuted)
        x_dechunked = x_dechunked.transpose(1, 2) # (B, T_restored, D)

        # Crop to match original length (ConvTranspose might produce extra frames)
        if x_dechunked.size(1) > T:
            x_dechunked = x_dechunked[:, :T, :]
        elif x_dechunked.size(1) < T:
            # Should not happen if padding was correct, but safety first
            x_dechunked = F.pad(x_dechunked, (0, 0, 0, T - x_dechunked.size(1)))

        # Remove the padding we added before Chunking if necessary to match x_emb
        original_T = x_emb.size(1)
        x_dechunked = x_dechunked[:, :original_T, :]

        # --- 6. Decoder (D) ---
        x_dec = x_dechunked
        for layer in self.decoder_layers:
            x_dec = layer(x_dec)
        x_dec = self.norm_dec(x_dec)

        # --- 7. Residual Connection & Output ---
        # "Decoder + Encoder" (Skip connection from start to finish)
        x_final = x_dec + x_emb

        # Final Linear Projection
        output = self.output_proj(x_final)

        return output, ilens_sub, masks