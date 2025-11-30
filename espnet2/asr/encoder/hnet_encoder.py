from espnet2.asr.encoder.abs_encoder import AbsEncoder
import torch
import torch.nn as nn
from typing import Optional, Tuple
from typeguard import typechecked
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.encoder.mamba import Mamba, ModelArgs, ResidualBlock
# from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding



class HNetEncoder(AbsEncoder):
    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        hidden_size: int = 256,
        # Architecture depth configuration
        num_encoder_layers: int = 2,  # Depth of epsilon (E)
        num_main_layers: int = 2,     # Depth of Main (M)
        num_decoder_layers: int = 2,  # Depth of Delta (D)
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

        # 1. Input Projection
        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            # RelPositionalEncoding(output_size, positional_dropout_rate, max_pos_emb_len),
        )

        # 2. Encoder (E) - Fine-grained processing
        model_args = ModelArgs(d_model=hidden_size, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.encoder_layers = nn.ModuleList([
            ResidualBlock(model_args)
            for _ in range(num_encoder_layers)
        ])
        self.norm_enc = nn.LayerNorm(hidden_size)

        # 3. Chunking Layer (Downsampling)
        # Reduces Sequence Length by factor R
        self.chunking_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=downsample_rate, stride=downsample_rate),
            nn.BatchNorm1d(hidden_size)
        )

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

        # 5. Dechunking Layer (Upsampling)
        # Restores Sequence Length
        self.dechunking_proj = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=downsample_rate, stride=downsample_rate),
            nn.BatchNorm1d(hidden_size)
        )

        # 6. Decoder (D) - Fine-grained reconstruction
        self.decoder_layers = nn.ModuleList([
            ResidualBlock(model_args)
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
        print(f"STARTING FORWARD.....{xs_pad.shape}")
        B, T, _ = xs_pad.shape
        device = xs_pad.device

        # -------------------------
        # Compute input mask: (B, T)
        # -------------------------
        if masks is None:
            mask = ~make_pad_mask(ilens).to(device)
        else:
            mask = ~masks.to(device)

        # -------------------------
        # 1. Input embedding
        # -------------------------
        x = self.embed(xs_pad)  # (B, T, D)
        print(f"EMBEDDING........{x.shape}")

        # -------------------------
        # 2. Encoder (E) — Mamba w/ mask
        # -------------------------
        for layer in self.encoder_layers:
            x, mask = layer(x, mask=mask)

        x = self.norm_enc(x)
        T_orig = x.size(1)

        # -------------------------
        # 3. Chunking
        # -------------------------
        R = self.downsample_rate
        if T_orig % R != 0:
            pad_len = R - (T_orig % R)
            x = F.pad(x, (0, 0, 0, pad_len))  # pad time
            mask = F.pad(mask, (0, pad_len), value=False)
        else:
            pad_len = 0

        x_chunk = self.chunking_proj(x.transpose(1, 2)).transpose(1, 2)  # (B, T/R, D)

        ilens_sub = torch.ceil(ilens / R).long()
        mask_sub = ~make_pad_mask(ilens_sub).to(device)  # (B, T/R)

        # -------------------------
        # 4. Main Transformer (M)
        # -------------------------
        x_main = self.main_layers(
            x_chunk,
            src_key_padding_mask=~mask_sub  # Transformer expects False=keep, True=pad
        )
        x_main = self.norm_main(x_main)

        # -------------------------
        # 5. Dechunking
        # -------------------------
        x_up = self.dechunking_proj(x_main.transpose(1, 2)).transpose(1, 2)

        # Trim or pad to T_orig + pad_len
        T_exp = T_orig + pad_len
        if x_up.size(1) > T_exp:
            x_up = x_up[:, :T_exp, :]
        elif x_up.size(1) < T_exp:
            x_up = F.pad(x_up, (0, 0, 0, T_exp - x_up.size(1)))

        # Remove padding
        x_up = x_up[:, :T_orig, :]
        mask_up = mask[:, :T_orig]

        # -------------------------
        # 6. Decoder (D) — Mamba w/ mask
        # -------------------------
        x_dec = x_up
        for layer in self.decoder_layers:
            x_dec, mask_up = layer(x_dec, mask=mask_up)

        x_dec = self.norm_dec(x_dec)

        # -------------------------
        # Residual + Final Linear
        # -------------------------
        xs_emb = self.embed(xs_pad)
        x_final = x_dec + xs_emb
        out = self.output_proj(x_final)

        return out, ilens, None