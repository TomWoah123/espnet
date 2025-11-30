"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int = 256
    n_layer: int = 4
    vocab_size: int = 256
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, xs_pad, masks=None, olens=None):
        """
        Args:
            xs_pad : (B, T) long → token IDs
            masks  : (B, 1, T) or (B, T) boolean
            olens  : (B,) original lengths

        Returns:
            xs_pad_out : (B, T, D)
            olens      : (B,)
        """
        # ----- EMBEDDING -----
        x = self.embedding(xs_pad)

        # If mask provided, normalize shape to (B, 1, T)
        if masks is not None and masks.dim() == 2:
            masks = masks[:, None, :]

        # ----- ENCODER -----
        for layer in self.layers:
            # pass masks even if unused internally (future-proofing)
            x, masks = layer(x, mask=masks)

        # ----- FINAL NORM -----
        x = self.norm_f(x)

        # ----- DECODER (LM HEAD) -----
        # logits = self.lm_head(x)  # If needed, uncomment for LM-style output
        # return logits

        # You requested: return xs_pad, olens
        return x, olens

    
    # @staticmethod
    # def from_pretrained(pretrained_model_name: str):
    #     """Load pretrained weights from HuggingFace into model.
    
    #     Args:
    #         pretrained_model_name: One of
    #             * 'state-spaces/mamba-2.8b-slimpj'
    #             * 'state-spaces/mamba-2.8b'
    #             * 'state-spaces/mamba-1.4b'
    #             * 'state-spaces/mamba-790m'
    #             * 'state-spaces/mamba-370m'
    #             * 'state-spaces/mamba-130m'
                            
    #     Returns:
    #         model: Mamba model with weights loaded
    
    #     """
    #     from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    #     from transformers.utils.hub import cached_file
        
    #     def load_config_hf(model_name):
    #         resolved_archive_file = cached_file(model_name, CONFIG_NAME,
    #                                             _raise_exceptions_for_missing_entries=False)
    #         return json.load(open(resolved_archive_file))
        
        
    #     def load_state_dict_hf(model_name, device=None, dtype=None):
    #         resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
    #                                             _raise_exceptions_for_missing_entries=False)
    #         return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
    #     config_data = load_config_hf(pretrained_model_name)
    #     args = ModelArgs(
    #         d_model=config_data['d_model'],
    #         n_layer=config_data['n_layer'],
    #         vocab_size=config_data['vocab_size']
    #     )
    #     model = Mamba(args)
        
    #     state_dict = load_state_dict_hf(pretrained_model_name)
    #     new_state_dict = {}
    #     for key in state_dict:
    #         new_key = key.replace('backbone.', '')
    #         new_state_dict[new_key] = state_dict[key]
    #     model.load_state_dict(new_state_dict)
        
    #     return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x, mask=None):
        # `self.mixer` returns a tuple `(out_tensor, mask)`; unpack it
        out, mask = self.mixer(self.norm(x), mask=mask)
        if mask is not None:
            out = out * mask.unsqueeze(-1)  # (B, T, 1)
        return out + x, mask
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            args.d_inner,
            args.d_inner,
            kernel_size=args.d_conv,
            padding=args.d_conv - 1,
            groups=args.d_inner,
            bias=args.conv_bias,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x, mask=None):
        (b, l, d) = x.shape

        # -------- Mask input --------
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x_proj = self.in_proj(x)
        x_inner, res = x_proj.chunk(2, dim=-1)

        # -------- Conv1d --------
        x_inner = rearrange(x_inner, 'b l d -> b d l')
        x_inner = self.conv1d(x_inner)[..., :l]
        x_inner = rearrange(x_inner, 'b d l -> b l d')
        x_inner = F.silu(x_inner)

        # -------- SSM (mask-aware) --------
        y, mask = self.ssm(x_inner, mask)

        # Selective output gate
        y = y * F.silu(res)

        # Mask output
        if mask is not None:
            y = y * mask.unsqueeze(-1)

        return self.out_proj(y), mask

    def ssm(self, x, mask=None):
        (b, l, d_in) = x.shape
        A = -torch.exp(self.A_log)
        n = A.shape[1]
        D = self.D

        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # mask delta and B/C
        if mask is not None:
            m = mask.unsqueeze(-1)
            delta = delta * m
            B = B * m
            C = C * m

        return self.selective_scan(x, delta, A, B, C, D, mask)

    def selective_scan(self, u, delta, A, B, C, D, mask):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []

        for t in range(l):
            if mask is not None:
                m = mask[:, t].view(b, 1, 1)  # (B,1,1)
            else:
                m = 1.0

            # masked timesteps do NOT update state
            x = x * m + (deltaA[:, t] * x + deltaB_u[:, t]) * m

            y = einsum(x, C[:, t], 'b d_in n, b n -> b d_in')
            ys.append(y)

        y = torch.stack(ys, dim=1)
        return y + u * D, mask



class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output