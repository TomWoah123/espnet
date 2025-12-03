import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, List
from einops import rearrange, repeat, einsum
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
import torch.nn.functional as F
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling, check_short_utt, TooShortUttError
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    
class MambaBlock(nn.Module):
    def __init__(self, 
        d_model: int = 256,
        n_layer: int = 2,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int = "auto",
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_inner = int(expand * d_model)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Parallelized selective scan (no sequential Python loop)."""

        # Shapes
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # -------------------------
        # 1. Discretize parameters
        # -------------------------
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # -------------------------
        # 2. Parallel prefix scan
        #    over (A_i, B_i)
        #
        # Recurrence per step:
        #   x_i = A_i * x_(i-1) + B_i
        #
        # Associative operator:
        #   (A2, B2) ⊙ (A1, B1) = (A2*A1 , B2 + A2*B1)
        # -------------------------

        Avals = deltaA                       # (b, l, d_in, n)
        Bvals = deltaB_u                     # (b, l, d_in, n)

        step = 1
        while step < l:
            # Shifted A/B values
            Ashift = torch.roll(Avals, shifts=step, dims=1)
            Bshift = torch.roll(Bvals, shifts=step, dims=1)

            # Zero-out invalid prefix regions
            Ashift[:, :step] = 1.0           # multiplicative identity
            Bshift[:, :step] = 0.0           # additive identity

            # Combine using associative scan operator
            # newA = A_i * A_(i-step)
            # newB = B_i + A_i * B_(i-step)
            newA = Avals * Ashift
            newB = Bvals + Avals * Bshift

            Avals, Bvals = newA, newB
            step *= 2

        # After scan:  Bvals[b, i] = x_i
        X = Bvals                              # (b, l, d_in, n)

        # -------------------------
        # 3. Compute output y_i = C_i @ x_i
        # -------------------------
        y = (X * C.unsqueeze(2)).sum(dim=-1)   # (b, l, d_in)

        # -------------------------
        # 4. Add skip connection
        # -------------------------
        y = y + u * D                          # shape (b, l, d_in)

        return y
    
class ResidualBlock(nn.Module):
    def __init__(self,
        d_model: int = 256,
        n_layer: int = 2,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int = "auto",
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(
            d_model=d_model, n_layer=n_layer, d_state=d_state, expand=expand,
            dt_rank=dt_rank, d_conv=d_conv, conv_bias=conv_bias, bias=bias
        )

    def forward(self, x):
        return x + self.mixer(self.norm(x))
    
class MambaEncoder(AbsEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        d_model: int = 256,
        n_layer: int = 2,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: int = "auto",
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        positional_dropout_rate: float = 0.1,
        max_pos_emb_len: int = 5000,
    ):
        super().__init__()
        self._output_size = output_size

        # self.embed = nn.Linear(input_size, args.d_model)
        self.embed = Conv2dSubsampling(
            input_size,
            d_model,
            positional_dropout_rate,
            RelPositionalEncoding(d_model, positional_dropout_rate, max_pos_emb_len),
        )

        self.layers = nn.ModuleList([ResidualBlock(d_model=d_model, n_layer=n_layer, d_state=d_state, expand=expand,
            dt_rank=dt_rank, d_conv=d_conv, conv_bias=conv_bias, bias=bias) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
    
    def output_size(self) -> int:
        return self._output_size

    # --------------------------------------------------------
    # Forward (ESPnet signature)
    # --------------------------------------------------------
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        masks: torch.Tensor = None,
        ctc=None,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

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
        hs_all = []

        # 3. Mamba layers
        for i, layer in enumerate(self.layers):
            # ---- Mamba block ----
            xs_pad = layer(xs_pad)

            if return_all_hs:
                hs_all.append(xs_pad)

            if ctc is not None and hasattr(ctc, "intermediate_ctc"):
                ctc.attach_intermediate(i, xs_pad, ilens)

        # 4. Final normalization
        xs_pad = self.norm_f(xs_pad)
        olens = masks.squeeze(1).sum(1)

        if return_all_hs:
            return xs_pad, ilens, hs_all

        return xs_pad, olens, None