from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

def simple_mamba_scan(u, dt, A, B, C, D=None, z=None, initial_state=None):
    """
    A robust Mamba-style scan that matches the caller shapes.

    Args:
        u: (B, L, H, P)  input per time, per head, per head-dim p
        dt: (B, L, H) or broadcastable  time-step scalars
        A: (H,) or (H,1)          per-head scalar (continuous A -> discrete by exp(A * dt))
        B: (B, L, 1, 1) or broadcastable  input gain scalar(s)
        C: (B, L, 1, 1) or broadcastable  output gain scalar(s)
        D: optional (P,) or (1,1,P)       skip weight over u
        z: optional gating (B, L, H, P)   elementwise multiply on state after update
        initial_state: (B, H, P) or None

    Returns:
        y: (B, L, H, P)
    """
    # Validate shapes / coerce
    if u.ndim != 4:
        raise ValueError("u must have shape (B, L, H, P)")

    BATCH, L, H, P = u.shape
    device = u.device
    dtype = u.dtype

    # Ensure dt is (B, L, H)
    dt = dt.to(device=device, dtype=dtype)
    if dt.ndim == 2:
        # maybe (B, L) -> expand to heads
        dt = dt.unsqueeze(-1).expand(BATCH, L, H)
    elif dt.ndim == 1:
        # (L,) -> expand
        dt = dt.unsqueeze(0).unsqueeze(-1).expand(BATCH, L, H)
    # A -> (H,)
    A = A.to(device=device, dtype=dtype).reshape(-1)
    if A.numel() != H:
        # allow scalar A broadcast if needed
        if A.numel() == 1:
            A = A.expand(H)
        else:
            raise ValueError(f"A must have length H={H} or be scalar; got {A.shape}")

    # Broadcast A to (B, L, H) to compute decay
    # A is (H,), dt is (B,L,H) -> decay (B,L,H)
    decay = torch.exp(A.view(1, 1, H) * dt)

    # Prepare B and C to broadcast to (B, L, H, P)
    B_ = B.to(device=device, dtype=dtype)
    C_ = C.to(device=device, dtype=dtype)
    # Expand last dims if necessary
    B_ = B_.reshape(BATCH, L, 1, 1).expand(BATCH, L, H, P)
    C_ = C_.reshape(BATCH, L, 1, 1).expand(BATCH, L, H, P)

    # initial state
    if initial_state is None:
        x = torch.zeros((BATCH, H, P), device=device, dtype=dtype)
    else:
        x = initial_state.to(device=device, dtype=dtype)
        if x.shape != (BATCH, H, P):
            raise ValueError("initial_state must be (B, H, P)")

    outputs = []
    for t in range(L):
        # decay_t shape (B, H) -> expand to (B, H, P)
        decay_t = decay[:, t, :].unsqueeze(-1)  # (B, H, 1)
        u_t = u[:, t, :, :]                       # (B, H, P)
        B_t = B_[:, t, :, :]                      # (B, H, P)

        # x <- decay * x + B_t * u_t
        x = decay_t * x + B_t * u_t               # broadcasting to (B, H, P)

        # optional gating z (must match shape)
        if z is not None:
            z_t = z[:, t, :, :]
            x = x * z_t

        # output y = C_t * x + D * u_t (D broadcasted over P)
        C_t = C_[:, t, :, :]
        y = C_t * x

        if D is not None:
            D_ = D.to(device=device, dtype=dtype)
            # D may be (P,) or (1,1,P); normalize to (P,)
            D_ = D_.reshape(-1) if D_.ndim >= 1 else D_.expand(P)
            y = y + u_t * D_.view(1, 1, P)

        outputs.append(y)

    return torch.stack(outputs, dim=1)  # (B, L, H, P)

def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
    """

    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModule(nn.Module):

    def __init__(self, d_model, device=None, dtype=None):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(self, hidden_states, cu_seqlens=None, mask=None, inference_params=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert (
                ~inference_params.has_seen_tokens
            ).all(), "Cannot have seen tokens when inference_params is not provided"

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = hidden_states.unsqueeze(0)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask,
                    hidden_states[
                        torch.arange(
                            hidden_states.shape[0], device=hidden_states.device
                        ),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )

    def step(self, hidden_states, inference_params):
        # hidden_states is (B, 1, D)
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states)
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (B, 2)
            boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),  # (B, 1)
        )


class ChunkLayer(nn.Module):

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
    ):
        super().__init__()
        self.d_model = d_model

        # Just for Mamba2 kernel.
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is not None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"
            assert boundary_mask[
                :, 0
            ].all(), "First token must be a boundary if running prefill"

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
        else:
            B, L = boundary_mask.shape
            seq_idx = None

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1")
        )
        out = rearrange(out, "b l h p -> b l (h p)")

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(
                out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        # B_selected = hidden_states.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)