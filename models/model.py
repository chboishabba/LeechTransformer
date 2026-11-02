"""
🎯 Leech-Lila DOI: 10.5281/zenodo.18784424
This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
Commercial Licensing: For proprietary R&D, integration into private AI stacks, or hardware implementation,
please contact the Architect directly.
Copyright (C) 2026 Anatolii Kornienko This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/agpl-3.0.txt/>.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.history_orbit_moe import HistoryOrbitMoE
from models.e8_shared_orbit_history_moe import E8SharedOrbitHistoryMoE
from models.dense_to_signed_experts import canonical_even_input


@dataclass
class LeechConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 8
    block_size: int = 512
    dropout: float = 0.05
    bias: bool = False
    tie_weights: bool = True
    lambda_geo: float = 0.01
    resonance_threshold: float = 0.95
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_expert_width: int | None = None
    moe_route_geometry: str = "raw"
    moe_transition_rank: int | None = None
    moe_deployed_parity: bool = False
    moe_even_width: int | None = None
    moe_guarded_ffn: bool = False
    moe_shared_e8_history: bool = False

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 24 == 0, "head_dim должен быть кратен 24"


def generate_leech_kernel(dim: int = 24) -> torch.Tensor:
    """Ортогональная матрица dim x dim (PoC через QR)."""
    base = torch.zeros(dim, dim, dtype=torch.float32)
    for i in range(dim - 1):
        base[i, i] = 2.0
        base[i, i + 1] = 2.0
    base[-1, -1] = 2.0
    base[-1, 0] = -2.0
    q, _ = torch.linalg.qr(base)
    return q


class LeechAttention(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim**-0.5
        self.num_blocks = self.head_dim // 24

        kernel = generate_leech_kernel(24)
        total_blocks = self.n_heads * self.num_blocks
        W_list = [kernel] * total_blocks
        self.register_buffer("W_leech", torch.block_diag(*W_list))

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size)),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Если use_cache=True, возвращает (out, present_kv).
        past_kv/present_kv: (k, v) в форме [B, H, T, head_dim].
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B,H,T,hd]

        # Leech kernel applied to q/k (blockwise 24)
        q = q.view(B, self.n_heads, T, self.num_blocks, 24)
        k = k.view(B, self.n_heads, T, self.num_blocks, 24)
        kernel = self.W_leech[0:24, 0:24]
        q = torch.einsum("...i,ij->...j", q, kernel)
        k = torch.einsum("...i,ij->...j", k, kernel)
        q = q.reshape(B, self.n_heads, T, self.head_dim)
        k = k.reshape(B, self.n_heads, T, self.head_dim)

        if use_cache:
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            present = (k, v)
            T_k = k.size(2)
            # если T == 1 и кэш используется, "будущих" ключей нет -> маска не нужна
            scores = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,T,T_k]
            if T > 1:
                # On resumed multi-token chunks, query row 0 is at
                # past_len, not position zero. Zero-offset slicing
                # mistakenly masks all earlier keys in such chunks.
                past_len = T_k - T
                mask = self.causal_mask[:, :, past_len:past_len+T, :T_k]
                scores = scores.masked_fill(mask == 0, float("-inf"))
        else:
            present = None
            scores = (q @ k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # Avoid the unstable transpose+reshape flatten path observed on gfx803 ROCm.
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(B, T, -1)
        # Materialize the output projection explicitly; this is easier to probe than nn.Linear.
        out = torch.matmul(out, self.out.weight.t())
        if self.out.bias is not None:
            out = out + self.out.bias
        return out, present


class LeechBlock(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LeechAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.moe = None
        if getattr(cfg, "moe_experts", 0) >= 2:
            if getattr(cfg, "moe_shared_e8_history", False):
                self.moe = E8SharedOrbitHistoryMoE(
                    cfg.d_model, cfg.moe_experts, cfg.moe_top_k,
                    cfg.moe_expert_width,
                    transition_rank=getattr(cfg, "moe_transition_rank", None),
                    route_geometry=getattr(cfg, "moe_route_geometry", "e8_root_invariant"))
            else:
                self.moe = HistoryOrbitMoE(
                    cfg.d_model, cfg.moe_experts, cfg.moe_top_k,
                    cfg.moe_expert_width,
                    route_geometry=getattr(cfg, "moe_route_geometry", "raw"),
                    transition_rank=getattr(cfg, "moe_transition_rank", None))
        self.even_student = None
        self.quality_gate = None
        self.last_local_distillation_loss = None
        if self.moe is not None:
            # Quality-guarded hybrid: frozen teacher FFN remains a
            # no-regression fallback; gate=0 gives the exact dense path.
            # Unguarded historical MoE checkpoints keep their old layout.
            if getattr(cfg, "moe_guarded_ffn", False):
                self.quality_gate = nn.Parameter(torch.tensor(0.0))
            else:
                self.ffn = None
            if getattr(cfg, "moe_deployed_parity", False):
                width = cfg.moe_even_width or max(24, cfg.d_model // 2)
                self.even_student = nn.Sequential(
                    nn.Linear(cfg.d_model, width), nn.GELU(),
                    nn.Linear(width, cfg.d_model))

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        expert_history: torch.Tensor | None = None,
        return_expert_history: bool = False,
    ):
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        normalized = self.ln2(x)
        if self.moe is None:
            x = x + self.ffn(normalized)
            if return_expert_history:
                return x, present, None
            return x, present
        expert_out, new_history, _, _, _ = self.moe(normalized, expert_history, train_sign=self.training)
        student_delta = expert_out
        if self.even_student is not None:
            # The independent even channel is essential: signed experts
            # alone restrict the model to odd local outputs.
            student_delta = student_delta + self.even_student(
                canonical_even_input(normalized))
        if self.quality_gate is not None:
            teacher_delta = self.ffn(normalized)
            # This local loss gives the student gradients even while the
            # quality-preserving mixture gate is identically zero.
            self.last_local_distillation_loss = F.mse_loss(
                student_delta,teacher_delta.detach())
            # Straight teacher residual at gate=0. No 12-layer
            # replacement is admitted just for matching a local FFN MSE.
            alpha = self.quality_gate.clamp(0., 1.)
            x = x + teacher_delta + alpha*(student_delta-teacher_delta)
        else:
            x = x + student_delta
        if return_expert_history:
            return x, present, new_history
        return x, present


class LeechResonanceBiasing(nn.Module):
    """Смещение логитов по резонансу с базисом Лича (как в ноутбуке)."""

    def __init__(self, d_model: int, vocab_size: int, leech_basis: torch.Tensor, alpha_init: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_blocks = d_model // 24
        self.register_buffer("basis", leech_basis)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.register_buffer("token_proj", torch.zeros(vocab_size, 24))

    @torch.no_grad()
    def compute_token_proj(self, tok_emb: torch.Tensor) -> None:
        emb_blocks = tok_emb.view(-1, self.num_blocks, 24)
        proj = torch.einsum("vki,ij->vkj", emb_blocks, self.basis)
        proj = proj.sum(dim=1)
        self.token_proj.copy_(proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        B, T, D = hidden_states.shape
        h_blocks = hidden_states.view(B, T, self.num_blocks, 24)
        h_proj = torch.einsum("btki,ij->btkj", h_blocks, self.basis)
        h_proj = h_proj.sum(dim=2)
        bias = self.alpha * (h_proj @ self.token_proj.T)
        if T == 1:
            bias = bias.squeeze(1)
        return bias


class LeechResonanceLoss(nn.Module):
    def __init__(self, leech_basis: torch.Tensor):
        super().__init__()
        self.register_buffer("basis", leech_basis)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape
        h = hidden_states.view(B, T, D // 24, 24)
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.basis, dim=-1)
        sim = torch.matmul(h_norm, b_norm.T)
        max_sim = torch.max(sim, dim=-1)[0]
        return 1.0 - max_sim.mean()


class DreamDecoder:
    def __init__(self, leech_basis: torch.Tensor, threshold: float = 0.95):
        self.basis = leech_basis
        self.threshold = threshold

    def check(self, hidden_state: torch.Tensor) -> tuple[float, str]:
        h = hidden_state[:24].unsqueeze(0)
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.basis, dim=-1)
        sim = torch.matmul(h_norm, b_norm.T)
        max_res = torch.max(sim).item()
        if max_res > 0.999:
            status = "ABSOLUTE GENESIS"
        elif max_res > self.threshold:
            status = "AWAKE"
        else:
            status = "DREAMING"
        return max_res, status


class LeechGPT(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.blocks = nn.ModuleList([LeechBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        leech_basis = generate_leech_kernel(24)
        self.register_buffer("leech_basis", leech_basis)
        self.resonance_loss_fn = LeechResonanceLoss(leech_basis)

        self.resonator: LeechResonanceBiasing | None = None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def attach_resonator(self, alpha_init: float = 0.1) -> None:
        self.resonator = LeechResonanceBiasing(
            d_model=self.cfg.d_model,
            vocab_size=self.cfg.vocab_size,
            leech_basis=self.leech_basis,
            alpha_init=alpha_init,
        ).to(self.tok_emb.weight.device)
        self.resonator.compute_token_proj(self.tok_emb.weight.detach())

    @torch.no_grad()
    def refresh_resonator_token_proj(self) -> None:
        if self.resonator is not None:
            self.resonator.compute_token_proj(self.tok_emb.weight.detach())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_resonator: bool = False,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        past_expert_histories: list[torch.Tensor] | None = None,
        return_expert_histories: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        b, t = idx.size()
        assert t <= self.cfg.block_size

        if use_cache:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].size(2)
            if past_len + t > self.cfg.block_size:
                raise ValueError(
                    f"KV-cache поддерживается только пока (past_len+t) <= block_size ({self.cfg.block_size}). "
                    f"Сейчас past_len={past_len}, t={t}."
                )
            x = self.tok_emb(idx) + self.pos_emb[:, past_len : past_len + t, :]
        else:
            x = self.tok_emb(idx) + self.pos_emb[:, :t, :]

        moe_enabled = getattr(self.cfg, "moe_experts", 0) >= 2
        if moe_enabled and use_cache and past_key_values is not None and past_expert_histories is None:
            raise ValueError("MoE KV continuation requires synchronized expert histories")
        if past_expert_histories is not None and len(past_expert_histories) != len(self.blocks):
            raise ValueError("expert history count must equal number of layers")
        new_expert_histories = [] if moe_enabled else None
        new_past: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = None
            if use_cache and past_key_values is not None:
                pkv = past_key_values[i]
            if moe_enabled:
                prior_history = None if past_expert_histories is None else past_expert_histories[i]
                x, present, updated_history = block(
                    x, past_kv=pkv, use_cache=use_cache,
                    expert_history=prior_history, return_expert_history=True)
                new_expert_histories.append(updated_history)
            else:
                x, present = block(x, past_kv=pkv, use_cache=use_cache)
            if use_cache:
                assert present is not None
                new_past.append(present)
        x = self.final_norm(x)
        logits = self.head(x)
        hidden = x

        if use_resonator and self.resonator is not None:
            logits = logits + self.resonator(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            if self.cfg.lambda_geo > 0 and not use_resonator:
                loss_geo = self.resonance_loss_fn(hidden)
                loss = loss + self.cfg.lambda_geo * loss_geo

        if return_expert_histories:
            return logits, hidden, loss, new_past, new_expert_histories
        return logits, hidden, loss, new_past
