#!/usr/bin/env python3
# Author: Fengze Yang <fred.yang@utah.edu>
# License: Apache License 2.0

"""VIVID patch helpers for HuggingFace Qwen3-VL vision attention.

This module adapts VIVID KV compression to Qwen3-VL's packed-vision-token format
(`hidden_states`: [seq_len, dim], `cu_seqlens`: prefix-sum segment boundaries).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

logger = logging.get_logger(__name__)


def _resolve_visual_blocks(model: nn.Module):
    """Resolve Qwen3 vision `blocks` from common wrapper layouts."""
    candidates = [
        getattr(getattr(model, "model", None), "visual", None),
        getattr(model, "visual", None),
        getattr(getattr(getattr(model, "module", None), "model", None), "visual", None),
        getattr(getattr(model, "module", None), "visual", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "visual", None),
        getattr(getattr(model, "base_model", None), "visual", None),
    ]
    for visual in candidates:
        blocks = getattr(visual, "blocks", None)
        if blocks is not None:
            return visual, blocks
    return None, None


class Qwen3VIVIDVisionAttention(nn.Module):
    """Drop-in replacement for `Qwen3VLVisionAttention` with KV compression.

    Keeps Qwen3-VL projections (`qkv`, `proj`) intact and adds a learnable sparse
    assignment module to compress K/V from T to S semantic anchors.
    """

    def __init__(
        self,
        base_attn: nn.Module,
        num_anchors: int = 256,
        topk: int = 8,
        use_token_aggregation: bool = False,
        num_aggregated_tokens: int = 768,
        aggregation_topk: int = 4,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.config = base_attn.config
        self.num_heads = base_attn.num_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = getattr(base_attn, "scaling", self.head_dim**-0.5)

        # Reuse original projection layers to preserve all pretrained weights.
        self.qkv = base_attn.qkv
        self.proj = base_attn.proj

        self.enabled = enabled
        self.num_anchors = max(1, int(num_anchors))
        self.topk = max(1, int(topk))
        self.use_token_aggregation = bool(use_token_aggregation)
        self.num_aggregated_tokens = max(1, int(num_aggregated_tokens))
        self.aggregation_topk = max(1, int(aggregation_topk))

        hidden = max(256, self.hidden_size // 6)
        self.assign_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.assign_proj = nn.Sequential(
            nn.Linear(self.hidden_size, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, self.num_anchors, bias=False),
        )

        # Token aggregation modules (T -> T' -> T), optional.
        if self.use_token_aggregation:
            self.token_agg_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
            self.token_agg_proj = nn.Sequential(
                nn.Linear(self.hidden_size, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, self.num_aggregated_tokens, bias=False),
            )
        else:
            self.token_agg_norm = None
            self.token_agg_proj = None

        # Keep new params on same device/dtype as original attention.
        tgt_weight = self.qkv.weight
        self.assign_norm.to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        self.assign_proj.to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        if self.token_agg_norm is not None:
            self.token_agg_norm.to(device=tgt_weight.device, dtype=tgt_weight.dtype)
        if self.token_agg_proj is not None:
            self.token_agg_proj.to(device=tgt_weight.device, dtype=tgt_weight.dtype)

    def _unpack_packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack `[L, ...]` packed sequence into padded `[B, T, ...]` + mask."""
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        batch = lengths.numel()
        max_len = int(lengths.max().item()) if batch > 0 else 0
        out = x.new_zeros((batch, max_len, *x.shape[1:]))
        mask = torch.zeros((batch, max_len), dtype=torch.bool, device=x.device)

        start = 0
        for b, length in enumerate(lengths.tolist()):
            end = start + length
            out[b, :length] = x[start:end]
            mask[b, :length] = True
            start = end
        return out, mask, lengths

    def _pack_padded(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pack padded `[B, T, ...]` back to `[L, ...]` by valid lengths."""
        pieces = []
        for b, length in enumerate(lengths.tolist()):
            pieces.append(x[b, :length])
        return torch.cat(pieces, dim=0) if pieces else x.new_zeros((0, *x.shape[2:]))

    def _build_sparse_assignment(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        norm_layer: nn.LayerNorm,
        proj_layer: nn.Module,
        num_slots: int,
        topk: int,
    ) -> torch.Tensor:
        """Build sparse assignment matrix P: `[B, T, S]` (top-k non-zeros per row)."""
        logits = proj_layer(norm_layer(hidden_states))
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        # Mask invalid padded tokens.
        logits = logits.masked_fill(~valid_mask.unsqueeze(-1), -1e9)

        k = min(int(topk), int(num_slots))
        vals, idx = torch.topk(logits, k=k, dim=-1)

        probs = F.softmax(vals.float(), dim=-1).to(vals.dtype)
        probs = probs * valid_mask.unsqueeze(-1).to(probs.dtype)

        P = torch.zeros_like(logits)
        P.scatter_(-1, idx, probs)
        return P

    def _build_assignment(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self._build_sparse_assignment(
            hidden_states=hidden_states,
            valid_mask=valid_mask,
            norm_layer=self.assign_norm,
            proj_layer=self.assign_proj,
            num_slots=self.num_anchors,
            topk=self.topk,
        )

    def _merge_tokens_3d(self, x: torch.Tensor, P: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Merge `[B,T,D]` to `[B,S,D]` using P and masked normalization."""
        Pt = P.transpose(1, 2)  # [B,S,T]
        Pt = Pt * valid_mask[:, None, :].to(Pt.dtype)
        denom = Pt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        Pt_norm = Pt / denom
        return torch.matmul(Pt_norm, x)

    def _merge_tokens_4d(self, x: torch.Tensor, P: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Merge `[B,H,T,d]` to `[B,H,S,d]` using P and masked normalization."""
        bsz, heads, _, _ = x.shape
        P_h = P.unsqueeze(1).expand(bsz, heads, P.shape[1], P.shape[2])  # [B,H,T,S]
        Pt = P_h.transpose(2, 3)  # [B,H,S,T]
        Pt = Pt * valid_mask[:, None, None, :].to(Pt.dtype)
        denom = Pt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        Pt_norm = Pt / denom
        return torch.matmul(Pt_norm, x)

    def _expand_tokens_4d(self, y: torch.Tensor, P: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Expand `[B,H,S,d]` back to `[B,H,T,d]` with P."""
        bsz, heads, t_len, _ = P.shape[0], y.shape[1], P.shape[1], y.shape[-1]
        P_h = P.unsqueeze(1).expand(bsz, heads, t_len, P.shape[2])  # [B,H,T,S]
        out = torch.matmul(P_h, y)
        out = out * valid_mask[:, None, :, None].to(out.dtype)
        return out

    def _compressed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """KV-compressed attention for padded batch tensors.

        Args:
            q, k, v: `[B, H, T, d]`
            hidden_states: `[B, T, D]`
            valid_mask: `[B, T]`
        Returns:
            `[B, H, T, d]`
        """
        if self.use_token_aggregation and self.token_agg_norm is not None and self.token_agg_proj is not None:
            P_tok = self._build_sparse_assignment(
                hidden_states=hidden_states,
                valid_mask=valid_mask,
                norm_layer=self.token_agg_norm,
                proj_layer=self.token_agg_proj,
                num_slots=self.num_aggregated_tokens,
                topk=self.aggregation_topk,
            )  # [B,T,T']

            hidden_agg = self._merge_tokens_3d(hidden_states, P_tok, valid_mask)  # [B,T',D]
            q_agg = self._merge_tokens_4d(q, P_tok, valid_mask)  # [B,H,T',d]
            k_agg = self._merge_tokens_4d(k, P_tok, valid_mask)  # [B,H,T',d]
            v_agg = self._merge_tokens_4d(v, P_tok, valid_mask)  # [B,H,T',d]

            # Aggregated token validity.
            valid_agg = (P_tok.sum(dim=1) > 0)  # [B,T']

            P = self._build_assignment(hidden_agg, valid_agg)  # [B,T',S]
            bsz, heads, seq_agg, _ = q_agg.shape
            P_h = P.unsqueeze(1).expand(bsz, heads, seq_agg, self.num_anchors)  # [B,H,T',S]
            Pt = P_h.transpose(2, 3)  # [B,H,S,T']
            Pt = Pt * valid_agg[:, None, None, :].to(Pt.dtype)
            denom = Pt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            Pt_norm = Pt / denom

            k_comp = torch.matmul(Pt_norm, k_agg)  # [B,H,S,d]
            v_comp = torch.matmul(Pt_norm, v_agg)  # [B,H,S,d]

            attn_scores = torch.matmul(q_agg, k_comp.transpose(-2, -1)) * self.scaling  # [B,H,T',S]
            attn_weights = F.softmax(attn_scores, dim=-1)
            out_agg = torch.matmul(attn_weights, v_comp)  # [B,H,T',d]

            out = self._expand_tokens_4d(out_agg, P_tok, valid_mask)  # [B,H,T,d]
            return out

        bsz, heads, seq_len, _ = q.shape
        P = self._build_assignment(hidden_states, valid_mask)  # [B, T, S]
        P_h = P.unsqueeze(1).expand(bsz, heads, seq_len, self.num_anchors)  # [B, H, T, S]

        Pt = P_h.transpose(2, 3)  # [B, H, S, T]
        valid_h = valid_mask[:, None, None, :].to(Pt.dtype)
        Pt = Pt * valid_h

        denom = Pt.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        Pt_norm = Pt / denom

        k_comp = torch.matmul(Pt_norm, k)  # [B, H, S, d]
        v_comp = torch.matmul(Pt_norm, v)  # [B, H, S, d]

        attn_scores = torch.matmul(q, k_comp.transpose(-2, -1)) * self.scaling  # [B, H, T, S]
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_comp)  # [B, H, T, d]

        out = out * valid_mask[:, None, :, None].to(out.dtype)
        return out

    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Masked standard attention fallback for padded batch tensors."""
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B, H, T, T]
        key_mask = valid_mask[:, None, None, :]
        attn_scores = attn_scores.masked_fill(~key_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out * valid_mask[:, None, :, None].to(out.dtype)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        # Qwen3-VL packed-token projection layout.
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Unpack to padded batch.
        q_bt, valid_mask, lengths = self._unpack_packed(query_states, cu_seqlens)  # [B, T, H, d]
        k_bt, _, _ = self._unpack_packed(key_states, cu_seqlens)
        v_bt, _, _ = self._unpack_packed(value_states, cu_seqlens)
        h_bt, _, _ = self._unpack_packed(hidden_states, cu_seqlens)  # [B, T, D]

        q = q_bt.transpose(1, 2).contiguous()  # [B, H, T, d]
        k = k_bt.transpose(1, 2).contiguous()
        v = v_bt.transpose(1, 2).contiguous()

        if self.enabled:
            out = self._compressed_attention(q, k, v, h_bt, valid_mask)
        else:
            out = self._standard_attention(q, k, v, valid_mask)

        # Back to packed-token format expected by Qwen3 vision blocks.
        out_bt = out.transpose(1, 2).contiguous()  # [B, T, H, d]
        out_packed = self._pack_padded(out_bt, lengths)  # [L, H, d]
        out_packed = out_packed.reshape(seq_len, -1).contiguous()  # [L, D]
        out_packed = self.proj(out_packed)
        return out_packed


def apply_vivid_qwen3_vision_patch(
    model: nn.Module,
    num_anchors: int = 256,
    topk: int = 8,
    use_token_aggregation: bool = False,
    num_aggregated_tokens: int = 768,
    aggregation_topk: int = 4,
    enabled: bool = True,
) -> int:
    """Replace Qwen3-VL vision attention blocks with VIVID adapter.

    Returns:
        Number of patched vision blocks.
    """
    visual, blocks = _resolve_visual_blocks(model)
    if blocks is None:
        logger.warning("VIVID patch skipped: no Qwen3 vision `blocks` found.")
        return 0

    patched = 0
    for i, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        if attn is None or isinstance(attn, Qwen3VIVIDVisionAttention):
            continue
        block.attn = Qwen3VIVIDVisionAttention(
            base_attn=attn,
            num_anchors=num_anchors,
            topk=topk,
            use_token_aggregation=use_token_aggregation,
            num_aggregated_tokens=num_aggregated_tokens,
            aggregation_topk=aggregation_topk,
            enabled=enabled,
        )
        patched += 1
        logger.info("Patched Qwen3 vision block %s with VIVID attention", i)

    logger.info(
        "VIVID patch result: patched %s vision blocks (enabled=%s, anchors=%s, topk=%s, token_agg=%s, agg_tokens=%s, agg_topk=%s)",
        patched,
        enabled,
        num_anchors,
        topk,
        use_token_aggregation,
        num_aggregated_tokens,
        aggregation_topk,
    )
    return patched


def count_vivid_qwen3_blocks(model: nn.Module) -> int:
    """Count how many Qwen3 vision blocks currently use VIVID attention."""
    _, blocks = _resolve_visual_blocks(model)
    if blocks is None:
        return 0
    return sum(1 for blk in blocks if isinstance(getattr(blk, "attn", None), Qwen3VIVIDVisionAttention))


def is_vivid_qwen3_vision_patched(model: nn.Module) -> bool:
    """Return True iff at least one Qwen3 vision block uses VIVID attention."""
    return count_vivid_qwen3_blocks(model) > 0
