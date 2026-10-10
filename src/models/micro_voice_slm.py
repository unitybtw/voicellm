"""Local causal text/audio backbone; dimensions and capabilities come from its configuration."""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return normalized.to(x.dtype) * self.weight


def precompute_rope_frequencies(dim: int, max_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin frequencies for Rotary Position Embeddings (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to query or key tensors."""
    # x: [B, num_heads, seq_len, head_dim]
    b, h, s, d = x.shape
    cos = cos[:s, :].unsqueeze(0).unsqueeze(1) # [1, 1, s, d//2]
    sin = sin[:s, :].unsqueeze(0).unsqueeze(1) # [1, 1, s, d//2]

    x_half1 = x[..., :d // 2]
    x_half2 = x[..., d // 2:]

    rx_half1 = x_half1 * cos - x_half2 * sin
    rx_half2 = x_half1 * sin + x_half2 * cos
    return torch.cat([rx_half1, rx_half2], dim=-1)


class GroupedQueryAttention(nn.Module):
    """GQA: Multi-head Query with reduced Key-Value heads to minimize KV-cache footprint."""
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, num_kv_heads: int = 2):
        super().__init__()
        if hidden_size % num_heads or num_heads % num_kv_heads or (hidden_size // num_heads) % 2:
            raise ValueError("GQA requires divisible heads and an even head dimension")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Handle KV cache for real-time streaming
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Expand Key and Value for GQA
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        context = context.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        return self.out_proj(context), kv_cache


class SwiGLUMlp(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1536):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single causal transformer block with RMSNorm and SwiGLU."""
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, num_kv_heads: int = 2, intermediate_size: int = 1536):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        normed = self.attn_norm(x)
        attn_out, kv_cache = self.attn(normed, cos, sin, mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv_cache


class MicroVoiceSLM(nn.Module):
    """
    Locally initialized causal text/audio Transformer.
    Processes interleaved text & acoustic tokens.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        audio_vocab_size: int = 1024,
        num_audio_codebooks: int = 4,
        hidden_size: int = 512,
        num_layers: int = 14,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        intermediate_size: int = 1536,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_audio_codebooks = num_audio_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.max_seq_len = max_seq_len

        # 1. Embeddings
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)
        # One embedding per configured acoustic table
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(audio_vocab_size, hidden_size) for _ in range(num_audio_codebooks)
        ])

        # 2. Backbone Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, num_kv_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_size)

        # 3. Heads
        self.text_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Prediction heads for the 4 RVQ codebooks
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_audio_codebooks)
        ])

        # Precompute RoPE
        cos, sin = precompute_rope_frequencies(hidden_size // num_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def embed_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Combines the configured codebook embeddings into one vector per audio frame.
        Args:
            audio_tokens: [B, num_codebooks, T_frames]
        Returns:
            [B, T_frames, hidden_size]
        """
        B, K, T = audio_tokens.shape
        embedded = 0.0
        for i in range(self.num_audio_codebooks):
            embedded = embedded + self.audio_embeddings[i](audio_tokens[:, i, :])
        return embedded

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        past_kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        pad_to_multiple: int = 1,
        compute_audio_logits: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for training or generation.
        Returns:
            outputs: dictionary with 'text_logits' and 'audio_logits' [B, num_codebooks, seq_len, vocab]
        """
        if inputs_embeds is None:
            if text_tokens is not None and audio_tokens is not None:
                # Interleaved representation
                text_emb = self.text_embedding(text_tokens)
                audio_emb = self.embed_audio_tokens(audio_tokens)
                inputs_embeds = torch.cat([text_emb, audio_emb], dim=1)
            elif text_tokens is not None:
                inputs_embeds = self.text_embedding(text_tokens)
            elif audio_tokens is not None:
                inputs_embeds = self.embed_audio_tokens(audio_tokens)
            else:
                raise ValueError("Must provide either tokens or inputs_embeds")

        B, S, _ = inputs_embeds.shape
        original_length = S
        if pad_to_multiple < 1:
            raise ValueError('Padding multiple must be positive')
        if pad_to_multiple > 1:
            if past_kv_caches is not None:
                raise ValueError('Training padding cannot be combined with an incremental cache')
            # A bounded set of shapes avoids retaining a GPU graph per clip length.
            padded_length = min(self.max_seq_len, ((S + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple)
            if padded_length > S:
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, padded_length - S))
                if mask is not None:
                    mask = F.pad(mask, (0, padded_length - S, 0, padded_length - S))
                S = padded_length
        if past_kv_caches is None:
            past_kv_caches = [{} for _ in self.layers]
        if len(past_kv_caches) != len(self.layers):
            raise ValueError("One KV cache is required per layer")
        past_len = past_kv_caches[0].get("k").shape[2] if past_kv_caches and "k" in past_kv_caches[0] else 0
        if S < 1 or past_len + S > self.max_seq_len:
            raise ValueError("Input exceeds the model context window or is empty")
        cos = self.cos[past_len:past_len + S, :]
        sin = self.sin[past_len:past_len + S, :]

        # Cached chunks can attend to the whole prefix, but never future tokens.
        query_positions = torch.arange(past_len, past_len + S, device=inputs_embeds.device)
        key_positions = torch.arange(past_len + S, device=inputs_embeds.device)
        causal_mask = torch.zeros(S, past_len + S, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        causal_mask.masked_fill_(key_positions[None, :] > query_positions[:, None], float('-inf'))
        mask = causal_mask if mask is None else mask + causal_mask

        x = inputs_embeds
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = past_kv_caches[i] if past_kv_caches is not None else None
            x, cache_out = layer(x, cos, sin, mask=mask, kv_cache=cache_i)
            if S != original_length:
                cache_out = {name: value[:, :, :original_length] for name, value in cache_out.items()}
            new_caches.append(cache_out)

        x = self.final_norm(x)

        # Output predictions
        text_logits = self.text_head(x)
        audio_logits = (torch.stack([head(x) for head in self.audio_heads], dim=1)
                        if compute_audio_logits and len(self.audio_heads) else None)

        return {
            "text_logits": text_logits[:, :original_length],
            "audio_logits": audio_logits[:, :, :original_length] if audio_logits is not None else None,
            "hidden_states": x[:, :original_length]
        }, new_caches
