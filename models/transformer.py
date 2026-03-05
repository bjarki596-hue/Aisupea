"""
Aisupea Transformer Architecture

Lightweight transformer model with rotary positional embeddings,
multi-head self attention, feed-forward blocks, and KV cache support.
"""

import math
from typing import Optional, Tuple, List
from ..core import Tensor
from ..torch_api import nn, functional as F


class RotaryPositionalEmbedding:
    """
    Rotary Positional Embedding (RoPE).

    Applies rotary position embedding to query and key tensors.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute rotary matrices
        self._compute_rotary_matrices()

    def _compute_rotary_matrices(self):
        """Precompute rotary matrices for all positions."""
        # RoPE: R^d @ x where R^d is rotation matrix
        # For each position and each pair of dimensions

        self.cos_cache = []
        self.sin_cache = []

        for pos in range(self.max_seq_len):
            cos_row = []
            sin_row = []
            for i in range(0, self.dim, 2):
                theta = pos / (10000 ** (2 * i / self.dim))
                cos_row.extend([math.cos(theta), math.cos(theta)])
                sin_row.extend([math.sin(theta), math.sin(theta)])
            self.cos_cache.append(cos_row[:self.dim])
            self.sin_cache.append(sin_row[:self.dim])

    def apply_rotary_emb(self, x: Tensor, positions: Optional[List[int]] = None) -> Tensor:
        """
        Apply rotary embedding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            positions: Position indices, if None uses sequential positions

        Returns:
            Tensor with rotary embedding applied
        """
        batch_size, seq_len, dim = x.shape

        if positions is None:
            positions = list(range(seq_len))

        # For each position, apply rotation
        result_data = []

        for b in range(batch_size):
            batch_result = []
            for s in range(seq_len):
                pos = positions[s]
                if pos >= self.max_seq_len:
                    # Fallback for out-of-range positions
                    cos_vals = [1.0] * dim
                    sin_vals = [0.0] * dim
                else:
                    cos_vals = self.cos_cache[pos]
                    sin_vals = self.sin_cache[pos]

                # Apply rotation: x[i:i+2] = [x[i]*cos - x[i+1]*sin, x[i]*sin + x[i+1]*cos]
                rotated = []
                x_row = x[b, s]  # This would need proper indexing

                # Simplified - real implementation would handle the rotation properly
                rotated = x_row  # Placeholder

                batch_result.append(rotated)
            result_data.append(batch_result)

        return Tensor(result_data, x.dtype)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self Attention with rotary positional embeddings.
    """

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        # KV cache for inference
        self.kv_cache = None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, use_cache: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Attention mask
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, kv_cache)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply rotary embeddings to Q and K
        q_positions = list(range(seq_len))
        k_positions = list(range(seq_len))

        # Apply RoPE
        q = self._apply_rope_to_heads(q, q_positions)
        k = self._apply_rope_to_heads(k, k_positions)

        # Handle KV cache
        if use_cache and self.kv_cache is not None:
            # Concatenate with cached K, V
            cached_k, cached_v = self.kv_cache
            k = self._concat_with_cache(k, cached_k)
            v = self._concat_with_cache(v, cached_v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            # Apply mask (add large negative value where mask is True)
            attn_weights = attn_weights + mask * (-1e9)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Transpose back: (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Reshape: (batch_size, seq_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Update KV cache
        if use_cache:
            self.kv_cache = (k, v)

        return output, self.kv_cache

    def _apply_rope_to_heads(self, x: Tensor, positions: List[int]) -> Tensor:
        """Apply RoPE to each attention head."""
        # This is simplified - real implementation would apply RoPE to each head
        return x

    def _concat_with_cache(self, current: Tensor, cached: Tensor) -> Tensor:
        """Concatenate current K/V with cached K/V."""
        # Simplified concatenation
        return current  # Placeholder


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, use_cache: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor
            mask: Attention mask
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (output, kv_cache)
        """
        # Self-attention with residual connection
        attn_output, kv_cache = self.attention(self.norm1(x), mask, use_cache)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x, kv_cache


class Transformer(nn.Module):
    """
    Complete transformer model.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int,
                 hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learnable)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.layers = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, hidden_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        # Output layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights between input embeddings and output layer
        # In a real implementation, we'd tie the weights

    def forward(self, input_ids: Tensor, use_cache: bool = False) -> Tuple[Tensor, Optional[List]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            use_cache: Whether to use KV cache

        Returns:
            Tuple of (logits, kv_caches)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Position embeddings
        positions = Tensor.arange(0, seq_len, dtype='int')
        pos_embeds = self.position_embeddings(positions)

        # Combine embeddings
        x = token_embeds + pos_embeds

        # Apply transformer layers
        kv_caches = []
        for layer in self.layers._modules.values():
            x, kv_cache = layer(x, use_cache=use_cache)
            kv_caches.append(kv_cache)

        # Final layer norm
        x = self.norm(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits, kv_caches if use_cache else None

    def generate(self, input_ids: Tensor, max_length: int = 50,
                temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token IDs
        """
        # This is a simplified generation loop
        # Real implementation would handle KV cache properly

        generated = input_ids.tolist()
        current_length = len(generated[0]) if generated else 0

        for _ in range(max_length - current_length):
            # Get logits for next token
            logits, _ = self.forward(Tensor(generated), use_cache=True)

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                # Simplified top-k
                pass

            # Sample next token (greedy for now)
            next_token = next_logits.argmax(dim=-1)

            # Append to generated sequence
            for i in range(len(generated)):
                generated[i].append(next_token.tolist()[i])

        return Tensor(generated, 'int')