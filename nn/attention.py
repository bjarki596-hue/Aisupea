"""
Aisupea Attention Mechanisms Module

Multi-head attention and transformer components.
"""

from typing import Optional, Tuple, Union
from ..core import Tensor
import math
import random


class ScaledDotProductAttention:
    """Scaled dot-product attention mechanism."""

    def __init__(self, dropout: float = 0.0):
        """
        Initialize scaled dot-product attention.

        Args:
            dropout: Dropout probability
        """
        self.dropout = dropout

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch_size, num_heads, seq_len, head_dim)
        """
        # Compute attention scores: Q * K^T / sqrt(d_k)
        d_k = query.shape[-1]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * float('-inf')

        # Apply softmax
        attention_weights = scores.softmax(dim=-1)

        # Apply dropout (simplified - just scale)
        if self.dropout > 0:
            # In a real implementation, would randomly zero out elements
            attention_weights = attention_weights * (1 - self.dropout)

        # Compute weighted sum: attention_weights * V
        output = attention_weights @ value

        return output


class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias in linear layers
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Linear transformation layers
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.shape

        # Linear transformations and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        attn_output = self.attention(q, k, v, mask)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output


class Linear:
    """Linear transformation layer."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
        """
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = Tensor.zeros(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

        # Xavier initialization
        scale = (2.0 / (in_features + out_features)) ** 0.5
        self._init_weights(self.weight, scale)
        if self.bias is not None:
            self._init_weights(self.bias, scale)

    def _init_weights(self, tensor: Tensor, scale: float):
        """Initialize weights with random values."""
        if tensor.ndim == 2:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    tensor.data[i][j] = random.gauss(0, scale)
        else:  # 1D tensor
            for i in range(tensor.shape[0]):
                tensor.data[i] = random.gauss(0, scale)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of linear layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        output = x @ self.weight.transpose()
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).expand(x.shape[0], -1)
        return output


class TransformerEncoderLayer:
    """Transformer encoder layer."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer encoder layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.linear1 = Linear(embed_dim, ff_dim)
        self.linear2 = Linear(ff_dim, embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer encoder layer.

        Args:
            src: Source tensor of shape (batch_size, seq_len, embed_dim)
            src_mask: Optional source mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self._dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.linear2(self._relu(self.linear1(src)))
        src = self.norm2(src + self._dropout(ff_output))

        return src

    def _dropout(self, x: Tensor) -> Tensor:
        """Apply dropout (simplified)."""
        if self.dropout > 0:
            # In a real implementation, would randomly zero out elements
            return x * (1 - self.dropout)
        return x

    def _relu(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        # Simple ReLU implementation
        def relu_op(val):
            return max(0, val)
        return Tensor(x._apply_func_recursive(x.data, relu_op), x.dtype)


class LayerNorm:
    """Layer normalization."""

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        """
        Initialize layer normalization.

        Args:
            normalized_shape: Shape to normalize over
            eps: Small value for numerical stability
        """
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps

        # Learnable parameters
        param_size = 1
        for dim in self.normalized_shape:
            param_size *= dim

        self.weight = Tensor.ones(param_size)
        self.bias = Tensor.zeros(param_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of layer normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Compute mean and variance along the last dimensions
        mean = self._compute_mean(x)
        var = self._compute_var(x, mean)

        # Normalize
        normalized = (x - mean) / (var + self.eps).sqrt()

        # Apply learnable parameters
        return normalized * self.weight + self.bias

    def _compute_mean(self, x: Tensor) -> Tensor:
        """Compute mean along normalized dimensions."""
        # Simplified: assume normalizing over last dimension
        if x.ndim == 1:
            return Tensor([sum(x.data) / len(x.data)])
        elif x.ndim == 2:
            means = []
            for row in x.data:
                means.append(sum(row) / len(row))
            return Tensor([means])
        else:
            # For higher dimensions, normalize over last dimension
            result_shape = x.shape[:-1] + (1,)
            result = Tensor.zeros(*result_shape)

            def compute_mean_recursive(data, current_dim=0):
                if current_dim == x.ndim - 2:  # Last dimension to normalize
                    for i, row in enumerate(data):
                        mean_val = sum(row) / len(row)
                        result.data[i] = [mean_val] * len(row)
                else:
                    for i, subdata in enumerate(data):
                        compute_mean_recursive(subdata, current_dim + 1)

            compute_mean_recursive(x.data)
            return result

    def _compute_var(self, x: Tensor, mean: Tensor) -> Tensor:
        """Compute variance along normalized dimensions."""
        # Simplified implementation similar to mean
        if x.ndim == 1:
            mean_val = mean.data[0]
            var_val = sum((val - mean_val) ** 2 for val in x.data) / len(x.data)
            return Tensor([var_val])
        elif x.ndim == 2:
            vars = []
            for i, row in enumerate(x.data):
                mean_val = mean.data[0][i]
                var_val = sum((val - mean_val) ** 2 for val in row) / len(row)
                vars.append(var_val)
            return Tensor([vars])
        else:
            # Simplified for higher dimensions
            return Tensor.ones_like(mean) * 0.1  # Placeholder


class TransformerEncoder:
    """Transformer encoder consisting of multiple layers."""

    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer encoder.

        Args:
            num_layers: Number of encoder layers
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        self.layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.norm = LayerNorm(embed_dim)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer encoder.

        Args:
            src: Source tensor of shape (batch_size, seq_len, embed_dim)
            src_mask: Optional source mask

        Returns:
            Encoded tensor of shape (batch_size, seq_len, embed_dim)
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)

        return self.norm(output)


class TransformerDecoderLayer:
    """Transformer decoder layer."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer decoder layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.linear1 = Linear(embed_dim, ff_dim)
        self.linear2 = Linear(ff_dim, embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer decoder layer.

        Args:
            tgt: Target tensor of shape (batch_size, seq_len, embed_dim)
            memory: Memory tensor from encoder of shape (batch_size, seq_len, embed_dim)
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self._dropout(attn_output))

        # Cross-attention with residual connection and layer norm
        attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self._dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.linear2(self._relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + self._dropout(ff_output))

        return tgt

    def _dropout(self, x: Tensor) -> Tensor:
        """Apply dropout (simplified)."""
        if self.dropout > 0:
            return x * (1 - self.dropout)
        return x

    def _relu(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        def relu_op(val):
            return max(0, val)
        return Tensor(x._apply_func_recursive(x.data, relu_op), x.dtype)


class TransformerDecoder:
    """Transformer decoder consisting of multiple layers."""

    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer decoder.

        Args:
            num_layers: Number of decoder layers
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        self.layers = [
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.norm = LayerNorm(embed_dim)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer decoder.

        Args:
            tgt: Target tensor of shape (batch_size, seq_len, embed_dim)
            memory: Memory tensor from encoder of shape (batch_size, seq_len, embed_dim)
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask

        Returns:
            Decoded tensor of shape (batch_size, seq_len, embed_dim)
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        return self.norm(output)


class Transformer:
    """Complete transformer model."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 6, ff_dim: int = 2048, dropout: float = 0.1):
        """
        Initialize transformer model.

        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            ff_dim: Feed-forward dimension
            dropout: Dropout probability
        """
        self.src_embed = Embedding(src_vocab_size, embed_dim)
        self.tgt_embed = Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)

        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, dropout)

        self.generator = Linear(embed_dim, tgt_vocab_size)

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Encode source sequence."""
        src_embed = self.pos_encoder(self.src_embed(src))
        return self.encoder(src_embed, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None) -> Tensor:
        """Decode target sequence."""
        tgt_embed = self.pos_decoder(self.tgt_embed(tgt))
        return self.decoder(tgt_embed, memory, tgt_mask)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        """Complete forward pass."""
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, tgt_mask)
        return self.generator(decoded)


class Embedding:
    """Embedding layer."""

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize embedding matrix
        self.weight = Tensor.zeros(vocab_size, embed_dim)
        scale = (2.0 / (vocab_size + embed_dim)) ** 0.5
        for i in range(vocab_size):
            for j in range(embed_dim):
                self.weight.data[i][j] = random.gauss(0, scale)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices

        Returns:
            Embedded tensor
        """
        # Simple embedding lookup (simplified implementation)
        batch_size, seq_len = x.shape
        output = Tensor.zeros(batch_size, seq_len, self.embed_dim)

        for b in range(batch_size):
            for s in range(seq_len):
                token_idx = int(x.data[b][s])
                if 0 <= token_idx < self.vocab_size:
                    for d in range(self.embed_dim):
                        output.data[b][s][d] = self.weight.data[token_idx][d]

        return output

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class PositionalEncoding:
    """Positional encoding for transformer."""

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        self.dropout = dropout

        # Create positional encoding matrix
        pe = Tensor.zeros(max_len, embed_dim)
        position = Tensor.arange(0, max_len, 1).unsqueeze(1)
        div_term = Tensor.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)

        pe[:, 0::2] = (position * div_term.unsqueeze(0)).exp().sin()
        pe[:, 1::2] = (position * div_term.unsqueeze(0)).exp().cos()

        self.pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len]

        # Apply dropout (simplified)
        if self.dropout > 0:
            x = x * (1 - self.dropout)

        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


__all__ = [
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'Linear',
    'TransformerEncoderLayer',
    'LayerNorm',
    'TransformerEncoder',
    'TransformerDecoderLayer',
    'TransformerDecoder',
    'Transformer',
    'Embedding',
    'PositionalEncoding'
]