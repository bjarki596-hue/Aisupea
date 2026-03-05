"""
Aisupea Inference Engine

Text generation engine with greedy decoding, temperature sampling, and top-k sampling.
Supports token streaming and incremental inference with KV cache.
"""

import math
from typing import List, Optional, Tuple, Callable
from ..core import Tensor
from ..models.transformer import Transformer


class InferenceEngine:
    """
    Text generation engine for transformer models.

    Supports various decoding strategies and KV cache for efficient inference.
    """

    def __init__(self, model: Transformer, tokenizer):
        """
        Initialize inference engine.

        Args:
            model: Trained transformer model
            tokenizer: Tokenizer for encoding/decoding text
        """
        self.model = model
        self.tokenizer = tokenizer

        # KV cache for incremental inference
        self.kv_cache: Optional[List] = None
        self.current_length = 0

    def reset_cache(self):
        """Reset KV cache."""
        self.kv_cache = None
        self.current_length = 0

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0,
                top_k: Optional[int] = None, top_p: Optional[float] = None,
                repetition_penalty: float = 1.0, stream: bool = False,
                callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_length: Maximum total sequence length
            temperature: Sampling temperature (1.0 = greedy)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            stream: Whether to stream output token by token
            callback: Callback function for streaming

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = Tensor([input_ids], 'int')

        # Reset cache for new generation
        self.reset_cache()

        generated_tokens = input_ids.copy()
        self.current_length = len(generated_tokens)

        # Generation loop
        while len(generated_tokens) < max_length:
            # Get next token logits
            logits, self.kv_cache = self.model.forward(input_tensor, use_cache=True)

            # Get logits for last position
            next_logits = logits[0, -1, :]  # Shape: (vocab_size,)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(next_logits, generated_tokens, repetition_penalty)

            # Apply sampling strategy
            next_token = self._sample_token(next_logits, temperature, top_k, top_p)

            # Append to sequence
            generated_tokens.append(next_token)
            self.current_length += 1

            # Update input tensor for next iteration
            input_tensor = Tensor([generated_tokens], 'int')

            # Stream output if requested
            if stream and callback:
                decoded_token = self.tokenizer.decode([next_token])
                callback(decoded_token)

            # Check for EOS token
            if next_token == self.tokenizer.special_tokens['<eos>']:
                break

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special=True)

        # Remove the prompt from the beginning
        prompt_length = len(prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[prompt_length:]

        return generated_text.strip()

    def _sample_token(self, logits: Tensor, temperature: float, top_k: Optional[int],
                     top_p: Optional[float]) -> int:
        """
        Sample next token using specified strategy.

        Args:
            logits: Token logits
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k sampling
        if top_k is not None:
            logits = self._top_k_filter(logits, top_k)

        # Apply top-p sampling
        if top_p is not None:
            logits = self._top_p_filter(logits, top_p)

        # Convert to probabilities
        probs = logits.softmax(dim=-1)

        # For now, use greedy decoding (argmax)
        # Real sampling would use torch.multinomial or similar
        next_token = probs.argmax(dim=-1).item()

        return next_token

    def _top_k_filter(self, logits: Tensor, top_k: int) -> Tensor:
        """Apply top-k filtering to logits."""
        # Get top k values and indices
        # This is simplified - real implementation would use proper topk
        return logits  # Placeholder

    def _top_p_filter(self, logits: Tensor, top_p: float) -> Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        # Sort logits in descending order
        # This is simplified - real implementation would sort and filter
        return logits  # Placeholder

    def _apply_repetition_penalty(self, logits: Tensor, generated_tokens: List[int],
                                penalty: float) -> Tensor:
        """Apply repetition penalty to logits."""
        # Penalize tokens that have already been generated
        # This is simplified
        return logits  # Placeholder

    def greedy_decode(self, prompt: str, max_length: int = 50) -> str:
        """Greedy decoding (always pick highest probability token)."""
        return self.generate(prompt, max_length, temperature=0.0)

    def temperature_sampling(self, prompt: str, max_length: int = 50,
                           temperature: float = 0.8) -> str:
        """Temperature-based sampling."""
        return self.generate(prompt, max_length, temperature=temperature)

    def top_k_sampling(self, prompt: str, max_length: int = 50,
                      temperature: float = 0.8, top_k: int = 40) -> str:
        """Top-k sampling."""
        return self.generate(prompt, max_length, temperature=temperature, top_k=top_k)

    def top_p_sampling(self, prompt: str, max_length: int = 50,
                      temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Top-p (nucleus) sampling."""
        return self.generate(prompt, max_length, temperature=temperature, top_p=top_p)

    def stream_generate(self, prompt: str, max_length: int = 50,
                       callback: Callable[[str], None] = None) -> str:
        """Generate text with token streaming."""
        return self.generate(prompt, max_length, stream=True, callback=callback)

    def get_cache_size(self) -> int:
        """Get current KV cache size in tokens."""
        return self.current_length

    def clear_cache(self):
        """Clear KV cache to free memory."""
        self.reset_cache()