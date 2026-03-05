"""
Aisupea Tokenization System

Tokenizer that supports word and punctuation tokenization with vocabulary training.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


class Tokenizer:
    """
    Simple tokenizer with word and punctuation tokenization.

    Supports vocabulary training, encoding, and decoding.
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }

        # Initialize with special tokens
        self.vocab.update(self.special_tokens)
        for token, idx in self.special_tokens.items():
            self.inverse_vocab[idx] = token

    def train(self, texts: List[str], min_freq: int = 2):
        """
        Train vocabulary on a corpus of texts.

        Args:
            texts: List of training texts
            min_freq: Minimum frequency for token inclusion
        """
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Filter by frequency and vocab size
        filtered_tokens = [
            token for token, count in token_counts.most_common()
            if count >= min_freq and token not in self.special_tokens
        ]

        # Limit to vocab size (accounting for special tokens)
        available_slots = self.vocab_size - len(self.special_tokens)
        filtered_tokens = filtered_tokens[:available_slots]

        # Assign indices
        next_idx = len(self.special_tokens)
        for token in filtered_tokens:
            self.vocab[token] = next_idx
            self.inverse_vocab[next_idx] = token
            next_idx += 1

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize a single text into words and punctuation.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Split on whitespace and punctuation
        # Use regex to split on word boundaries and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)

        return tokens

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Whether to add beginning of sequence token
            add_eos: Whether to add end of sequence token

        Returns:
            List of token IDs
        """
        tokens = self._tokenize_text(text)
        token_ids = []

        if add_bos:
            token_ids.append(self.special_tokens['<bos>'])

        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.special_tokens['<unk>'])

        if add_eos:
            token_ids.append(self.special_tokens['<eos>'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if skip_special and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append('<unk>')

        # Join tokens with spaces, but handle punctuation
        result = []
        for i, token in enumerate(tokens):
            if i > 0 and re.match(r'\w', token) and re.match(r'\w', tokens[i-1]):
                result.append(' ')
            result.append(token)

        return ''.join(result)

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                    padding: bool = True, truncation: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            List of encoded token ID sequences
        """
        encoded = []
        for text in texts:
            token_ids = self.encode(text)
            encoded.append(token_ids)

        if max_length is not None:
            processed = []
            for seq in encoded:
                if truncation and len(seq) > max_length:
                    seq = seq[:max_length]
                if padding and len(seq) < max_length:
                    pad_length = max_length - len(seq)
                    seq = seq + [self.special_tokens['<pad>']] * pad_length
                processed.append(seq)
            encoded = processed

        return encoded

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def save_vocab(self, filepath: str):
        """Save vocabulary to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for token, idx in self.vocab.items():
                f.write(f"{token}\t{idx}\n")

    def load_vocab(self, filepath: str):
        """Load vocabulary from file."""
        self.vocab.clear()
        self.inverse_vocab.clear()

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx_str = line.strip().split('\t')
                idx = int(idx_str)
                self.vocab[token] = idx
                self.inverse_vocab[idx] = token

    def __len__(self) -> int:
        return len(self.vocab)