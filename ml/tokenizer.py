"""LaTeX Tokenizer for Mathematical Expressions.

Supports standard mathematical LaTeX tokens, structural markers,
digits, variables, Greek symbols, and special tokens (PAD, BOS, EOS, UNK).
"""

from __future__ import annotations
import json
import os
import re
from typing import List, Dict, Optional, Union

# Special Token Constants
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Standard Mathematical LaTeX Vocabulary
DEFAULT_MATH_TOKENS = [
    # LaTeX Commands & Operators
    r"\frac", r"\sqrt", r"\cdot", r"\times", r"\div", r"\pm", r"\mp",
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\theta", r"\lambda", r"\pi", r"\sigma",
    r"\sin", r"\cos", r"\tan", r"\csc", r"\sec", r"\cot",
    r"\arcsin", r"\arccos", r"\arctan",
    r"\log", r"\ln", r"\exp",
    r"\int", r"\iint", r"\sum", r"\prod", r"\lim", r"\infty",
    r"\partial", r"\Delta", r"\nabla",
    r"\left", r"\right",
    # Delimiters and grouping
    "{", "}", "(", ")", "[", "]", "|",
    # Superscripts and subscripts
    "^", "_",
    # Operators
    "+", "-", "*", "/", "=", "<", ">", "!", ":", ";", ",", ".",
]

# Add digits 0-9
DEFAULT_DIGITS = [str(i) for i in range(10)]

# Add lower-case and upper-case letters
DEFAULT_LETTERS = [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
                  [chr(c) for c in range(ord('A'), ord('Z') + 1)]


class LatexTokenizer:
    """Tokenizer designed specifically for mathematical LaTeX expressions."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.token2id = dict(vocab)
        else:
            self.token2id = self._build_default_vocab()
        
        self.id2token = {v: k for k, v in self.token2id.items()}

        # Tokenization regex: LaTeX commands (e.g. \frac), single letters, digits, or single punctuation
        self._pattern = re.compile(
            r"(\\[a-zA-Z]+|\d+|[a-zA-Z]|\^|_|\{|\}|\(|\)|\[|\]|\+|-|=|/|<|>|\*|!|,|\.|\S)"
        )

    def _build_default_vocab(self) -> Dict[str, int]:
        vocab: Dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID,
        }
        idx = len(vocab)
        # Add math tokens
        for tok in DEFAULT_MATH_TOKENS:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
        # Add digits
        for d in DEFAULT_DIGITS:
            if d not in vocab:
                vocab[d] = idx
                idx += 1
        # Add letters
        for letter in DEFAULT_LETTERS:
            if letter not in vocab:
                vocab[letter] = idx
                idx += 1
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def tokenize(self, latex_str: str) -> List[str]:
        """Split a mathematical LaTeX string into tokens.
        
        Handles LaTeX macros (e.g. \\frac, \\sqrt), numbers, variables, and operators.
        """
        if not latex_str:
            return []
        
        # Clean extra whitespace
        latex_str = latex_str.strip()
        tokens: List[str] = []
        for match in self._pattern.finditer(latex_str):
            tok = match.group(0).strip()
            if tok:
                # If a multi-digit sequence was matched, optionally split into single digits for character-level precision
                # In standard math OCR (e.g. CROHME), multi-digits are split into single digits: '2', '5' for '25'
                if tok.isdigit() and len(tok) > 1:
                    tokens.extend(list(tok))
                else:
                    tokens.append(tok)
        return tokens

    def encode(self, latex_str: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a LaTeX string into a list of token IDs."""
        tokens = self.tokenize(latex_str)
        token_ids: List[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_id)
        
        for tok in tokens:
            token_ids.append(self.token2id.get(tok, self.unk_id))
            
        if add_special_tokens:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back into a LaTeX string."""
        tokens: List[str] = []
        for tid in token_ids:
            if tid == self.pad_id and remove_special_tokens:
                continue
            if tid == self.bos_id and remove_special_tokens:
                continue
            if tid == self.eos_id and remove_special_tokens:
                break
            tok = self.id2token.get(tid, UNK_TOKEN)
            if remove_special_tokens and tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            tokens.append(tok)
        
        # Reconstruct string with natural spacing for LaTeX commands
        result = []
        for i, tok in enumerate(tokens):
            if tok.startswith("\\"):
                # Ensure LaTeX command has a space after it if followed by a letter
                result.append(tok)
                if i + 1 < len(tokens) and (tokens[i + 1].isalpha() or tokens[i + 1].startswith("\\")):
                    result.append(" ")
            else:
                result.append(tok)
        return "".join(result)

    def add_token(self, token: str) -> int:
        """Add a new token to the vocabulary if not present."""
        if token not in self.token2id:
            new_id = len(self.token2id)
            self.token2id[token] = new_id
            self.id2token[new_id] = token
            return new_id
        return self.token2id[token]

    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> LatexTokenizer:
        """Load tokenizer from a JSON vocabulary file."""
        with open(filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
