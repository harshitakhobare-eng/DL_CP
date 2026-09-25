"""Tests for LaTeX Tokenizer."""

import pytest
from ml.tokenizer import LatexTokenizer, PAD_ID, BOS_ID, EOS_ID, UNK_ID


def test_tokenizer_initialization():
    tokenizer = LatexTokenizer()
    assert tokenizer.vocab_size > 50
    assert tokenizer.pad_id == PAD_ID
    assert tokenizer.bos_id == BOS_ID
    assert tokenizer.eos_id == EOS_ID
    assert tokenizer.unk_id == UNK_ID


def test_tokenize_basic_equation():
    tokenizer = LatexTokenizer()
    tokens = tokenizer.tokenize("2x + 5 = 15")
    assert tokens == ["2", "x", "+", "5", "=", "1", "5"]


def test_tokenize_math_commands():
    tokenizer = LatexTokenizer()
    tokens = tokenizer.tokenize(r"\frac{1}{x - 2} = 3")
    assert r"\frac" in tokens
    assert "{" in tokens
    assert "}" in tokens
    assert "-" in tokens


def test_tokenize_sqrt_and_powers():
    tokenizer = LatexTokenizer()
    tokens = tokenizer.tokenize(r"\sqrt{x + 1} = x^2")
    assert r"\sqrt" in tokens
    assert "^" in tokens


def test_encode_decode_roundtrip():
    tokenizer = LatexTokenizer()
    expr = "2x + 5 = 15"
    token_ids = tokenizer.encode(expr, add_special_tokens=True)
    assert token_ids[0] == tokenizer.bos_id
    assert token_ids[-1] == tokenizer.eos_id
    
    decoded = tokenizer.decode(token_ids, remove_special_tokens=True)
    assert "2" in decoded
    assert "x" in decoded
    assert "15" in decoded


def test_empty_string_handling():
    tokenizer = LatexTokenizer()
    assert tokenizer.tokenize("") == []
    assert tokenizer.encode("", add_special_tokens=False) == []
    assert tokenizer.decode([]) == ""
