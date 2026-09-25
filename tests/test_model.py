"""Tests for Vision-to-LaTeX Model Forward Pass and Generation."""

import torch
import pytest
from ml.model import MathFormulaRecognitionModel
from ml.tokenizer import LatexTokenizer


def test_model_forward_pass():
    tokenizer = LatexTokenizer()
    model = MathFormulaRecognitionModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        max_seq_len=32,
    )
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 1, 128, 512)
    tgt_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, 10))

    logits = model(images, tgt_tokens)
    assert logits.shape == (batch_size, 10, tokenizer.vocab_size)


def test_model_loss_computation():
    tokenizer = LatexTokenizer()
    model = MathFormulaRecognitionModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        max_seq_len=32,
    )
    batch_size = 2
    images = torch.randn(batch_size, 1, 128, 512)
    tgt_tokens = torch.randint(1, tokenizer.vocab_size, (batch_size, 10))

    logits = model(images, tgt_tokens[:, :-1])
    loss = model.compute_loss(logits, tgt_tokens[:, 1:])
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_model_generate():
    tokenizer = LatexTokenizer()
    model = MathFormulaRecognitionModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        max_seq_len=16,
    )
    model.eval()

    images = torch.randn(1, 1, 128, 512)
    results = model.generate(images, max_len=8)
    assert len(results) == 1
    res = results[0]
    assert "token_ids" in res
    assert "overall_confidence" in res
    assert 0.0 <= res["overall_confidence"] <= 1.0
