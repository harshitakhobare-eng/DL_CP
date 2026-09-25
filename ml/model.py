"""Vision-to-LaTeX Neural Architecture: CNN Visual Encoder + Transformer Decoder.

Implements:
- 2D Visual Encoder with convolutional feature extraction
- 2D Spatial Positional Encodings
- Autoregressive Transformer Decoder with Causal Masking
- Teacher Forcing and Padding-Aware Loss
- Confidence Estimation and Low-Confidence Token Detection
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ml.tokenizer import PAD_ID, BOS_ID, EOS_ID


class PositionalEncoding2D(nn.Module):
    """2D Positional Encoding for visual feature maps."""

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 128):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4, got {d_model}")
        
        d_sub = d_model // 2
        pe = torch.zeros(max_h, max_w, d_model)
        
        # 1D encodings for Y and X
        div_term = torch.exp(torch.arange(0, d_sub, 2).float() * (-math.log(10000.0) / d_sub))
        
        pos_y = torch.arange(0, max_h).unsqueeze(1).float()
        pos_x = torch.arange(0, max_w).unsqueeze(1).float()
        
        pe_y = torch.zeros(max_h, d_sub)
        pe_y[:, 0::2] = torch.sin(pos_y * div_term)
        pe_y[:, 1::2] = torch.cos(pos_y * div_term)
        
        pe_x = torch.zeros(max_w, d_sub)
        pe_x[:, 0::2] = torch.sin(pos_x * div_term)
        pe_x[:, 1::2] = torch.cos(pos_x * div_term)
        
        pe[:, :, :d_sub] = pe_y.unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, d_sub:] = pe_x.unsqueeze(0).repeat(max_h, 1, 1)
        
        # Shape: (1, max_h * max_w, d_model)
        self.register_buffer("pe", pe.view(1, max_h * max_w, d_model))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Add 2D positional embeddings to flattened feature sequence.
        
        Args:
            x: (B, h * w, d_model)
        """
        # Extract corresponding grid
        pe_grid = self.pe[:, : h * w, :]
        return x + pe_grid


class VisualCNNEncoder(nn.Module):
    """Convolutional Visual Encoder extracting 2D visual feature maps."""

    def __init__(self, in_channels: int = 1, d_model: int = 256):
        super().__init__()
        # Input: (B, 1, 128, 512)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (64, 64, 256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (128, 32, 128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Downsample height more than width to preserve horizontal math structure
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # -> (256, 16, 64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # -> (d_model, 8, 32)
        )
        self.pos_encoder = PositionalEncoding2D(d_model=d_model, max_h=32, max_w=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract visual token sequences.
        
        Args:
            x: (B, in_channels, H, W)
        Returns:
            features: (B, S, d_model) where S = H' * W'
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (B, d_model, H', W')
        
        b, c, h, w = x.shape
        # Permute to (B, H'*W', d_model)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoder(x, h, w)
        return x


class TransformerMathDecoder(nn.Module):
    """Autoregressive Transformer Decoder generating LaTeX token sequences."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for teacher forcing.
        
        Args:
            tgt: Target token IDs (B, T)
            memory: Visual encoder features (B, S, d_model)
            tgt_mask: Causal mask (T, T)
            tgt_key_padding_mask: Padding mask (B, T)
        Returns:
            logits: (B, T, vocab_size)
        """
        b, t = tgt.shape
        device = tgt.device
        
        positions = torch.arange(0, t, device=device).unsqueeze(0).repeat(b, 1)
        x = self.token_embedding(tgt) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t, device=device)
            
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc_out(out)
        return logits


class MathFormulaRecognitionModel(nn.Module):
    """End-to-End Vision-to-LaTeX Mathematical Formula Recognition Model."""

    def __init__(
        self,
        vocab_size: int,
        in_channels: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.encoder = VisualCNNEncoder(in_channels=in_channels, d_model=d_model)
        self.decoder = TransformerMathDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing forward pass.
        
        Args:
            images: (B, 1, H, W)
            tgt_tokens: (B, T)
        Returns:
            logits: (B, T, vocab_size)
        """
        # Memory features from CNN visual encoder
        memory = self.encoder(images)  # (B, S, d_model)
        
        # Target key padding mask: True where token is PAD
        tgt_key_padding_mask = (tgt_tokens == PAD_ID)
        
        logits = self.decoder(
            tgt=tgt_tokens,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute padding-aware cross entropy loss.
        
        Args:
            logits: (B, T, vocab_size)
            targets: (B, T)
        """
        # Shift so predictions at step t are compared to target at step t+1
        # For standard teacher forcing: targets is (B, T)
        b, t, v = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, v),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: Optional[int] = None,
        confidence_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Autoregressive sequence decoding with confidence estimation.
        
        Args:
            images: (B, 1, H, W)
            max_len: Maximum generation steps
            confidence_threshold: Threshold to flag low-confidence tokens
        Returns:
            results: List of dicts with token_ids, token_confidences, overall_confidence,
                     low_confidence_indices
        """
        self.eval()
        device = images.device
        b = images.size(0)
        max_len = max_len or self.max_seq_len

        memory = self.encoder(images)
        
        # Start with <BOS> token
        cur_tokens = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        
        # Store probabilities
        all_token_probs = [[] for _ in range(b)]
        generated_ids = [[] for _ in range(b)]

        for step in range(max_len):
            logits = self.decoder(tgt=cur_tokens, memory=memory)
            next_logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(next_logits, dim=-1)  # (B, vocab_size)
            
            top_probs, next_tokens = torch.max(probs, dim=-1)  # (B,), (B,)
            
            for i in range(b):
                if not finished[i]:
                    token_id = next_tokens[i].item()
                    prob = top_probs[i].item()
                    
                    if token_id == EOS_ID:
                        finished[i] = True
                    else:
                        generated_ids[i].append(token_id)
                        all_token_probs[i].append(prob)
                        
            if finished.all():
                break
                
            cur_tokens = torch.cat([cur_tokens, next_tokens.unsqueeze(1)], dim=1)

        # Build output objects
        results = []
        for i in range(b):
            t_ids = generated_ids[i]
            probs = all_token_probs[i]
            
            if len(probs) > 0:
                overall_conf = float(np.mean(probs))
            else:
                overall_conf = 1.0 if len(t_ids) == 0 else 0.0

            low_conf_tokens = [
                {"token_id": tid, "confidence": p, "step_index": idx}
                for idx, (tid, p) in enumerate(zip(t_ids, probs))
                if p < confidence_threshold
            ]

            results.append({
                "token_ids": t_ids,
                "token_confidences": probs,
                "overall_confidence": round(overall_conf, 4),
                "low_confidence_tokens": low_conf_tokens,
            })

        return results
