"""Model Training and Evaluation Pipeline.

Implements:
- Teacher forcing training loop
- AdamW optimizer with Cosine Annealing learning rate schedule
- Gradient clipping
- Validation metrics: Token Accuracy, Exact Match (EM), Character Error Rate (CER)
- Checkpoint saving and loading
- Automatic device fallback (CUDA -> CPU)
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.tokenizer import LatexTokenizer, PAD_ID, BOS_ID, EOS_ID
from ml.dataset import MathWritingDataset, math_collate_fn
from ml.model import MathFormulaRecognitionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute Character Error Rate (CER) across a batch."""
    total_dist = 0
    total_len = 0
    for pred, gt in zip(predictions, ground_truths):
        dist = compute_levenshtein_distance(pred, gt)
        total_dist += dist
        total_len += max(len(gt), 1)
    return total_dist / total_len if total_len > 0 else 0.0


class MathTrainer:
    """Trainer for Vision-to-LaTeX recognition model."""

    def __init__(
        self,
        model: MathFormulaRecognitionModel,
        tokenizer: LatexTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using compute device: {self.device}")

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip_val = gradient_clip_val
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-5,
        )

        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch with teacher forcing."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)          # (B, 1, H, W)
            token_ids = batch["token_ids"].to(self.device)    # (B, T)

            # Teacher forcing: input is token_ids[:, :-1], target is token_ids[:, 1:]
            tgt_input = token_ids[:, :-1]
            tgt_target = token_ids[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(images, tgt_input)            # (B, T-1, vocab_size)
            loss = self.model.compute_loss(logits, tgt_target)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(self.train_loader):
                logger.info(
                    f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Avg: {total_loss / (batch_idx + 1):.4f}"
                )

        avg_loss = total_loss / max(1, len(self.train_loader))
        elapsed = time.time() - start_time
        logger.info(f"Epoch [{epoch}] Finished in {elapsed:.1f}s - Avg Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation and compute loss, token accuracy, exact match (EM), and CER."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        
        all_pred_texts: List[str] = []
        all_gt_texts: List[str] = []

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)
            gt_latex = batch["latex"]

            # Compute validation loss
            tgt_input = token_ids[:, :-1]
            tgt_target = token_ids[:, 1:]
            logits = self.model(images, tgt_input)
            loss = self.model.compute_loss(logits, tgt_target)
            total_loss += loss.item()

            # Token-level accuracy on teacher-forced predictions
            preds = torch.argmax(logits, dim=-1)  # (B, T-1)
            mask = (tgt_target != PAD_ID)
            correct_tokens += ((preds == tgt_target) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Autoregressive generation for Exact Match & CER evaluation
            gen_results = self.model.generate(images)
            for item in gen_results:
                decoded_str = self.tokenizer.decode(item["token_ids"], remove_special_tokens=True)
                all_pred_texts.append(decoded_str)
            all_gt_texts.extend(gt_latex)

        val_loss = total_loss / max(1, len(self.val_loader))
        token_acc = correct_tokens / max(1, total_tokens)
        
        # Exact sequence match
        exact_matches = sum(p.strip() == gt.strip() for p, gt in zip(all_pred_texts, all_gt_texts))
        em_rate = exact_matches / max(1, len(all_gt_texts))

        # Character Error Rate
        cer = compute_cer(all_pred_texts, all_gt_texts)

        metrics = {
            "val_loss": round(val_loss, 4),
            "token_acc": round(token_acc, 4),
            "exact_match": round(em_rate, 4),
            "cer": round(cer, 4),
        }
        logger.info(f"Validation Metrics: {metrics}")
        return metrics

    def train(self, epochs: int = 10, save_best_only: bool = True) -> Dict[str, Any]:
        """Complete training loop across epochs with checkpointing."""
        best_val_loss = float("inf")
        best_em = 0.0

        for epoch in range(1, epochs + 1):
            logger.info(f"\n--- Starting Epoch {epoch}/{epochs} ---")
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                **metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(epoch_record)

            # Checkpoint condition: best validation loss or best exact match
            is_best = metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = metrics["val_loss"]
                best_em = metrics["exact_match"]
                self.save_checkpoint("best_model.pt", epoch, metrics)

            # Always save latest model after each epoch
            self.save_checkpoint("latest_model.pt", epoch, metrics)

            if not save_best_only:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, metrics)

        logger.info("Training complete.")
        return {
            "best_val_loss": best_val_loss,
            "best_exact_match": best_em,
            "history": self.history,
        }

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> str:
        """Save model checkpoint to disk."""
        save_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "vocab": self.tokenizer.token2id,
            "metrics": metrics,
            "d_model": self.model.d_model,
            "vocab_size": self.model.vocab_size,
            "max_seq_len": self.model.max_seq_len,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        return save_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> Tuple[MathFormulaRecognitionModel, LatexTokenizer, Dict[str, Any]]:
        """Load trained model and tokenizer from checkpoint file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
            
        map_location = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        tokenizer = LatexTokenizer(vocab=checkpoint["vocab"])
        model = MathFormulaRecognitionModel(
            vocab_size=checkpoint["vocab_size"],
            d_model=checkpoint.get("d_model", 256),
            max_seq_len=checkpoint.get("max_seq_len", 128),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(map_location)
        model.eval()
        
        return model, tokenizer, checkpoint
