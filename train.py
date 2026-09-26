"""CLI Training Entry Point for Handwritten Mathematical Expression Recognition.

Usage:
    python train.py --epochs 5 --batch-size 16 --train-subset 200
"""

from __future__ import annotations
import argparse
import logging
import os
import yaml
import torch
from torch.utils.data import DataLoader

from ml.tokenizer import LatexTokenizer
from ml.dataset import load_math_dataset, MathWritingDataset, math_collate_fn
from ml.model import MathFormulaRecognitionModel
from ml.training import MathTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Vision-to-LaTeX Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--train-subset", type=int, default=None, help="Subset size for training")
    parser.add_argument("--val-subset", type=int, default=None, help="Subset size for validation")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument(
        "--allow-synthetic-data",
        action="store_true",
        help="Allow the synthetic typed-expression fallback for smoke tests; unsuitable for handwriting recognition.",
    )
    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    prep_cfg = config.get("preprocessing", {})

    epochs = args.epochs or train_cfg.get("epochs", 5)
    batch_size = args.batch_size or dataset_cfg.get("batch_size", 16)
    lr = args.lr or train_cfg.get("lr", 0.0005)
    train_subset = args.train_subset or dataset_cfg.get("train_subset", 200)
    val_subset = args.val_subset or dataset_cfg.get("val_subset", 50)
    test_subset = dataset_cfg.get("test_subset", 50)
    checkpoint_dir = args.checkpoint_dir or train_cfg.get("checkpoint_dir", "checkpoints")

    logger.info("Initializing LaTeX Tokenizer...")
    tokenizer = LatexTokenizer()

    logger.info(f"Loading dataset: train={train_subset}, val={val_subset}, test={test_subset}...")
    train_samples, val_samples, _ = load_math_dataset(
        dataset_name=dataset_cfg.get("hf_dataset_name", "deepcopy/MathWriting-human"),
        cache_dir=dataset_cfg.get("cache_dir", "data/cache"),
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        allow_synthetic_fallback=args.allow_synthetic_data,
    )

    if not args.allow_synthetic_data and any(
        sample.get("data_type") == "synthetic_benchmark" for sample in train_samples
    ):
        raise RuntimeError("Synthetic benchmark data cannot be used to train the handwriting recognizer.")

    # PyTorch Datasets
    train_ds = MathWritingDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.get("max_seq_len", 128),
        target_height=prep_cfg.get("max_height", 128),
        target_width=prep_cfg.get("max_width", 512),
    )
    val_ds = MathWritingDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.get("max_seq_len", 128),
        target_height=prep_cfg.get("max_height", 128),
        target_width=prep_cfg.get("max_width", 512),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=math_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=math_collate_fn,
        num_workers=0,
    )

    logger.info("Initializing MathFormulaRecognitionModel...")
    model = MathFormulaRecognitionModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 4),
        num_decoder_layers=model_cfg.get("num_decoder_layers", 3),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg.get("max_seq_len", 128),
    )

    trainer = MathTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        device=train_cfg.get("device", "auto"),
    )

    logger.info(f"Starting training for {epochs} epochs...")
    results = trainer.train(epochs=epochs)
    logger.info(f"Training completed! Best val loss: {results['best_val_loss']}, Best exact match: {results['best_exact_match']}")


if __name__ == "__main__":
    main()
