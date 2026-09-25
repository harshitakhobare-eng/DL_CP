"""Dataset loader and PyTorch Dataset for handwritten mathematical expressions.

Connects to Hugging Face 'deepcopy/MathWriting-human'.
Supports train/validation/test splits, subset slicing, local caching,
and provides custom DataLoader collate functions.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.tokenizer import LatexTokenizer, PAD_ID
from ml.preprocessing import preprocess_image, resize_and_pad

logger = logging.getLogger(__name__)


class MathWritingDataset(Dataset):
    """PyTorch Dataset wrapping handwritten math expression samples."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: LatexTokenizer,
        max_seq_len: int = 128,
        target_height: int = 128,
        target_width: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_data = item.get("image")
        latex_str = item.get("latex", "")

        # Preprocess image into tensor
        try:
            tensor, _ = preprocess_image(
                image_data,
                target_height=self.target_height,
                target_width=self.target_width,
            )
            # Squeeze batch dimension added by preprocess_image -> (1, H, W)
            image_tensor = tensor.squeeze(0)
        except Exception as e:
            logger.warning(f"Error preprocessing image at index {idx}: {e}. Using blank.")
            image_tensor = torch.zeros((1, self.target_height, self.target_width), dtype=torch.float32)

        # Encode LaTeX to token IDs with <BOS> and <EOS>
        token_ids = self.tokenizer.encode(latex_str, add_special_tokens=True)
        if len(token_ids) > self.max_seq_len:
            # Truncate and ensure last token is EOS
            token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]

        return {
            "image": image_tensor,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "latex": latex_str,
            "sample_id": item.get("sample_id", f"sample_{idx}"),
        }


def math_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function to batch variable-length token sequences and images."""
    images = torch.stack([item["image"] for item in batch], dim=0)  # (B, 1, H, W)
    latex_list = [item["latex"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    # Pad token sequences to maximum length in batch
    token_tensors = [item["token_ids"] for item in batch]
    max_len = max(len(t) for t in token_tensors)

    padded_tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, t in enumerate(token_tensors):
        padded_tokens[i, :len(t)] = t

    return {
        "images": images,
        "token_ids": padded_tokens,
        "latex": latex_list,
        "sample_ids": sample_ids,
    }


def generate_synthetic_math_image(latex_expr: str, width: int = 512, height: int = 128) -> Image.Image:
    """Generate a clean synthetic mathematical image for testing and offline fallback."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Render readable text representation
    render_text = latex_expr.replace(r"\frac", "").replace("{", "(").replace("}", ")")
    render_text = render_text.replace(r"\sqrt", "√").replace(r"\times", "×")
    render_text = render_text.replace(r"\cdot", "·").replace(r"\pm", "±")
    
    # Try to draw with default font, sized proportionally
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    bbox = draw.textbbox((0, 0), render_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = max(10, (width - text_w) // 2)
    y = max(10, (height - text_h) // 2)
    
    # Draw stroke multiple times with slight jitter to simulate pen strokes
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        draw.text((x + dx, y + dy), render_text, fill=20, font=font)
        
    return img


def load_math_dataset(
    dataset_name: str = "deepcopy/MathWriting-human",
    cache_dir: Optional[str] = r"D:\HFCache\datasets",  # Cached on D drive, outside project
    train_subset: Optional[int] = 5000,
    val_subset: Optional[int] = 500,
    test_subset: Optional[int] = 500,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load the MathWriting-human dataset from Hugging Face.
    
    Automatically creates train, validation, and test splits.
    Supports development subsets.
    If HF is offline or inaccessible, provides clear guidance and falls back to synthetic offline data.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    try:
        from datasets import load_dataset

        logger.info(f"Connecting to Hugging Face dataset '{dataset_name}' (cache: {cache_dir})...")
        
        # Determine subset slices if specified
        train_split = f"train[:{train_subset}]" if train_subset else "train"
        val_split = f"validation[:{val_subset}]" if val_subset else "validation"
        test_split = f"test[:{test_subset}]" if test_subset else "test"

        # Attempt to load dataset
        hf_dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        # Check split tags or standard splits
        available_splits = list(hf_dataset.keys())
        logger.info(f"Successfully loaded HF dataset. Available splits: {available_splits}")

        train_ds = hf_dataset["train"]
        num_train = min(len(train_ds), train_subset) if train_subset else len(train_ds)
        train_data = [train_ds[i] for i in range(num_train)]

        val_key = "val" if "val" in hf_dataset else ("validation" if "validation" in hf_dataset else None)
        if val_key:
            val_ds = hf_dataset[val_key]
            num_val = min(len(val_ds), val_subset) if val_subset else len(val_ds)
            val_data = [val_ds[i] for i in range(num_val)]
        else:
            split_idx = int(len(train_data) * 0.9)
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]

        test_key = "test" if "test" in hf_dataset else None
        if test_key:
            test_ds = hf_dataset[test_key]
            num_test = min(len(test_ds), test_subset) if test_subset else len(test_ds)
            test_data = [test_ds[i] for i in range(num_test)]
        else:
            test_data = val_data[:min(len(val_data), test_subset or 50)]

        logger.info(f"Loaded dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.")
        return train_data, val_data, test_data

    except Exception as e:
        logger.error(
            "\n" + "=" * 70 + "\n"
            "NOTE ON HUGGING FACE DATASET ACCESS:\n"
            f"Could not automatically download '{dataset_name}' from Hugging Face.\n"
            f"Details: {e}\n\n"
            "If this is due to internet connectivity or Hugging Face access, you can:\n"
            "1. Run `huggingface-cli login` to provide your Hugging Face API token.\n"
            "2. Ensure an active internet connection to huggingface.co.\n"
            "For offline development and immediate testing, generating synthetic benchmark dataset.\n"
            + "=" * 70 + "\n"
        )
        return create_synthetic_benchmark_dataset(
            num_train=train_subset or 200,
            num_val=val_subset or 50,
            num_test=test_subset or 50,
        )


def create_synthetic_benchmark_dataset(
    num_train: int = 200,
    num_val: int = 50,
    num_test: int = 50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a realistic mathematical expression dataset for testing and offline runs."""
    expressions = [
        "2x + 5 = 15",
        "x^2 - 5x + 6 = 0",
        "3(x + 4) = 21",
        r"\frac{1}{x - 2} = 3",
        r"\sqrt{x + 1} = 4",
        r"\frac{x + 1}{x - 2} = 2",
        "2x = 10",
        "x = 5",
        "3x + 12 = 21",
        "3x = 9",
        "x = 3",
        "x^2 + 2x + 1 = 0",
        "x - 4 = 10",
        "2x + 3 = 10",
        "5(x - 2) = 30",
        "-2(x + 6) = 10",
        "4x - 8 = 16",
        r"\frac{2x + 4}{2} = 8",
        "x^2 - 9 = 0",
        "2(x + 3) = 14",
    ]

    def _build_split(count: int, prefix: str) -> List[Dict[str, Any]]:
        samples = []
        for i in range(count):
            latex_expr = expressions[i % len(expressions)]
            img = generate_synthetic_math_image(latex_expr)
            samples.append({
                "image": img,
                "latex": latex_expr,
                "sample_id": f"{prefix}_{i}",
                "split_tag": prefix,
                "data_type": "synthetic_benchmark",
            })
        return samples

    return (
        _build_split(num_train, "train"),
        _build_split(num_val, "val"),
        _build_split(num_test, "test"),
    )
