"""Tests for Image Preprocessing and Line Segmentation."""

import numpy as np
from PIL import Image
import torch
import pytest

from ml.preprocessing import (
    crop_whitespace,
    resize_and_pad,
    preprocess_image,
    segment_solution_steps,
)


def test_crop_whitespace_blank():
    blank = Image.new("L", (100, 100), color=255)
    cropped = crop_whitespace(blank)
    assert cropped.size == (100, 100)


def test_crop_whitespace_with_stroke():
    img = Image.new("L", (200, 200), color=255)
    # Draw dark box in middle: (50, 50) to (150, 150)
    arr = np.array(img)
    arr[50:150, 50:150] = 0
    img = Image.fromarray(arr)
    
    cropped = crop_whitespace(img, pad=5)
    w, h = cropped.size
    # Expected size ~ 110 x 110 with pad=5
    assert 90 <= w <= 120
    assert 90 <= h <= 120


def test_resize_and_pad_aspect_ratio():
    # 200x50 image (4:1 aspect ratio)
    img = Image.new("L", (200, 50), color=200)
    padded = resize_and_pad(img, target_height=128, target_width=512)
    assert padded.size == (512, 128)


def test_preprocess_image_pipeline():
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    tensor, pil_img = preprocess_image(img, target_height=128, target_width=512)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 128, 512)
    assert pil_img.size == (512, 128)


def test_segment_solution_steps_single_line():
    img = Image.new("L", (300, 50), color=255)
    steps = segment_solution_steps(img)
    assert len(steps) >= 1


def test_segment_solution_steps_multi_line():
    # Create image with two separated horizontal bands
    arr = np.ones((200, 300), dtype=np.uint8) * 255
    arr[20:60, 50:250] = 0     # Line 1
    arr[120:160, 50:250] = 0   # Line 2
    img = Image.fromarray(arr)
    
    steps = segment_solution_steps(img)
    assert len(steps) == 2
