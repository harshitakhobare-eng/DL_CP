"""Image Preprocessing and Line Segmentation for Handwritten Mathematical Expressions.

Implements:
- Aspect-ratio preserving resizing with padding
- Grayscale conversion and stroke contrast enhancement
- Whitespace cropping
- Normalization
- Multi-line step extraction via horizontal projection profile
"""

from __future__ import annotations
import io
import math
from typing import List, Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as TF


def load_image(image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray]) -> Image.Image:
    """Load an image from various input types into a PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.copy()
    if isinstance(image_input, str):
        return Image.open(image_input)
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input))
    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        return Image.open(image_input)
    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255).astype(np.uint8)
        return Image.fromarray(image_input)
    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def crop_whitespace(image: Image.Image, threshold: int = 245, pad: int = 8) -> Image.Image:
    """Crop redundant whitespace around handwritten strokes.
    
    Detects ink bounding box using pixel intensity threshold.
    Adds a safe padding margin.
    If the image is completely blank, returns the original image.
    """
    gray = image.convert("L")
    arr = np.array(gray)
    
    # Ink pixels are darker than background
    mask = arr < threshold
    if not np.any(mask):
        # Blank image
        return image
    
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    # Add padding
    h, w = arr.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h, y1 + pad)
    x1 = min(w, x1 + pad)
    
    return image.crop((x0, y0, x1, y1))


def resize_and_pad(
    image: Image.Image,
    target_height: int = 128,
    target_width: int = 512,
    pad_color: int = 255
) -> Image.Image:
    """Resize image preserving exact aspect ratio, then pad to target dimensions.
    
    Does NOT stretch or distort the handwritten strokes.
    Centers the content vertically and places it with padding.
    """
    gray = image.convert("L")
    w, h = gray.size
    
    if w <= 0 or h <= 0:
        return Image.new("L", (target_width, target_height), color=pad_color)
    
    # Calculate scale factor to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    # High-quality antialiasing resize
    resized = gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create canvas with target dimensions
    canvas = Image.new("L", (target_width, target_height), color=pad_color)
    
    # Center vertically and horizontally
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    
    return canvas


def preprocess_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    target_height: int = 128,
    target_width: int = 512,
    whitespace_threshold: int = 245,
    pad_margin: int = 8,
    normalize_mean: float = 0.5,
    normalize_std: float = 0.5
) -> Tuple[torch.Tensor, Image.Image]:
    """Complete image preprocessing pipeline for handwriting recognition.
    
    Returns:
        tensor: Normalized PyTorch tensor of shape (1, 1, target_height, target_width)
        preprocessed_pil: Preprocessed PIL Image (for inspection and UI display)
    """
    img = load_image(image_input)
    
    # Handle alpha channel (RGBA -> RGB on white background)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, alpha_img).convert("RGB")
    else:
        img = img.convert("RGB")
        
    # Crop whitespace
    cropped = crop_whitespace(img, threshold=whitespace_threshold, pad=pad_margin)
    
    # Aspect-ratio preserving resize and pad
    padded = resize_and_pad(cropped, target_height=target_height, target_width=target_width)
    
    # Convert to PyTorch Tensor: shape (1, H, W), range [0.0, 1.0]
    tensor = TF.to_tensor(padded)  # (1, H, W)
    
    # Invert so ink strokes are positive signals (~1.0) and background is ~0.0
    # Standard in math OCR architectures
    tensor = 1.0 - tensor
    
    # Normalize: (x - mean) / std
    tensor = (tensor - normalize_mean) / normalize_std
    
    # Add batch dimension -> (1, 1, H, W)
    tensor = tensor.unsqueeze(0)
    
    return tensor, padded


def segment_solution_steps(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    min_line_height: int = 15,
    valley_threshold_ratio: float = 0.15
) -> List[Image.Image]:
    """Segment a multi-line handwritten solution into individual step images.
    
    Uses horizontal projection profile to detect valleys between handwritten lines.
    Returns a list of PIL Images, one per recognized step line.
    """
    img = load_image(image_input).convert("L")
    cropped = crop_whitespace(img)
    arr = np.array(cropped)
    h, w = arr.shape
    
    if h < min_line_height * 2:
        return [cropped]
    
    # Binarize: ink = 1, background = 0
    binary = (arr < 230).astype(np.float32)
    
    # Horizontal projection profile: sum ink pixels along each row
    proj = binary.sum(axis=1)
    
    if proj.max() == 0:
        return [cropped]
    
    # Smooth projection with moving average
    kernel_size = max(5, h // 40)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(proj, kernel, mode="same")
    
    # Peak-valley segmentation
    max_val = smoothed.max()
    threshold = max_val * valley_threshold_ratio
    
    # Find active row bands where ink is present
    is_ink_row = smoothed > threshold
    
    # Group contiguous rows
    bands: List[Tuple[int, int]] = []
    in_band = False
    start_y = 0
    
    for y in range(h):
        if is_ink_row[y] and not in_band:
            in_band = True
            start_y = y
        elif not is_ink_row[y] and in_band:
            in_band = False
            if y - start_y >= min_line_height:
                bands.append((start_y, y))
                
    if in_band and (h - start_y >= min_line_height):
        bands.append((start_y, h))
        
    if len(bands) <= 1:
        return [cropped]
        
    # Crop each detected step line with extra margin
    margin = 4
    step_images: List[Image.Image] = []
    for y0, y1 in bands:
        sy0 = max(0, y0 - margin)
        sy1 = min(h, y1 + margin)
        step_crop = cropped.crop((0, sy0, w, sy1))
        # Trim horizontal margins
        step_crop = crop_whitespace(step_crop)
        step_images.append(step_crop)
        
    return step_images
