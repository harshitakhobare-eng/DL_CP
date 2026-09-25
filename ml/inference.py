"""Inference engine for Handwritten Mathematical Expression Recognition.

Loads trained checkpoints, handles preprocessing, autoregressive decoding,
confidence estimation, and uncertain token flagging.
"""

from __future__ import annotations
import base64
import io
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import torch

from ml.tokenizer import LatexTokenizer
from ml.preprocessing import preprocess_image, load_image
from ml.model import MathFormulaRecognitionModel

logger = logging.getLogger(__name__)


class MathRecognizer:
    """Inference recognizer for mathematical expression images."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.6,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.confidence_threshold = confidence_threshold
        self.model: Optional[MathFormulaRecognitionModel] = None
        self.tokenizer: LatexTokenizer = LatexTokenizer()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
        else:
            # Initialize with default architecture
            logger.info("Initializing MathRecognizer with base architecture (no checkpoint specified).")
            self.model = MathFormulaRecognitionModel(
                vocab_size=self.tokenizer.vocab_size,
                d_model=256,
                max_seq_len=128,
            ).to(self.device)
            self.model.eval()

    def load(self, checkpoint_path: str) -> None:
        """Load model weights and vocabulary from a checkpoint."""
        logger.info(f"Loading recognition model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.tokenizer = LatexTokenizer(vocab=checkpoint["vocab"])
        
        self.model = MathFormulaRecognitionModel(
            vocab_size=checkpoint["vocab_size"],
            d_model=checkpoint.get("d_model", 256),
            max_seq_len=checkpoint.get("max_seq_len", 128),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info("Checkpoint loaded successfully.")

    def recognize(
        self,
        image_input: Union[str, bytes, io.BytesIO, Image.Image],
    ) -> Dict[str, Any]:
        """Recognize mathematical expression from an image.
        
        Returns:
            Dict containing:
                - latex: Recognized LaTeX string
                - confidence: Float overall confidence [0, 1]
                - tokens: List of recognized token strings
                - token_confidences: List of individual token confidence scores
                - low_confidence_tokens: List of tokens with low confidence
                - preprocessed_base64: Base64-encoded preprocessed image
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Preprocess input image
        tensor, preprocessed_pil = preprocess_image(image_input)
        tensor = tensor.to(self.device)

        # Autoregressive generation
        gen_outputs = self.model.generate(
            tensor,
            confidence_threshold=self.confidence_threshold,
        )
        result = gen_outputs[0]

        token_ids = result["token_ids"]
        token_probs = result["token_confidences"]
        overall_conf = result["overall_confidence"]

        # Decode tokens to LaTeX string
        latex_str = self.tokenizer.decode(token_ids, remove_special_tokens=True)
        tokens = [self.tokenizer.id2token.get(tid, "") for tid in token_ids]

        # Format low-confidence tokens for user review
        low_confidence_tokens = []
        for low_info in result["low_confidence_tokens"]:
            tid = low_info["token_id"]
            tok_str = self.tokenizer.id2token.get(tid, "")
            low_confidence_tokens.append({
                "token": tok_str,
                "confidence": round(low_info["confidence"], 4),
                "position": low_info["step_index"],
                "warning": f"Uncertain token '{tok_str}' (confidence: {round(low_info['confidence'] * 100, 1)}%)",
            })

        # Encode preprocessed image to base64
        buffer = io.BytesIO()
        preprocessed_pil.save(buffer, format="PNG")
        b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "latex": latex_str,
            "confidence": overall_conf,
            "tokens": tokens,
            "token_confidences": [round(p, 4) for p in token_probs],
            "low_confidence_tokens": low_confidence_tokens,
            "preprocessed_image_base64": b64_img,
        }
