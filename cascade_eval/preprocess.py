"""Light image preprocessing shared by all engines.

Deliberately conservative: grayscale + contrast normalization always run
(numpy only), and deskew runs when OpenCV is available, falling back cleanly if
not. We keep this minimal on purpose — heavy preprocessing would mask the
engines' real robustness, which the degradation slice (Step 7) is meant to
measure honestly.
"""

from __future__ import annotations

import importlib.util

import numpy as np


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """RGB(A)/grayscale ndarray -> 2D uint8 grayscale."""
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    rgb = arr[:, :, :3].astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # luminosity
    return gray.astype(np.uint8)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Linear min-max stretch to use the full 0-255 range."""
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 1e-6:
        return gray
    stretched = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return stretched.clip(0, 255).astype(np.uint8)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate and correct small skew. Returns (image, angle_degrees).

    Uses OpenCV when present; otherwise returns the image unchanged with angle
    0.0 so the pipeline still runs on a machine without cv2."""
    if importlib.util.find_spec("cv2") is None:
        return gray, 0.0
    import cv2

    inv = 255 - gray  # text -> foreground
    coords = np.column_stack(np.where(inv > 0))
    if coords.shape[0] < 10:
        return gray, 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:  # ignore negligible skew
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, float(angle)


def preprocess(arr: np.ndarray, do_deskew: bool = True) -> np.ndarray:
    """Full pipeline: grayscale -> normalize -> (optional) deskew.
    Returns a 3-channel image so every engine accepts it uniformly."""
    gray = normalize_contrast(to_grayscale(arr))
    if do_deskew:
        gray, _angle = deskew(gray)
    return np.stack([gray, gray, gray], axis=-1)  # back to 3-channel
