"""Input degradations for the robustness slice.

Three perturbations, each modeling a real scan defect:
  - blur    : Gaussian blur (out-of-focus / low-quality scan)
  - skew    : rotation (page not laid flat on the scanner)
  - lowres  : downscale-then-upscale (low-DPI capture)

Implemented with Pillow + numpy so they run anywhere (no OpenCV needed). They
are intentionally simple and legible — the point is to move the metrics in a
controlled, explainable way, not to simulate every artifact.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr if arr.ndim == 3 else np.stack([arr] * 3, -1))


def blur(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Gaussian blur; `kernel` maps to a blur radius of kernel/2."""
    radius = max(0.5, kernel / 2.0)
    return np.array(_to_pil(arr).filter(ImageFilter.GaussianBlur(radius)))


def skew(arr: np.ndarray, degrees: float = 7.0) -> np.ndarray:
    """Rotate by `degrees`, filling exposed corners with white (paper)."""
    img = _to_pil(arr).rotate(
        degrees, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
    )
    return np.array(img)


def lowres(arr: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downscale by `scale` then upscale back — destroys fine detail at original size."""
    img = _to_pil(arr)
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.array(small.resize((w, h), Image.BILINEAR))


_FUNCS = {"blur": blur, "skew": skew, "lowres": lowres}


def apply_perturbations(arr: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Apply every perturbation configured under robustness.perturbations.
    Returns {perturbation_name: degraded_image}."""
    spec = config.get("robustness", {}).get("perturbations", {})
    out: dict[str, np.ndarray] = {}
    for name, params in spec.items():
        fn = _FUNCS.get(name)
        if fn is not None:
            out[name] = fn(arr, **(params or {}))
    return out
