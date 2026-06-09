"""Engine adapter interface.

Every engine exposes the same contract: `recognize(image, page) -> EngineResult`.
Heavy dependencies (pytesseract, easyocr, anthropic) are imported lazily inside
each adapter so the package stays importable on a machine where an engine isn't
installed — the engine simply reports `is_available == False` and `recognize`
returns an EngineResult with `available=False` and a reason. Unavailability is
a logged outcome, never a crash.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from ..schemas import EngineName, EngineResult

ImageInput = Union[str, Path, "np.ndarray"]


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no extra dependency). Only sets vars not already
    present in the environment, so real env always wins."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and val and key not in os.environ:
            os.environ[key] = val


class Engine(ABC):
    """Base class for all OCR/HTR engines."""

    name: EngineName

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this engine can actually run (deps installed, key present)."""

    @abstractmethod
    def recognize(self, image: ImageInput, page: str) -> EngineResult:
        """Run recognition on one page image."""

    # --- shared helpers -----------------------------------------------------

    def _unavailable(self, page: str, note: str) -> EngineResult:
        """Uniform 'engine could not run' result — recorded, not raised."""
        return EngineResult(
            page=page, engine=self.name, text="", confidence=None,
            available=False, note=note,
        )

    @staticmethod
    def _to_ndarray(image: ImageInput) -> "np.ndarray":
        """Load any supported input into an RGB uint8 ndarray."""
        if isinstance(image, np.ndarray):
            return image
        from PIL import Image

        return np.array(Image.open(image).convert("RGB"))

    @staticmethod
    def _to_pil(image: ImageInput):
        from PIL import Image

        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return Image.open(image).convert("RGB")
