#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cascade-eval setup: gets a fresh machine from clone to runnable.
# Installs the one true system dependency (Tesseract + language data),
# then the Python environment via uv (preferred) or pip (fallback).
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> cascade-eval setup"

# --- 1. System dependency: Tesseract + traineddata --------------------------
# Tesseract is a C++ binary, not a pip package. We need language packs for:
#   eng (baseline), ara (Arabic / RTL), cop (Coptic, low-resource).
# 'msa'/Jawi support varies by distro; install if available, warn if not.
install_tesseract() {
    if command -v tesseract >/dev/null 2>&1; then
        echo "    tesseract already installed: $(tesseract --version | head -1)"
        return
    fi
    echo "    installing tesseract..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y tesseract-ocr \
            tesseract-ocr-ara tesseract-ocr-cop tesseract-ocr-eng \
            tesseract-ocr-script-arab || true
    elif command -v brew >/dev/null 2>&1; then
        brew install tesseract tesseract-lang
    else
        echo "    !! No apt-get or brew found. Install tesseract manually:" >&2
        echo "       https://tesseract-ocr.github.io/tessdoc/Installation.html" >&2
        exit 1
    fi
}
install_tesseract

# --- 2. Python environment --------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    echo "==> uv found — syncing environment"
    uv sync --extra llm || uv sync
    echo "    run with:  uv run python -m cascade_eval.run"
else
    echo "==> uv not found — falling back to venv + pip"
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[llm]" || pip install -e .
    echo "    run with:  source .venv/bin/activate && python -m cascade_eval.run"
fi

echo "==> setup complete."
