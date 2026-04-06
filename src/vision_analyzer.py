"""
Vision AI analyzer — processes image datasets using Ollama vision models.
"""

import requests
import base64
import os
import json
from pathlib import Path
from typing import Generator
import pandas as pd


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

VISION_MODELS = {
    "llama3.2-vision:11b": "Best quality — 11B params (needs ~12GB RAM)",
    "llava:13b":           "Great accuracy — 13B params (needs ~10GB RAM)",
    "llava:7b":            "Fast & good — 7B params (needs ~6GB RAM)",
    "minicpm-v":           "Excellent for charts/diagrams (needs ~6GB RAM)",
    "llava-phi3":          "Lightweight — fast responses (needs ~4GB RAM)",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def get_available_vision_models() -> list[str]:
    """Return vision models that are pulled locally."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code == 200:
            local = [m["name"] for m in r.json().get("models", [])]
            return [m for m in local
                    if any(v in m for v in
                           ["llava", "vision", "minicpm", "bakllava"])]
    except Exception:
        pass
    return []


def encode_image(image_path: str) -> str:
    """Convert image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: str, prompt: str,
                  model: str = "llava:7b") -> str:
    """Send image + prompt to Ollama vision model."""
    try:
        b64 = encode_image(image_path)
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload, timeout=120
        )
        if r.status_code == 200:
            return r.json().get("response", "No response")
        return f"Error {r.status_code}"
    except Exception as e:
        return f"Error: {e}"


def analyze_image_stream(image_path: str, prompt: str,
                          model: str = "llava:7b") -> Generator[str, None, None]:
    """Stream vision model response."""
    try:
        b64 = encode_image(image_path)
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": True,
        }
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload, stream=True, timeout=120
        ) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
    except Exception as e:
        yield f"Error: {e}"


def scan_image_folder(folder_path: str) -> list[dict]:
    """Scan folder and return list of image files with metadata."""
    images = []
    folder = Path(folder_path)
    if not folder.exists():
        return images
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            size_kb = round(f.stat().st_size / 1024, 1)
            images.append({
                "filename": f.name,
                "path":     str(f),
                "ext":      f.suffix.lower(),
                "size_kb":  size_kb,
            })
    return images


def batch_analyze(image_paths: list[str], prompt: str,
                   model: str = "llava:7b") -> pd.DataFrame:
    """Analyze multiple images and return results as DataFrame."""
    results = []
    for path in image_paths:
        result = analyze_image(path, prompt, model)
        results.append({
            "Image": Path(path).name,
            "Analysis": result,
        })
    return pd.DataFrame(results)