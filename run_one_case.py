import json
import os
from pathlib import Path
from typing import List, Optional, Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import requests
from pydantic import BaseModel


CASE_DIR = Path("case_data")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")


class PipelineSpec(BaseModel):
    name: str
    mask_path: str


class DicePlotArgs(BaseModel):
    task: Literal["dice_plot_one_case"] = "dice_plot_one_case"
    case_id: str
    gt_path: str
    pipelines: List[PipelineSpec]
    threshold: float = 0.5
    title: Optional[str] = None


def ollama_json_extract(user_text: str, available_files: list[str]) -> dict:
    schema_hint = {
        "task": "dice_plot_one_case",
        "case_id": "<string>",
        "gt_path": "<string path from available_files>",
        "pipelines": [{"name": "<string>", "mask_path": "<string path from available_files>"}],
        "threshold": 0.5,
        "title": "<optional string or null>",
    }

    prompt = f"""
You are a strict argument extractor. Return ONLY valid JSON (no markdown, no extra text).
Do not invent file paths. Use only paths from available_files.

available_files = {available_files}

JSON must match this example shape:
{json.dumps(schema_hint, indent=2)}

User request:
{user_text}
""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    return json.loads(text)


def load_mask(path: Path, thr: float) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return (data > thr).astype(np.uint8)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    denom = a.sum(dtype=np.float64) + b.sum(dtype=np.float64)
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def main() -> None:
    case_id_path = CASE_DIR / "case_id.txt"
    if not case_id_path.exists():
        raise FileNotFoundError(f"Missing {case_id_path}. Create it with one line, e.g. spleen_41")

    case_id = case_id_path.read_text(encoding="utf-8").strip()
    available = sorted([str(p.as_posix()) for p in CASE_DIR.glob("*.nii*")])
    if not available:
        raise FileNotFoundError("No .nii/.nii.gz files found in case_data")

    user_request = (
        f"Compute Dice for one case and plot a bar chart. "
        f"Case ID is {case_id}. Ground truth is case_data/gt.nii.gz. "
        f"Compare DeepSpleenSeg, JLF, TotalSegmentator if present."
    )

    args_dict = ollama_json_extract(user_request, available_files=available)
    args = DicePlotArgs.model_validate(args_dict)

    gt = load_mask(Path(args.gt_path), args.threshold)

    names, scores = [], []
    for p in args.pipelines:
        mp = Path(p.mask_path)
        if not mp.exists():
            print(f"[SKIP] Missing: {mp}")
            continue
        pred = load_mask(mp, args.threshold)
        if pred.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch: GT {gt.shape} vs {p.name} {pred.shape}")
        names.append(p.name)
        scores.append(dice(gt, pred))

    if not names:
        raise RuntimeError("No pipeline masks found to evaluate.")

    plt.figure(figsize=(7.5, 4.2))
    plt.bar(names, scores)
    plt.ylim(0, 1.0)
    plt.ylabel("Dice")
    plt.title(args.title or f"Dice (one case): {args.case_id}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = Path("dice_one_case.png")
    plt.savefig(out, dpi=200)
    print("Saved:", out.resolve())
    print("Results:")
    for n, s in zip(names, scores):
        print(f"  {n:18s}  Dice={s:.4f}")


if __name__ == "__main__":
    main()
