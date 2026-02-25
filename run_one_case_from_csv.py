#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import requests
from pydantic import BaseModel


DEFAULT_CSV_WSL = "/mnt/c/Users/adams/Documents/Projects/Spleen/data/abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv"
DEFAULT_CSV_WIN = r"C:\Users\adams\Documents\Projects\Spleen\data\abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


class DiceCsvArgs(BaseModel):
    task: Literal["dice_plot_one_case_from_csv"] = "dice_plot_one_case_from_csv"
    case_id: str
    pipelines: List[str]
    title: Optional[str] = None


def pick_default_csv() -> Path:
    cands = [Path(DEFAULT_CSV_WSL), Path(DEFAULT_CSV_WIN)]
    found = next((p for p in cands if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            "CSV not found. Pass --csv explicitly.\n"
            f"Checked:\n- {DEFAULT_CSV_WSL}\n- {DEFAULT_CSV_WIN}"
        )
    return found


def load_csv_rows(csv_path: Path) -> tuple[List[str], List[dict]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")
    return header, rows


def ollama_json_extract(
    *,
    user_text: str,
    model: str,
    ollama_url: str,
    available_cases: List[str],
    available_pipelines: List[str],
) -> dict:
    schema_hint = {
        "task": "dice_plot_one_case_from_csv",
        "case_id": "<string from available_cases>",
        "pipelines": ["<pipeline name from available_pipelines>"],
        "title": "<optional string or null>",
    }

    prompt = f"""
You are a strict argument extractor. Return ONLY valid JSON (no markdown, no extra text).
Do not invent names. Use only values from provided lists.

available_cases = {available_cases[:50]}
available_pipelines = {available_pipelines}

Return JSON matching:
{json.dumps(schema_hint, indent=2)}

User request:
{user_text}
""".strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(ollama_url, json=payload, timeout=120)
    r.raise_for_status()
    response_text = r.json().get("response", "").strip()
    return json.loads(response_text)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-case Dice plot from CSV using Ollama JSON extraction.")
    ap.add_argument("--csv", default=None, help="Path to Dice CSV")
    ap.add_argument("--request", default=None, help="Natural-language request")
    ap.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    ap.add_argument("--ollama_url", default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    ap.add_argument("--out", default="dice_one_case_from_csv.png", help="Output figure path")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else pick_default_csv()
    header, rows = load_csv_rows(csv_path)

    required = {"case", "best_dice", "best_pipeline"}
    if not required.issubset(set(header)):
        raise ValueError(f"CSV missing required columns: {required}. Found: {header}")

    pipeline_cols = [c for c in header if c not in ("dataset", "case", "best_pipeline", "best_dice")]
    if not pipeline_cols:
        raise ValueError("No pipeline columns detected in CSV.")

    available_cases = sorted({r["case"] for r in rows if r.get("case")})

    user_text = args.request or (
        "Plot dice for DeepMultiOrgSeg vs JLF vs MLP for case 001-0001_CT_1 and save figure."
    )

    parsed = ollama_json_extract(
        user_text=user_text,
        model=args.model,
        ollama_url=args.ollama_url,
        available_cases=available_cases,
        available_pipelines=pipeline_cols,
    )
    spec = DiceCsvArgs.model_validate(parsed)

    target = next((r for r in rows if r.get("case") == spec.case_id), None)
    if target is None:
        raise RuntimeError(f"Case not found in CSV: {spec.case_id}")

    chosen = [p for p in spec.pipelines if p in pipeline_cols]
    if not chosen:
        raise RuntimeError(f"No valid pipelines from request. Available: {pipeline_cols}")

    values = []
    for p in chosen:
        try:
            values.append(float(target[p]))
        except Exception:
            values.append(float("nan"))

    plt.figure(figsize=(8.0, 4.5))
    plt.bar(chosen, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Dice")
    plt.title(spec.title or f"Dice for case {spec.case_id}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=200)

    print("CSV:", csv_path)
    print("Case:", spec.case_id)
    print("Pipelines:", chosen)
    for p, v in zip(chosen, values):
        print(f"  {p:20s} Dice={v:.4f}")
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()
