#!/usr/bin/env python3
import csv
import json
import os
import time
from pathlib import Path
from typing import List

import gradio as gr
import matplotlib.pyplot as plt
import requests


DEFAULT_CSV_WSL = "/mnt/c/Users/adams/Documents/Projects/Spleen/data/abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv"
DEFAULT_CSV_WIN = r"C:\Users\adams\Documents\Projects\Spleen\data\abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
OUT_DIR = Path("outputs")


def _s(x) -> str:
    return "" if x is None else str(x)


def normalize_path_for_os(p: str) -> Path:
    s = _s(p).strip()
    if not s:
        return Path("")
    # Handle accidental Windows-backslash form of WSL path: \mnt\c\...
    if os.name == "nt" and s.lower().startswith("\\mnt\\") and len(s) > 6:
        drive = s[5].upper()
        rest = s[6:]
        return Path(f"{drive}:{rest}")
    if os.name == "nt" and s.startswith("/mnt/") and len(s) > 6:
        drive = s[5].upper()
        rest = s[6:].replace("/", "\\")
        return Path(f"{drive}:{rest}")
    if os.name != "nt" and len(s) > 2 and s[1:3] == ":\\":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return Path(f"/mnt/{drive}{rest}")
    return Path(s)


def pick_default_csv() -> Path:
    cands = [Path(DEFAULT_CSV_WSL), Path(DEFAULT_CSV_WIN)]
    found = next((p for p in cands if p.exists()), None)
    if found is None:
        raise FileNotFoundError("Could not find default Dice CSV.")
    return found


def resolve_csv_path(csv_path_text: str) -> Path:
    raw = _s(csv_path_text).strip()
    if not raw:
        return pick_default_csv()

    cands = [normalize_path_for_os(raw), Path(raw)]
    # Also try a local relative fallback from SpleenArgViz -> ../Spleen/data/...
    cands.append(Path("../Spleen/data/abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv"))
    found = next((p for p in cands if p and p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"CSV not found: {raw}\nTry: C:\\Users\\adams\\Documents\\Projects\\Spleen\\data\\abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv"
        )
    return found


def load_csv_rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")
    return header, rows


def ollama_json_extract(user_text: str, model: str, ollama_url: str, available_cases: List[str], available_pipelines: List[str]):
    schema_hint = {
        "task": "dice_plot_one_case_from_csv",
        "case_id": "<string from available_cases>",
        "pipelines": ["<pipeline from available_pipelines>"],
        "title": "<optional string or null>",
    }
    prompt = f"""
You are a strict argument extractor. Return ONLY valid JSON.
Do not invent names. Use only values from provided lists.

available_cases = {available_cases[:80]}
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


def compute_and_plot(csv_path: Path, case_id: str, pipelines: List[str], title: str | None, order_mode: str):
    header, rows = load_csv_rows(csv_path)
    required = {"case", "best_dice", "best_pipeline"}
    if not required.issubset(set(header)):
        raise ValueError(f"CSV missing required columns: {required}")

    pipeline_cols = [c for c in header if c not in ("dataset", "case", "best_pipeline", "best_dice")]
    target = next((r for r in rows if r.get("case") == case_id), None)
    if target is None:
        raise RuntimeError(f"Case not found: {case_id}")

    chosen = [p for p in pipelines if p in pipeline_cols]
    # Keep first occurrence only, preserve request order.
    chosen = list(dict.fromkeys(chosen))
    if not chosen:
        raise RuntimeError(f"No valid pipelines selected. Available: {pipeline_cols}")

    pairs = []
    for p in chosen:
        v = float(target[p])
        pairs.append((p, v))

    if order_mode == "dice_desc":
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    chosen = [p for p, _ in pairs]
    scores = [v for _, v in pairs]
    table_rows = [[i + 1, p, round(v, 6)] for i, (p, v) in enumerate(pairs)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"dice_{case_id}_{int(time.time())}.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8.8, 4.8))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#17becf"]
    bar_colors = [colors[i % len(colors)] for i in range(len(chosen))]
    bars = plt.bar(chosen, scores, color=bar_colors, edgecolor="#222222", linewidth=0.8)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Dice")
    plt.title(title or f"Dice for case {case_id}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, scores):
        plt.text(b.get_x() + b.get_width() / 2.0, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    order_text = " -> ".join(chosen)
    return out_png, table_rows, order_text


def run_request(
    user_request: str,
    model: str,
    csv_path_text: str,
    ollama_url: str,
    manual_case: str,
    manual_pipelines: str,
    order_mode: str,
):
    user_request = _s(user_request).strip()
    model = _s(model).strip() or DEFAULT_MODEL
    ollama_url = _s(ollama_url).strip() or DEFAULT_OLLAMA_URL
    manual_case = _s(manual_case).strip()
    manual_pipelines = _s(manual_pipelines).strip()

    csv_path = resolve_csv_path(csv_path_text)
    header, rows = load_csv_rows(csv_path)
    pipeline_cols = [c for c in header if c not in ("dataset", "case", "best_pipeline", "best_dice")]
    available_cases = sorted({r["case"] for r in rows if r.get("case")})

    parsed = None
    if manual_case and manual_pipelines:
        parsed = {
            "task": "dice_plot_one_case_from_csv",
            "case_id": manual_case,
            "pipelines": [p.strip() for p in manual_pipelines.split(",") if p.strip()],
            "title": None,
        }
    else:
        parsed = ollama_json_extract(
            user_text=user_request,
            model=model or DEFAULT_MODEL,
            ollama_url=ollama_url or DEFAULT_OLLAMA_URL,
            available_cases=available_cases,
            available_pipelines=pipeline_cols,
        )

    case_id = _s(parsed.get("case_id", "")).strip()
    pipelines = parsed.get("pipelines", [])
    title = _s(parsed.get("title", "")).strip() or None
    if not case_id or not isinstance(pipelines, list):
        raise RuntimeError(f"Invalid parsed JSON: {parsed}")

    png_path, table_rows, order_text = compute_and_plot(csv_path, case_id, pipelines, title, order_mode=order_mode)
    return (
        json.dumps(parsed, indent=2),
        order_text,
        table_rows,
        str(png_path),
        f"Done. CSV={csv_path} | case={case_id}",
    )


def run_request_ui(user_request, model, csv_path_text, ollama_url, manual_case, manual_pipelines, order_mode):
    try:
        return run_request(user_request, model, csv_path_text, ollama_url, manual_case, manual_pipelines, order_mode)
    except Exception as e:
        return (
            "{}",
            "",
            [],
            None,
            f"Error: {type(e).__name__}: {e}",
        )


def build_ui():
    default_csv = str(pick_default_csv()) if Path(DEFAULT_CSV_WSL).exists() or Path(DEFAULT_CSV_WIN).exists() else (
        DEFAULT_CSV_WIN if os.name == "nt" else DEFAULT_CSV_WSL
    )
    with gr.Blocks(title="SpleenArgViz - Dice JSON Extractor") as demo:
        gr.Markdown("## SpleenArgViz\nSimple Dice app: request or manual args -> table + centered plot")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                user_request = gr.Textbox(
                    label="Request",
                    value="plot dice for DeepMultiOrgSeg vs JLF vs MLP for case 001-0001_CT_1 and save figure",
                    lines=3,
                )
                run_btn = gr.Button("Run", variant="primary")
                with gr.Accordion("Manual Override (fast mode)", open=False):
                    manual_case = gr.Textbox(label="Case ID")
                    manual_pipelines = gr.Textbox(label="Pipelines (comma-separated)")
                with gr.Accordion("Advanced Settings", open=False):
                    model = gr.Textbox(label="Ollama model", value=DEFAULT_MODEL)
                    csv_path = gr.Textbox(label="CSV path", value=default_csv)
                    ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
                    order_mode = gr.Radio(
                        choices=[("Request order", "request"), ("Sort by Dice (high to low)", "dice_desc")],
                        value="request",
                        label="Plot Order",
                    )
                status = gr.Textbox(label="Status", lines=2)

            with gr.Column(scale=2, min_width=700):
                parsed_json = gr.Code(label="Parsed JSON", language="json")
                plot_order = gr.Textbox(label="Plot Order Used", lines=1)
                image = gr.Image(label="Dice Plot", type="filepath", height=460)
                table = gr.Dataframe(headers=["rank", "pipeline", "dice"], datatype=["number", "str", "number"], label="Dice Results")

        run_btn.click(
            fn=run_request_ui,
            inputs=[user_request, model, csv_path, ollama_url, manual_case, manual_pipelines, order_mode],
            outputs=[parsed_json, plot_order, table, image, status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7861)
