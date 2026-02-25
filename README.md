# SpleenArgViz

One-case Dice evaluation from natural language args extracted by Ollama.

## Setup

```bash
pip install matplotlib pydantic requests nibabel numpy pandas
```

```bash
ollama pull qwen2.5:7b-instruct
```

## Required files

Put these in `case_data/`:

- `case_id.txt` (one line, e.g. `spleen_41`)
- `gt.nii.gz`
- pipeline masks, e.g. `DeepSpleenSeg.nii.gz`, `JLF.nii.gz`, `TotalSegmentator.nii.gz`

## Run

```bash
python run_one_case.py
```

Outputs:

- `dice_one_case.png`
- console table with Dice per pipeline

## CSV Mode (recommended for your current data)

Use your existing Dice CSV from:

- `/mnt/c/Users/adams/Documents/Projects/Spleen/data/abnl-marro_all_cases_dice_4pipes_jlf_mlp.csv`

Run:

```bash
python run_one_case_from_csv.py --request "plot dice for DeepMultiOrgSeg vs JLF vs MLP for case 001-0001_CT_1 and save figure"
```

Outputs:

- `dice_one_case_from_csv.png`
- console Dice values for requested pipelines
