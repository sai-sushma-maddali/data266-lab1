# Task 1 — README (Omkar Rajale)

Lab folder for the GPT-style character LM on TinyStories (reported run: **v4**).

Per DATA266 Lab1 instructions, this README covers setup/reproduction only. Metrics narrative is in `results.md`; failure cases are in `failure_analysis.md`.

## Folder layout (lab §5)

```
Omkar/
├── src/                 ← Lab-1_task-1_GPT_from_scratch-v1..v4.ipynb
├── data_processed/      ← member-specific preprocessing (if used)
├── checkpoints/         ← gpt_best_model-v*.pt (report: v4)
├── outputs/
│   ├── plots/           ← loss / training metric figures
│   └── samples/         ← generation snippets used in failure analysis
├── metrics_report.csv   ← required metrics in one file
├── failure_analysis.md  ← three failure cases with snippets
├── results.md           ← architecture + hyperparameter justification
└── README.md            ← this file
```

## Setup

1. Place TinyStories under `task1_llm/data/` (or set the notebook data path).
2. Use a CUDA PyTorch environment (RTX 5090 / equivalent). See repo-root `README.md`.

## Reproduce (smoke)

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  task1_llm/Omkar/src/Lab-1_task-1_GPT_from_scratch-v4.ipynb
```

For a smoke test, reduce epochs/steps in the config cell first.

## Traceability

| Item | Path |
|------|------|
| Best checkpoint | `checkpoints/gpt_best_model-v4.pt` |
| Metrics table | `metrics_report.csv` |
| Manifest | `reproducibility/manifests/task1_Omkar_Rajale_manifest.json` |
| Raw log extract | `reproducibility/raw_logs/run_log_task1_Omkar.txt` |
| Hardware / seed | RTX 5090 · seed **298** |
