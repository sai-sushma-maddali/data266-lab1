# Task 1

Lab folder for the GPT-style character LM on TinyStories (reported run: **v8**).

Per DATA266 Lab1 instructions, this README covers setup/reproduction only. Metrics narrative is in `results.md`; failure cases are in `failure_analysis.md`.

## Folder layout (lab §5)

```
Sushma/
├── src/                 ← Task_1_GPT_Style_LLM_from_Scratch-v1..v8.ipynb
├── data_processed/      ← member-specific preprocessing (if used)
├── checkpoints/         ← gpt_best_model-v*.pth (report: v8)
├── outputs/
│   ├── plots/           ← loss / generation metric figures
│   └── samples/         ← generation snippets used in failure analysis
├── metrics_report.csv
├── failure_analysis.md
├── results.md
└── README.md
```

## Setup

1. Place TinyStories under `task1_llm/data/` (or set notebook paths).
2. CUDA PyTorch env; seed **5775** (SID4).

## Reproduce (smoke)

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  "task1_llm/Sushma/src/Task_1_GPT_Style_LLM_from_Scratch-v8.ipynb"
```

Smoke tip: lower `NUM_EPOCHS` / batch size in the config cell.

## Traceability

| Item | Path |
|------|------|
| Best checkpoint | `checkpoints/gpt_best_model-v8.pth` |
| Metrics table | `metrics_report.csv` |
| Manifest | `reproducibility/manifests/task1_Sushma_Maddali_manifest.json` |
| Raw log | `reproducibility/raw_logs/run_log_task1_Sushma.txt` |
| Hardware / seed | RTX 5090 · BF16 · seed **5775** |
