# Task 3 — README (Omkar Rajale)

CycleGAN Photo ↔ Monet with **DiffAugment + generator EMA** (`cyclegan_r9_diffaug_ema`, reported **v7**).

Per DATA266 Lab1 instructions, this README covers setup/reproduction only. See `results.md` and `failure_analysis.md` for analysis.

## Folder layout (lab §5 / Task 3)

```
Omkar/
├── src/                 ← Task3_CycleGAN_v7.ipynb + Part3_Evaluation_Script_v7.ipynb
├── data_processed/
├── checkpoints/         ← best.pt (epoch 40 EMA)
├── outputs/
│   ├── plots/           ← training / FID / export example grids
│   ├── samples/         ← epoch progress grids
│   ├── pred_A2B/        ← Monet→Photo example grids
│   ├── pred_B2A/        ← Photo→Monet example grids
│   └── eval/            ← official metric tables / curves
├── metrics_report.csv
├── failure_analysis.md
├── results.md
└── README.md
```

## Setup

1. Place shared images in `task3_gan/data/monet_jpg/` and `task3_gan/data/photo_jpg/` (or set `DATA_ROOT`).
2. A100-class GPU recommended for full 60-epoch training.

## Reproduce (smoke)

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  task3_gan/Omkar/src/Task3_CycleGAN_v7.ipynb

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  task3_gan/Omkar/src/Part3_Evaluation_Script_v7.ipynb
```

Smoke tip: `epochs=1` + tiny image subset.

## Traceability

| Item | Path |
|------|------|
| Reported checkpoint | `checkpoints/best.pt` (epoch 40, EMA) |
| Metrics | `metrics_report.csv` |
| Manifest | `reproducibility/manifests/task3_Omkar_Rajale_manifest.json` |
| Raw log | `reproducibility/raw_logs/run_log_task3_Omkar.txt` |
| Seed / GPU | **266** · NVIDIA A100-SXM4-80GB |
