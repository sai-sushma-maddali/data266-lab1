# Task 3 

CycleGAN Photo ↔ Monet with **DiffAugment + EMA + feature matching** (primary official eval) and a DiffAug-EMA checkpoint sweep.

Per DATA266 Lab1 instructions, this README covers setup/reproduction only. See `results.md` and `failure_analysis.md` for analysis.

## Folder layout (lab §5 / Task 3)

```
Sushma/
├── src/                 ← Task - 3 GAN - v5.ipynb + Official_Evaluation v5
├── data_processed/
├── checkpoints/         ← generators_epoch_085.pt (strongest FID sweep)
├── outputs/
│   ├── plots/           ← training curves + translation example grids
│   ├── samples/         ← translation example copies
│   ├── pred_A2B/        ← sample Monet→Photo preds (+ full set under generators_epoch_085/)
│   ├── pred_B2A/        ← sample Photo→Monet preds
│   └── generators_epoch_085/  ← full 300-image dirs + submission.csv
├── metrics_report.csv
├── failure_analysis.md
├── results.md
└── README.md
```

## Setup

1. Place Monet/Photo JPGs under `task3_gan/data/` (or update notebook paths).
2. Seed **5775**.

## Reproduce (smoke)

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  "task3_gan/Sushma/src/Task - 3 GAN - v5.ipynb"

jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  "task3_gan/Sushma/src/Task3_CycleGAN_Official_Evaluation - v5.ipynb"
```

Smoke tip: 1 epoch + small image subset.

## Traceability

| Run | Checkpoint / outputs | Role |
|-----|----------------------|------|
| feature_matching_ema | best cycle ckpt (~ep 60/61) | Primary official FID/MiFID |
| diffaug_ema @ 85 | `checkpoints/generators_epoch_085.pt` | Strongest listed avg FID |

Manifest: `reproducibility/manifests/task3_Sushma_Maddali_manifest.json`  
Raw log: `reproducibility/raw_logs/run_log_task3_Sushma.txt`
