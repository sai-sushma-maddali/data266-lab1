# Task 3 Results

Architecture + hyperparameter justification for CycleGAN Photo↔Monet (feature-matching EMA **v5** + DiffAug-EMA sweep). Metrics: `metrics_report.csv`. Visual failures: `failure_analysis.md`.

## What I built

Unpaired CycleGAN (**2×G + 2×D**) with adversarial + cycle + identity losses, plus **feature-matching** and **DiffAug + EMA**.

| Choice | Setting | Why |
|--------|---------|-----|
| Domains | Photo **7038** ↔ Monet **300** | Class Kaggle data |
| G / D | ResNet G (~**11.95M** each), PatchGAN D (~**2.76M**); total **~29.4M** | Slightly wider G than teammate; still CycleGAN-family |
| λ_cycle / λ_id / λ_FM | **7.5** / **1.0** / **10.0** | Lower identity than Omkar; FM loss to stabilize D features under scarce Monet |
| DiffAug | `translation,cutout` | Regularize D without heavy color aug on photos |
| EMA | **0.999** | Inference / validation selection |
| Batch / LR / epochs | **4** / **2e-4** / up to **100** | Longer schedule; select by EMA val cycle loss |
| Seed | **5775** | Independent run |

## Checkpoint selection & metrics

**Primary official (feature_matching_ema, Official Eval v5, N=300):**  
avg FID **97.499**, avg MiFID **0.4118** (P→M **95.884**, M→P **99.115**). Best cycle ckpt ~epoch **61** (EMA val cycle **0.07136**).

**Strongest FID sweep (diffaug_ema @ epoch 85):**  
avg FID **96.008**, avg MiFID **0.4067** — better FID than feature-matching official; checkpoint `checkpoints/generators_epoch_085.pt` with preds under `outputs/generators_epoch_085/` and samples in `outputs/pred_A2B/`, `outputs/pred_B2A/`.

Official v5 prints FID/MiFID thoroughly; KID/LPIPS/content-cosine at N=300 were **not** measured in that notebook (partial suite in N=30 in-training eval only).

## How design ties to results

Feature matching improved stability (NaN **0**, late peak mem ~**4.9 GB**) and delivered a competitive official FID. The pure DiffAug+EMA sweep shows that **checkpoint selection** can beat the feature-matching official average — important for Kaggle submission choice. Lower λ_id (1.0) vs Omkar’s 5.0 is a deliberate trade: more style freedom, slightly different color preservation behavior.

## Evidence

- Curves: `outputs/plots/training_curves.png`
- Example grids: `outputs/plots/translation_examples_*.png`
- Predictions: `outputs/pred_A2B/`, `outputs/pred_B2A/` (+ full 300 under `generators_epoch_085/`)
- Raw log: `reproducibility/raw_logs/run_log_task3_Sushma.txt`
