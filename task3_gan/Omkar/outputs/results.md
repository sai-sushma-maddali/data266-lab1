# Task 3 Results — Omkar Rajale

Architecture + hyperparameter justification for CycleGAN Photo↔Monet (`cyclegan_r9_diffaug_ema`, **v7**). Metrics: `metrics_report.csv`. Visual failures: `failure_analysis.md`.

## What I built

Unpaired image-to-image translation with **two ResNet-9 generators** and **two PatchGAN discriminators**, trained with adversarial + cycle + identity losses.

| Choice | Setting | Why |
|--------|---------|-----|
| Domains | A=Monet (300), B=Photo (7038) | Kaggle class competition setup |
| Resolution | **256** (load **286** + random crop) | Paper-style jitter; native competition size |
| G / D | ResNet-9 (ngf/ndf **64**), PatchGAN **3**-layer | Classic CycleGAN capacity; total **~28.3M** params |
| λ_cycle / λ_id | **10.0** / **5.0** | Strong cycle for content; identity preserves color when domains share palette |
| DiffAug | Monet: `color,translation,cutout`; Photo: `translation` | Only 300 Monet images → D_A overfits without DiffAug |
| EMA | decay **0.999** on generators | Smoother inference weights; select by held-out FID |
| Batch / LR / epochs | **4** / **2e-4** / **60** (decay after 30) | Stable GAN defaults (Adam β1=0.5) |
| AMP | BF16 | Fit A100 training at ~**19 GB** peak |
| Seed | **266** | Reproducible |

## Checkpoint selection & metrics

- Selected `checkpoints/best.pt` at epoch **40** by **held-out FID average (91.871)** during training — not last epoch.
- Official eval N=**300** (Eval Script v7): avg FID **98.828**, avg MiFID **0.4134**.
- Photo→Monet FID **96.5** better than Monet→Photo **101.1** (photo realism is harder).
- Secondary suite completed: KID, precision/recall, density/coverage, cycle L1, LPIPS, content cosine (see `metrics_report.csv`).
- Training stable: nonfinite losses **0**; logged G/D grad norms.

## How design ties to results

DiffAug + EMA were the main stabilizers for the tiny Monet set. Identity λ=5 reduced color shifts (helps MiFID/content cosine). Remaining high FID (~99 avg) is expected under small-N Inception FID bias on this dataset (see notebook discussion).

## Evidence

- Curves: `outputs/plots/training_curves.png`, `outputs/plots/loss_and_fid_curves.png`
- Translations: `outputs/plots/export_examples.png`, `outputs/samples/epoch_040.png`
- Pred folders: `outputs/pred_A2B/`, `outputs/pred_B2A/`
- Eval table: `outputs/eval/checkpoint_comparison.csv`
