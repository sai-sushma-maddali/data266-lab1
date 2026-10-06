# Task 3 Failure Analysis — Sai Sushma Maddali

Visual quality / cycle-consistency failure write-up (lab §3.2).  
Evidence: `outputs/plots/translation_examples_A2B.png`, `translation_examples_B2A.png`, `outputs/pred_A2B/`, `outputs/pred_B2A/`, training curves in `outputs/plots/training_curves.png`.

## Failure 1 — Residual Monet texture on Photo outputs (A2B)

**Where:** `translation_examples_A2B.png` and `pred_A2B/*.png`.  
**Type:** Incomplete style transfer.  
**Observation:** Global scene layout is kept, but some outputs still look “painted” (soft edges, flattened detail). Aligns with Monet→Photo FID still near **98–99** even on the best sweeps.

## Failure 2 — Over-stylization / content softening (B2A)

**Where:** `translation_examples_B2A.png` and `pred_B2A/*.png`.  
**Type:** Content distortion / detail loss.  
**Observation:** Photo→Monet can smear fine textures (faces, signage, foliage) into Impressionist blobs. Style match improves while content preservation suffers — a known DiffAug + strong G tradeoff.

## Failure 3 — Checkpoint / metric mismatch risk

**Where:** Leaderboard-style selection across feature-matching official vs DiffAug-EMA @85.  
**Type:** Evaluation failure mode (process).  
**Observation:** Best avg FID (**96.008** @ ep85) is not the same checkpoint as the feature-matching official package (**97.499**). If submission and report checkpoint diverge, graders cannot trace metrics. Mitigation: pin one checkpoint ID in `metrics_report.csv` and Kaggle submission notes.

## Human audit status

Blinded 30-sample human audit scores / inter-rater agreement are **not present** in saved Official Evaluation v5 outputs (TBD for final PDF).

## Next fix to try

Run Omkar’s full N=300 secondary metric script on `generators_epoch_085.pt` so FID-leading checkpoints also report KID/LPIPS/content cosine before final submission.
