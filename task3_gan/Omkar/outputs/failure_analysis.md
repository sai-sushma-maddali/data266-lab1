# Task 3 Failure Analysis — Omkar Rajale

Visual quality / cycle-consistency failure write-up (lab §3.2).  
Evidence grids: `outputs/plots/export_examples.png`, `outputs/samples/epoch_040.png`, `outputs/pred_A2B/`, `outputs/pred_B2A/`.

## Failure 1 — Incomplete photo realism (Monet→Photo)

**Where:** A2B examples in `export_examples.png` / epoch-40 grids.  
**Type:** Incomplete style transfer / residual painterly texture.  
**Observation:** Structure (bridge, cliffs, water) is preserved (good cycle/content cosine ~0.79), but outputs retain brush-stroke noise instead of camera texture. Matches worse A2B FID (**101.1**) vs B2A (**96.5**).

## Failure 2 — Color / tone shift (Photo→Monet)

**Where:** B2A translations on sunsets / foliage.  
**Type:** Color bleed / palette drift.  
**Observation:** Monet stylization sometimes oversaturates or shifts sky/ground hues. Identity loss (λ=5) reduces but does not eliminate this; DiffAug color aug on Monet D may also encourage broader palette variance.

## Failure 3 — Content distortion on fine structures

**Where:** Thin objects (boat masts, distant buildings) in progress samples.  
**Type:** Content distortion / artifact.  
**Observation:** PatchGAN + 256² ResNet can warp fine geometry while keeping global layout. Cycle L1 stays low (~0.04) even when perceptual artifacts remain — L1 alone under-penalizes these errors (LPIPS captures more).

## Human audit status

Audit image packs were generated under the eval `human_audit` workflow; **numeric style/content/artifact scores and Cohen’s κ are not yet filed** in saved outputs (marked TBD in the team report).

## Next fix to try

Prefer EMA checkpoints with best **perceptual** secondary metrics (LPIPS + content cosine), not FID alone; add a light Laplacian / edge consistency term for thin structures.
