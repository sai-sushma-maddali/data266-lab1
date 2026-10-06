# Task 1 Results — Omkar Rajale

Architecture + hyperparameter justification for the reported **v4** TinyStories character LM. Full numeric table: `metrics_report.csv`. Failures: `failure_analysis.md`.

## What I built

A **from-scratch decoder-only Transformer** (no `nn.Transformer` / prebuilt attention modules) trained as a next-character language model on TinyStories.

| Choice | Setting | Why |
|--------|---------|-----|
| Tokenization | Character-level, vocab **114** (train-only chars after light cleaning) | Lab requires char LM; cleaning shrinks vocab and removes rare junk chars that inflate CE |
| Context (`block_size`) | **512** | Longer context than a 256 default to keep story continuity inside one window |
| Depth / width | **8** layers, **8** heads, emb **512**, FFN **2048**, dropout **0.1** | ~**25.6M** params — large enough to fit TinyStories style, small enough for a single-GPU lab run |
| Causal mask | Triangular attention mask in custom MHA | Required for autoregressive next-token training |
| Positional encoding | Learnable absolute positions | Simple, matches common GPT-style teaching stacks |
| Optimizer | **AdamW**, max LR **3e-4**, WD **0.1** | Weight decay helped generalization vs plain Adam in early versions |
| Schedule | Linear warmup **5%** of steps + cosine to **3e-5** | Stabilizes early updates; cosine avoids late LR noise |
| Batch | Physical **16** × grad accum **2** → effective **32** | Fits RTX 5090 with BF16 at context 512 |
| Precision | BF16 | Throughput without FP16 loss-scale hassle on Ampere+ |
| Epochs | **10** (min lab requirement) | Val CE still improving at epoch 10; selected best = epoch 10 |
| Seed | **298** (SID4) | Reproducible split / init |

## Training outcome (ties metrics to design)

- Best checkpoint `checkpoints/gpt_best_model-v4.pt`: val CE **0.4819**, PPL **1.619**, BPC **0.695**, top-1 **84.51%**.
- Generalization gap only **0.0079** — AdamW + WD + dropout kept train/val close despite 25M params.
- Distinct-1 **0.018** and repeated-4gram **0.457** show that strong next-char accuracy ≠ diverse storytelling; character models still loop local patterns.
- Peak mem **3.6 GB** and ~**251k** tokens/sec: context-512 is affordable with accum=2 on RTX 5090.
- NaN count **0**; late grad norms ~0.2–0.5 after early smoke spikes — schedule + clip (max norm 1.0) worked.

## Decoding

- Standard metrics: temperature **0.8**, top-k **40**.
- Separate quality pass (not re-trained): T=**0.7**, top-p=**0.9**, stop on double newline — more readable but still fails entity/agency checks (see failure analysis).

## Evidence

- Loss curves: `outputs/plots/training_metrics_v4.png`
- Sample failure text: `outputs/samples/generation_failures_v4.txt`
- Manifest: `reproducibility/manifests/task1_Omkar_Rajale_manifest.json`
