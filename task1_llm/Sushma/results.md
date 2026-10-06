# Task 1 Results

Architecture + hyperparameter justification for the reported **v8** TinyStories character LM. Full numeric table: `metrics_report.csv`. Failures: `failure_analysis.md`.

## What I built

A **from-scratch decoder-only Transformer** (custom multi-head self-attention, LayerNorm, FFN, residuals, causal mask) for next-character prediction on TinyStories.

| Choice | Setting | Why |
|--------|---------|-----|
| Tokenization | Character-level, vocab **152** | Lab requirement; keep full character set from my split without aggressive cleaning |
| Sequence length / stride | **256** / **128** | Overlapping windows increase sample count (**700,993** train seqs) while fitting larger batches |
| Depth / width | **8** blocks, **8** heads (dim 64), hidden **512**, FFN **2048**, dropout **0.1** | ~**25.5M** params — comparable capacity to teammate but different data/window recipe |
| Optimizer | **Adam** fused, LR **1e-4** (no WD printed) | Conservative LR for longer **15**-epoch schedule; fused Adam for GPU throughput |
| Schedule | Linear warmup **1000** steps + cosine decay | Smooth early training; avoids CE spikes |
| Batch | **64** | Larger batch than accum-based recipe; trades context length for throughput |
| Precision | BF16 | Stable mixed precision on RTX 5090 |
| Epochs | **15** | Beyond minimum 10 to push val CE lower; best @ epoch **15** |
| Seed | **5775** (SID4) | Own split / init independent of teammate |

## Training outcome (ties metrics to design)

- Best checkpoint `checkpoints/gpt_best_model-v8.pth`: val CE **0.5242**, PPL **1.689**, BPC **0.756**, val top-1 **83.19%**.
- Train CE **0.4850** → gap **0.0392** (wider than Omkar’s). Longer training + no WD likely contributed to a larger train/val gap.
- Throughput **~287k** tokens/sec benefits from batch 64 + shorter context; peak mem **~12 GB** is higher than the accum-16 recipe.
- Generation diversity is the weak spot: Distinct-1 **0.0115**, repeated-4gram **0.5865** — matches the bow-loop failure mode.
- Avg/max grad norms **0.34 / 4.60**, NaN/Inf **0** — training remained stable.

## Decoding

Temperature sampling at **T=0.4** for diversity metrics (20 samples × 4000 tokens). Lower T (0.2) improves local grammar but reduces diversity and can insert abrupt characters.

## Evidence

- Loss curves: `outputs/plots/loss_curves_v8.png`
- Generation metrics plot: `outputs/plots/generation_metrics_v8.png`
- Failure snippets: `outputs/samples/generation_failures_v8.txt`
- Raw log: `reproducibility/raw_logs/run_log_task1_Sushma.txt`
- Manifest: `reproducibility/manifests/task1_Sushma_Maddali_manifest.json`
