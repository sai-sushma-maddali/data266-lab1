# Task 2 Results

Architecture + hyperparameter justification for three from-scratch Yelp Polarity classifiers. Metrics: `metrics_report.csv`. Error review: `failure_analysis.md`.

## Shared data / embedding setup

| Choice | Setting | Why |
|--------|---------|-----|
| Split | **504K / 56K / 38K** train/val/test | Held-out test for fair model comparison within my suite |
| Vocab | **40,000** | Slightly richer than 30K for longer reviews |
| Embed / hidden | **128** / **128** | Keep models in the same capacity band (~5.2–5.4M params) |
| Max len | **256** | Truncation slice still hard; longer would raise mem/time |
| Dropout | **0.30** | Regularize recurrent models |
| Batch | train **256** / eval **512** | Balance speed and stability |
| Optimizer / LR | AdamW **1e-3** + ReduceLROnPlateau (min **1e-5**) | Plateau schedule when val loss stalls |
| Epochs / AMP | **6** / on | Enough to select best val epoch per model |
| Seed | **5775** | Independent of teammate |

## Model lineup (recurrent family — differs from Omkar’s CNN/Transformer set)

### Baseline — Unidirectional GRU
- **Why:** Efficient sequential baseline over learned embeddings; last hidden → logit.
- **Result:** Acc **0.9517**, ROC-AUC **0.9905**, **best ECE 0.0121**, train **271s**. Strong, well-calibrated baseline.

### Experimental 1 — Bidirectional LSTM (best accuracy)
- **Why:** Forward+backward context for polarity that depends on later contrast (“…but the service was terrible”).
- **Result:** Acc **0.9533** (best), MCC **0.9066**, strong on medium/long/negation/truncated slices. McNemar vs GRU not significant after Holm (p=0.062), so gain is small but consistent across slices.

### Experimental 2 — Attention LSTM
- **Why:** Additive attention over time steps should focus on opinion-bearing tokens.
- **Result:** Acc **0.9506** — did **not** beat BiLSTM; McNemar vs BiLSTM significant against Attn (p=0.002). Attention added cost (**467s**) without accuracy benefit under this schedule.

## How the metrics tie together

All three models sit near **95%** with tight bootstrap CIs. BiLSTM is the practical winner (accuracy + robustness slices), GRU is the calibration winner (lowest ECE), Attention LSTM is a negative ablation: more machinery ≠ better polarity under scratch embeddings. Truncated slice (~**0.93**) remains the hardest — matches truncation failures in the error review.

## Evidence

- Per-model metrics / CM CSVs under `outputs/{baseline_gru,bidirectional_lstm,attention_lstm}/`
- CM plots: `outputs/confusion_matrices/*_confusion_matrix.png`
- Comparison plots: `outputs/plots/`
- Error review samples: `outputs/samples/manual_error_review_20_completed.csv`
