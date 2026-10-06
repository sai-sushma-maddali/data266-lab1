# Task 2

Yelp Polarity sentiment bake-off with from-scratch embeddings: **GRU → BiLSTM → Attention LSTM**.

Per DATA266 Lab1 instructions, this README covers setup/reproduction only. See `results.md` and `failure_analysis.md` for analysis.

## Folder layout (lab §5)

```
Sushma/
├── src/                 ← EDA + training notebooks
├── data_processed/      ← preprocessing notes / Drive link
├── checkpoints/         ← baseline_gru/, bidirectional_lstm/, attention_lstm/
├── outputs/
│   ├── plots/           ← CI, slice, EDA figures
│   ├── confusion_matrices/  ← CSV + PNG per model
│   ├── predictions/     ← pointers to test_predictions.parquet
│   ├── samples/         ← manual error-review CSVs
│   ├── baseline_gru/ | bidirectional_lstm/ | attention_lstm/
│   ├── eda/
│   └── model_comparison/
├── metrics_report.csv
├── failure_analysis.md
├── results.md
└── README.md
```

## Setup

1. Point notebook data cells at Yelp Polarity (see `data_processed/README.md`).
2. Seed **5775**. Recommended order: EDA → training notebook.

## Reproduce (smoke)

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  task2_sentiment/Sushma/src/Yelp_Sentiment_Classification_Model_Training.ipynb
```

Smoke tip: `NUM_EPOCHS = 1` and a small data fraction.

## Traceability

| Model | Checkpoint dir |
|-------|----------------|
| GRU baseline | `checkpoints/baseline_gru/` |
| BiLSTM (best accuracy) | `checkpoints/bidirectional_lstm/` |
| Attention LSTM | `checkpoints/attention_lstm/` |

Manifest: `reproducibility/manifests/task2_Sushma_Maddali_manifest.json`  
Raw log: `reproducibility/raw_logs/run_log_task2_Sushma.txt`
