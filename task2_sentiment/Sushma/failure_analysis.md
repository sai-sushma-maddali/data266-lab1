# Task 2 Failure / Error Analysis — Sai Sushma Maddali

Lab §2.2: manually review **20** errors — 5 confident FP, 5 confident FN, 5 near-threshold, 5 slice-specific — with error type + testable fix.

**Reviewed model suite focus:** attention / comparison review + completed sheet on model errors (threshold **0.5**).  
**Files:**  
- `outputs/samples/manual_error_review_20_completed.csv`  
- `outputs/samples/qualitative_error_examples.csv`

## Error-type summary

| Type | What we saw | Testable fix |
|------|-------------|--------------|
| Mixed aspect sentiment | Great food / bad service (or reverse) | Aspect-aware sentence pooling; measure mixed-aspect slice |
| Possible annotation noise | Clearly positive text labeled 0 (e.g. “wow love place…”) | Source-label audit before retrain |
| Temporal / update conflict | “used to be favorite… now terrible” | Recency-weighted pooling / update markers |
| Truncation slice | Decisive sentiment after token 256 | Head-tail truncation (128+128) or max_len 384; re-measure truncated macro-F1 + McNemar vs BiLSTM |
| Near-threshold ambiguity | p≈0.50 with pros+cons | Sentence-level attention / calibrator on [0.45,0.55] |
| Sarcasm / humor narrative | Surface insults, positive label | Sarcasm slice + char n-gram / CNN-GRU probe |

## Slice link

Truncated accuracy is the weakest slice (BiLSTM **0.936** vs ~0.95 elsewhere) — aligns with truncation_review_failure rows in the completed CSV.

## Proposed next experiment

Increase effective context via **head-tail truncation** (keep first + last 128 tokens) on the BiLSTM checkpoint recipe and re-run: truncated-slice macro-F1, overall Acc CI, and McNemar vs current BiLSTM.
