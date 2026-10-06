# data266-lab1# DATA266 Lab 1 — Team 31

**Members:** Omkar Rajale · Sai Sushma Maddali  
**Course:** DATA266 · Fall 2026  
**Repo:** https://github.com/sai-sushma-maddali/data266-lab1  

LLM pretraining (TinyStories GPT from scratch) · Yelp polarity sentiment · CycleGAN Photo↔Monet.

Each member independently designed, trained, and evaluated their own models for all three tasks. This repository holds code, checkpoints, outputs, raw logs, manifests, and the combined team report.

---

## Repository layout

```
data266-lab1/
├── README.md                          ← this file
├── task1_llm/
│   ├── data/                          ← shared TinyStories (not committed if large)
│   ├── Omkar/                         ← member folder (see README.md inside)
│   └── Sushma/
├── task2_sentiment/
│   ├── data/                          ← shared Yelp polarity (or HF download)
│   ├── Omkar/
│   └── Sushma/
├── task3_gan/
│   ├── data/                          ← shared monet_jpg/ + photo_jpg/
│   ├── Omkar/
│   └── Sushma/
├── reproducibility/
│   ├── manifests/                     ← env + checkpoint↔metric mapping (JSON)
│   └── raw_logs/                      ← unedited training logs
└── report/
    ├── DATA266_Lab1_Report_Team_31.docx
    └── LAB1_EXTRACTED_RESULTS.md
```

Each member folder follows lab §5:

```
member_name/
├── src/                 ← notebooks with outputs
├── data_processed/      ← member-specific preprocessing (never shared across members)
├── checkpoints/         ← trained weights
├── outputs/             ← plots/, samples/, predictions/, confusion_matrices/, pred_A2B|B2A/ as applicable
├── metrics_report.csv   ← every required metric for this task
├── failure_analysis.md  ← Task 1 failures / Task 2 error review / Task 3 visual failures
├── results.md           ← architecture + hyperparameter justification
└── README.md            ← setup + reproduce only (metrics live in results.md)
```

---

## Quick start / smoke test (one command)

From the repo root, after installing dependencies and placing datasets (see below), reproduce **Sushma Task 2** as a documented smoke path (smaller than full CycleGAN):

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  task2_sentiment/Sushma/src/Yelp_Sentiment_Classification_Model_Training.ipynb
```

For an even faster smoke test, open that notebook and set `NUM_EPOCHS = 1` (and optionally a data fraction) in the config cell, then re-run the command.

Other member/task smoke commands are listed in each folder’s `README.md`.

---

## Environment

Typical stack used in GPU lab notebooks:

- Python 3.11+
- PyTorch 2.x with CUDA
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `pillow`
- Task 3 extras: `torchvision`, `lpips` (Omkar full metric suite)

Create a local env (example):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn matplotlib tqdm pillow jupyter lpips
```

No personal absolute paths or secrets should be committed. Prefer notebook config cells / relative `DATA_ROOT` variables.

---

## Datasets

| Task | Dataset | Where to put it |
|------|---------|-----------------|
| 1 | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | `task1_llm/data/` |
| 2 | [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity) | HF download in-notebook, or `task2_sentiment/data/` |
| 3 | Kaggle Monet CycleGAN photos/paintings | `task3_gan/data/monet_jpg/`, `task3_gan/data/photo_jpg/` |

Large raw images/checkpoints may be Git LFS or external Drive links (see member `data_processed/README.md` where noted).

---

## Where results live

| Artifact | Location |
|----------|----------|
| Combined team report | `report/DATA266_Lab1_Report_Team_31.docx` |
| Extracted metric dump | `report/LAB1_EXTRACTED_RESULTS.md` |
| Per-member write-ups | `task*/{Omkar,Sushma}/results.md` |
| Manifests (versions + checkpoint map) | `reproducibility/manifests/*_manifest.json` |
| Raw training logs | `reproducibility/raw_logs/` |
| Figures used in the report | Under each member’s `outputs/` (see captions in the DOCX) |

### Reported official runs (high level)

| Member | Task 1 | Task 2 | Task 3 |
|--------|--------|--------|--------|
| Omkar | GPT char-LM v4 (`gpt_best_model-v4.pt`) | FastText → TextCNN → Transformer | CycleGAN DiffAug+EMA `best.pt` @ ep40 |
| Sushma | GPT char-LM v8 (`gpt_best_model-v8.pth`) | GRU → BiLSTM → Attn-LSTM | Feature-matching EMA official + DiffAug-EMA @85 sweep |

---

## Citations

1. Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762  
2. Eldan & Li (2023). TinyStories. https://arxiv.org/abs/2305.07759  
3. Zhu et al. (2017). CycleGAN. https://arxiv.org/abs/1703.10593  
