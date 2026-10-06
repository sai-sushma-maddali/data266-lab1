# Task 1 Failure Analysis — Sai Sushma Maddali

Required: three generation failure cases with snippet, failure type, and observation (lab §1.4).  
Source notebook: `src/Task_1_GPT_Style_LLM_from_Scratch-v8.ipynb`. Text archive: `outputs/samples/generation_failures_v8.txt`.

## Case 1 — Repetition loop (bow)

**Snippet**  
`She opened the box and saw a big bow. It was a shiny red bow.` …  
`She put the bow in the bow. She said, "This is a special bow. It is a special bow. It is a special bow. It is a special bow."`

**Failure type:** Repetition  

**Observation:** High repeated-4gram rate (**0.5865**) matches this loop. Once “bow/special” n-grams dominate, temperature sampling keeps reinforcing them.

## Case 2 — Abrupt character insertion (low temperature)

**Snippet** (T=0.2)  
Nora finds a tiny door behind a bookshelf, then suddenly a crying girl appears with weak causal link.

**Failure type:** Loss of coherence / abrupt character insertion  

**Observation:** Low temperature reduces diversity (Distinct-1 already **0.0115**) but does not enforce plot logic; the model jumps to another high-probability story fragment.

## Case 3 — Truncation / topic shift

**Snippet**  
Attic/bear bakery sample ends mid-sentence (`Then they both started t`) after an unmotivated jump from fear to pastry.

**Failure type:** Truncation / topic shift  

**Observation:** Fixed-length generation can cut mid-word; topic drift reflects weak long-range planning under character tokenization and context 256.

## Takeaway

Longer training improved CE but hurt diversity relative to Omkar’s run. Prioritize generation-aware regularization and evaluate Distinct-n / 4-gram rate during training, not only at the end.
