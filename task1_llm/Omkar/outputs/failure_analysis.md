# Task 1 Failure Analysis — Omkar Rajale

Required: three generation failure cases with snippet, failure type, and observation (lab §1.4).  
Source notebook: `src/Lab-1_task-1_GPT_from_scratch-v4.ipynb`. Text archive: `outputs/samples/generation_failures_v4.txt`.

## Case 1 — Abrupt second-story restart

**Snippet**  
Prompt: `The little dog was afraid`  
Continuation ends with helping a kind man collect flowers, then restarts:  
`Once upon a time, there was a little girl named Lily...`

**Failure type:** Loss of coherence / abrupt story boundary  

**Observation:** The model finishes a local plot beat then samples a TinyStories-style opener, ignoring the ongoing narrative. Character LMs with finite context still treat “Once upon a time” as a high-probability reset. Decoding: T=0.8, top-k=40.

## Case 2 — Incoherent object behavior / broken grammar

**Snippet**  
`Lily wanted to sit on it. She ran to the barrel and pulled and tugged until it broke. The barrel was shut and smashed.` … trailing fragment `Ben he`

**Failure type:** Broken grammar / inconsistent object state  

**Observation:** The barrel is simultaneously “broken” and “shut”; generation truncates mid-token. Entity attributes are not constrained across sentences.

## Case 3 — Illogical agency (quality decoding)

**Snippet**  
`In a small forest, there was a big tree... One day, the tree was hungry.`

**Failure type:** Unsupported / illogical details (hallucinated agency)  

**Observation:** Quality decoding (T=0.7, top-p=0.9, no retrain) improves fluency but still assigns animate goals to inanimate objects. Sampling changes presentation more than world-consistency.

## Takeaway

Next-char CE/top-1 look strong, but Distinct-n / repeated-4gram and these cases show remaining story-level failures. Next experiments: repetition penalty / unlikelihood on n-grams, or slightly larger subword units while keeping a from-scratch attention stack.
