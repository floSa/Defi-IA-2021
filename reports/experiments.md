# Experiment log

Offline metrics on a stratified 15 % hold-out (seed 42), unless noted.
Macro-F1 higher is better; disparate impact (DI) lower → 1.0 is better.
Ground-truth labels' own DI = 3.898.

| # | Date | Model | Preproc | Macro-F1 | DI | Notes |
|---|------|-------|---------|---------:|---:|-------|
| A | 2026-07-13 | word TF-IDF (1–2) + LogReg, balanced | lower | 0.7335 | 4.10 | hardened baseline, floor |
| B | 2026-07-13 | word(1–2)+char_wb(2–5) TF-IDF + LinearSVC, balanced | lower | **0.7641** | 3.85 | strong classical, ~9 min, 6.5 GB RAM peak |
| F1 | 2026-07-13 | = B + **gender scrubbing** | lower + scrub | 0.7601 | **3.49** | fairness track: −0.36 DI for −0.4 F1 pt |
| — | 2026-07-13 | word(1–2) + SGD(hinge), balanced | lower | 0.5230 | 9.06 | SGD untuned = poor; abandoned, use LinearSVC |
| B' | 2026-07-13 | word(1–2)+**hashed** char_wb(2–5) + LinearSVC | lower | 0.7639 | 3.86 | HashingVectorizer = same score as B, bounded RAM → full-data submission now possible locally |
| C | 2026-07-13 | **roberta-base** fine-tuned (fp32, LR 1e-5, class-weighted, 2 ep) | strip+lower none | **0.8035** | 4.15 | Kaggle T4; loss 6.6->1.0 clean, still rising at ep2; beats classical +4pts, near 2021 top |
| C-thr | 2026-07-14 | classical wordchar + **per-class threshold tuning** | lower | ~~0.7714~~ | — | ⚠️ in-sample, see audit below; expect ≈0.767 |
| B'' | 2026-07-18 | = B' re-run on the rebuilt desktop env | lower | **0.7643** | 3.89 | **reproduction check**: matches the logged 0.7641 to 0.0002 under pandas 3.0.3 / sklearn 1.9.0 / numpy 2.1.3 — the stack upgrade did not move the baseline |

## ⚠️ Local compute constraint (7.4 GB WSL, no GPU)

Full-data fits with **char n-grams OOM-crash the WSL VM** (vocabulary
construction spikes >7.4 GB). Confirmed repeatedly on 2026-07-13. Consequences:

- Offline **holdout** numbers above are valid (holdout fits under the ceiling).
- **Submissions** generated locally are limited to the lighter **word-only**
  config. The strong word+char model (B, 0.7641) and the transformer are
  deferred to the **64 GB desktop** (tomorrow) or **Kaggle** (T4).
- **Fixed (2026-07-13):** the char channel now uses a `HashingVectorizer`
  (fixed `n_features=2**20`), so the full-data strong model fits locally
  without OOM. Score is unchanged (0.7639 vs 0.7641).

## Reading

- Step B lifts Macro-F1 by +3 pts over the baseline and, unlike the baseline,
  does **not** amplify the labels' bias (DI 3.85 < 3.898).
- Reference: 2021 public leaderboard top ≈ 0.81–0.82 Macro-F1. The gap to close
  is the transformer (Step C) + tuning + ensemble.

## ⚠️ Finding: the Macro-F1 bottleneck is NOT the rare classes (2026-07-18)

The whole augmentation strategy rests on a premise stated in
`preprocessing/augment.py` and ROADMAP §3 — *"rare classes are the ones dragging
macro-F1 down"*. On the classical model, measured on the 32.6k holdout
(`reports/error_analysis.md`), that premise is **false**.

| | mean F1 | median support |
|---|---:|---:|
| the 7 **rarest** classes | **0.738** | ~130 |
| the 7 **weakest** classes | **0.659** | 225 |

Spearman correlation between class support and class F1 is only **+0.37**.
Rarity explains little of the spread — `class_weight="balanced"` is already doing
its job. `rapper` (117 examples) scores 0.802, better than `teacher` (1372
examples) at 0.587 and `architect` (876) at 0.673.

What actually costs Macro-F1 is **semantic confusion between neighbouring
professions**, dominated by the `professor` mega-class (32 % of the data)
bleeding into every academically-adjacent job and back:

| true → predicted | count |
|---|---:|
| professor → teacher | 194 |
| professor → psychologist | 180 |
| professor → physician | 165 |
| psychologist → professor | 150 |
| physician → professor | 136 |
| architect ↔ software_engineer | 95 / 80 |
| nurse → physician | 92 |
| physician → surgeon | 78 |

**Consequences:**

1. This is the likely explanation for the augmentation result (−0.0028):
   `augment_rare_classes` synthesises examples for classes *below a count
   threshold*, i.e. precisely the classes that already score best. It adds
   paraphrase noise where there was no deficit, and touches none of the
   confusions above.
2. Augmentation should be **retargeted at confusable pairs**, not at rare
   classes — or dropped in favour of work that separates `professor` from
   `teacher`/`psychologist`/`physician`.
3. It also predicts **where the transformer's +4 pts come from**: telling
   "professor" from "teacher" needs context, which is exactly what a contextual
   encoder does and bag-of-words cannot. Expect the transformer's gain to be
   concentrated in these confusable mid-frequency classes — worth verifying
   per-class once a GPU run lands, because if it is *not* concentrated there,
   the ensemble has more to gain than a single blend weight suggests.

## ⚠️ REAL KAGGLE SCORE: 0.82166 — the holdout overestimated by 0.011 (2026-07-19)

`final_accuracy_track.csv` submitted. **Public score 0.82166** (second column
shown: 0.82403), against a holdout estimate of **0.8329**.

| | Macro-F1 | gap to us |
|---|---:|---:|
| 1st — France UR2, WeTried | 0.84247 | +0.0208 |
| 2nd — France UJM, BravoNils | 0.82932 | +0.0077 |
| 3rd — Cameroun ENSPY, Fred | 0.82733 | +0.0057 |
| **us** | **0.82166** | — |

**The offline estimate was optimistic by +0.0112 — twice the distance to the
podium.** This is the single most consequential measurement of the whole
session: every decision taken on holdout numbers alone is now suspect, including
the choice to ship the thresholded pipeline over the ensemble.

Three candidate causes, most likely first:

1. **The per-class biases do not fully transfer.** Fitted on 32,580 validation
   rows, applied to 54,300 test rows from a neighbouring but not identical
   distribution. The measured +0.019 is probably smaller in reality.
2. **Selection optimism on the final pipeline choice** — it won a 4-way
   comparison judged on the same rows. Flagged at the time, never quantified.
3. **Single split, single seed** behind every transformer figure.

**These are separable, and cheaply.** Submitting `roberta_large_6ep_ensemble.csv`
(holdout 0.8288) and a plain-argmax version isolates cause 1 from causes 2–3.
Until that is done, further optimisation is blind. Top teams used 9, 44 and 14
entries — a submission is a real measurement and is worth more than any offline
estimate.

Full analysis and the ranked plan: [`PRECONISATIONS.md`](PRECONISATIONS.md).

## FINAL: roberta-large converged at 0.8241, submission at 0.8329 (2026-07-19)

Given its full 6 epochs, roberta-large is the **only run on this project that
actually converged** — its best epoch is epoch 5, not its last, so early stopping
restored it and `check_convergence.py` does not flag it.

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Macro-F1 | 0.7710 | 0.8023 | 0.8060 | 0.8059 | **0.8241** | 0.8225 |

The flat spot at epochs 3–4 was **not** a plateau — it was mid-schedule, with the
learning rate still high. Its decay at epoch 5 unlocked another 1.8 points. This
is the third time on this project that reading a single epoch without accounting
for the LR schedule produced a wrong call, and it was called wrong here too
before epoch 5 landed.

### Final ranking (⚠ = still improving when it stopped)

| model | Macro-F1 | DI | |
|---|---:|---:|---|
| **roberta-large, 6 epochs** | **0.8241** | 4.150 | converged |
| roberta-base, 6 epochs | 0.8027 | 4.108 | ⚠ |
| roberta-base counterfactual | 0.8018 | **3.407** | ⚠ |
| roberta-base, 3 epochs | 0.7978 | 4.134 | ⚠ |
| classical word+char SVM | 0.7643 | 3.891 | — |

roberta-large beats roberta-base by **+0.021** — and the comparison now favours
roberta-base if anything, since base was still climbing while large had stopped.

### Accuracy-track submission

| pipeline | Macro-F1 | DI |
|---|---:|---:|
| roberta-large argmax | 0.8236 | 4.439 |
| **+ per-class thresholds** | **0.8329** | 5.142 |
| + classical ensemble | 0.8288 | 4.434 |
| + ensemble + thresholds | 0.8295 | 4.771 |

Composing both levers again lost to thresholds alone — same as on the
epoch-3 model, and for the same reason (α plus 28 biases on one 16k split).
`submissions/final_accuracy_track.csv` ships the thresholds pipeline.

Against the 2021 public leaderboard top of ≈0.81–0.82, and the project's own
starting points of 0.8035 (Kaggle reference) and 0.7643 (classical).

**Caveats that remain:** single seed for every transformer number; the 0.8329
won a 4-way choice on the judging half, so it carries a little selection
optimism; and DI 5.14 is the worst of any submission here — the accuracy track
buys its Macro-F1 squarely at the expense of the fairness tie-break.

## Counterfactual training is ~free on the transformer (2026-07-19)

The fairness track was capped at classical accuracy because counterfactual
training had only ever been tried on the linear model. Run on roberta-base
(3 epochs over the doubled set = the same gradient budget as the 6-epoch
non-counterfactual run, so the two are comparable):

| model | Macro-F1 | DI |
|---|---:|---:|
| roberta-base, no mitigation | 0.8027 | 4.108 |
| **roberta-base, counterfactual** | **0.8018** | **3.407** |
| | **−0.0009** | **−0.701** |

**It costs essentially nothing.** −0.0009 Macro-F1 — inside the noise — for
−0.701 DI. On the classical model the same lever cost −0.0069 for −0.547, so on
the transformer it is both **7× cheaper and 28 % more effective**.

The likely reason: a linear model over TF-IDF has to spend capacity memorising
that "she"-features and "he"-features both map to the same job, which competes
directly with its predictive features. A contextual encoder already represents
the job independently of the pronoun, so being shown both genders mostly removes
a spurious shortcut rather than costing it anything.

### This changes the fairness-track recommendation

| candidate | Macro-F1 | DI |
|---|---:|---:|
| classical counterfactual | 0.7522 | **3.281** |
| **roberta counterfactual** | **0.8018** | 3.407 |

The classical model still has the lower DI, by 0.126. But **DI is only the
tie-break among the top 10 on Macro-F1** — a 0.752 submission may never reach
the round where fairness is scored at all. Ship
`roberta_counterfactual_fairness.csv`: +0.050 Macro-F1 buys a place at the table
for +0.126 DI.

⚠️ Single seed on the transformer side. The classical figure is a 3-seed mean.

## Post-processing on roberta-large — and a reversal (2026-07-19)

All measured on rows the tuner/sweeper never saw (validation split in half:
16,290 calibrate, 16,290 judge).

| step | Macro-F1 | Δ | DI | ΔDI |
|---|---:|---:|---:|---:|
| roberta-large, argmax | 0.8071 | — | 4.460 | — |
| **+ per-class thresholds** | **0.8262** | **+0.0191** | 4.964 | +0.504 |
| + classical ensemble | 0.8209 | +0.0139 | 4.581 | +0.122 |

### ⚠️ Threshold tuning works on the transformer — the classical verdict does not transfer

On the classical model the honest threshold gain was **≈ 0** (between −0.003 and
+0.003, see the audit above). On roberta-large it is **+0.0191** — a real, large
lever, six times bigger than anything the classical testbed suggested.

This retracts the generalisation, not the audit. Both numbers stand; what was
wrong was assuming the classical testbed's verdict would carry over. It did not,
and the ablation-on-a-cheap-proxy methodology recorded in ROADMAP §2 is exactly
what produced that mistaken expectation. **Techniques must be re-measured on the
model that ships.**

The in-sample figure would have been +0.0307 — still inflated by 1.6×, so the
nested protocol remains necessary; it just is not the difference between a real
lever and a fake one here.

The DI cost is severe: **4.460 → 4.964**. Same mechanism as before (thresholds
push predictions into rare, gender-skewed classes), and larger because the lever
itself is larger.

### The ensemble is real, and the per-class analysis predicted it

α = 0.100 on the transformer, +0.0139 over the transformer alone. Note α is
**not** interpretable as a mixing weight here: softmaxed transformer logits are
near one-hot while softmaxed SVM margins are diffuse, so a small α still lets the
transformer dominate. The sweep found the operating point empirically; the number
should not be read as "10 % transformer".

The gain was predicted before it was measured. `error_analysis.md` argued the
classical model loses Macro-F1 to confusion between neighbouring professions, so
a contextual encoder should gain most exactly there —
`reports/per_class_comparison.md` confirms it:

- mean gain on the 7 flagged bottleneck classes: **+0.0530**
- mean gain on the other 21: **+0.0380** (ratio 1.4×)
- `pastor`, the classical model's worst class, gains **+0.133**

And the complementarity is genuine in both directions — the classical model
still *beats* roberta-large on `surgeon` (−0.019), `professor` (−0.015),
`dentist` and `physician` (−0.004). Different failure modes are what an ensemble
monetises, and this one does.

### Which submission to ship

`submissions/roberta_large_6ep_tuned.csv` is the best on Macro-F1 (≈0.826
expected) but carries DI 4.96. `..._ensemble.csv` gives ≈0.821 at DI 4.58. The
accuracy track should ship the tuned one; the fairness track keeps
`classical_counterfactual_fairness.csv` (DI 3.281). Not yet tried: thresholds on
top of the ensemble, which composes the two levers.

## roberta-large is the best model, and the comparison is safe (2026-07-19)

| model | epochs run | trajectory | Macro-F1 | DI |
|---|---:|---|---:|---:|
| classical word+char SVM | — | — | 0.7643 | 3.891 |
| roberta-base (Kaggle reference, 2026-07-13) | 2 | — | 0.8035 | 4.15 |
| roberta-base | 6 | 0.7350 → … → 0.8003 → 0.8027 | 0.8027 ⚠ | 4.108 |
| **roberta-large** | **3 of 6** | 0.7710 → 0.8023 → **0.8060** | **0.8061** ⚠ | 4.126 |

**roberta-large wins with half the epochs.** The comparison is unequal — and
unequal in the direction that *penalises the winner*, which is what makes the
conclusion safe. This was stated before the numbers were known: a larger model
converges more slowly, so a shared epoch budget handicaps it; if it wins anyway,
the result holds.

Both runs are flagged by `check_convergence.py` as still improving, so both are
lower bounds. roberta-large is by far the more truncated of the two (stopped at
epoch 3 of a 6-epoch schedule, still gaining +0.0037 on its last epoch).

**What can be claimed:** roberta-large is at least as good as roberta-base, on
half the epochs, and is the best model measured on this project.
**What cannot:** by how much. 0.8061 is a floor, not a ceiling.

### The slowdown was stale VRAM, not the resume — earlier diagnosis retracted

Same checkpoint, same script, same flags, run again a few hours later:

| | VRAM at start | speed |
|---|---:|---:|
| morning restart | 14.7 / 16 GB | 1.30 s/step |
| afternoon restart | 10.4 / 16 GB | **0.45 s/step** |

**2.9× faster, and faster than the 0.54 s/step of the original from-scratch
epoch.** The difference: after the morning `kill -9`, `nvidia-smi` still reported
**7033 MiB held with no process attached**. The restarted run therefore began
with ~9 GB available instead of 16 and spilled to system RAM.

The morning write-up blamed the *resume*, because the timing correlated. That
was wrong — resuming is fine; starting on a GPU that has not actually released
its memory is not. The two memory fixes (eval batch 2×→1×, `expandable_segments`)
were not useless, but neither could do anything about 7 GB already taken.

**Operational rule: after `kill -9` on a GPU job, confirm VRAM has actually
dropped before relaunching.** A stale allocation costs a factor of three and
looks exactly like a model problem.

### Why it stopped at epoch 3 (first attempt)

roberta-large sustained **1.30 s/step after a resume** against 0.54 s/step on its
first, from-scratch epoch — so the remaining epochs needed 8.3 h against a 3.75 h
window before the GPU had to be handed back. Two memory fixes were applied (eval
batch reduced from 2× to 1× the train batch after VRAM hit 14.9/16 GB;
`expandable_segments` against allocator fragmentation) and **neither restored the
original speed**. The root cause was not identified; the correlation is with
*resuming*, not with elapsed time. Recorded as an open question rather than
explained away.

Rather than let the run hit a wall and produce nothing,
`scripts/finish_by_deadline.sh` stopped it at the epoch-3 checkpoint and
`scripts/artifacts_from_checkpoint.py` rebuilt the logits and metrics without
retraining. `metrics.json` carries `from_checkpoint` and `checkpoint_epoch` so
nothing downstream can mistake this for a completed run.

**First thing to redo with a dedicated GPU window:** roberta-large for its full
6 epochs. Everything needed is on disk — resume from
`models/roberta_large_6ep/checkpoint-17310`.

## The stack still reproduces the reference (2026-07-19)

| run | epochs | trajectory (Macro-F1 per epoch) | final | DI |
|---|---:|---|---:|---:|
| `roberta_base_repro` | 3 | 0.7462 → 0.7916 → 0.7978 | 0.7978 ⚠ | 4.134 |
| `roberta_base_6ep` | 6 | 0.7350 → 0.7746 → 0.7798 → 0.7851 → 0.8003 → **0.8027** | 0.8027 ⚠ | 4.108 |

**0.8027 against the Kaggle reference of 0.8035** — a gap of 0.0008. The concern
that `transformers` resolving to **5.x** (the reference was measured on 4.x)
might have changed the stack's behaviour is answered: it has not.

Two caveats that the headline number hides:

1. **Not a strict reproduction.** The reference ran fp32 / LR 1e-5 / 2 epochs;
   this ran fp16 / LR 2e-5 / 6 epochs. Two different routes reaching the same
   place is reassuring about the stack, but it is not the same experiment, so a
   discrepancy could not have been attributed to the library version alone.
2. **Still not converged.** `check_convergence.py` flags both runs: the 6-epoch
   one was still gaining **+0.0024** on its last epoch. 0.8027 is a tight lower
   bound, not a ceiling. The marginal gain is collapsing (+0.0062 at 3 epochs,
   +0.0152 from epoch 4→5, +0.0024 from 5→6), so the plateau is close — the
   remaining GPU time is worth more on roberta-large than on squeezing this.

**Consequence for the comparison to come:** roberta-large will run the same
6-epoch budget and, being larger, converges more slowly — so it will be *more*
penalised by truncation than roberta-base. The asymmetry is usable: if
roberta-large wins despite that handicap, the conclusion is safe. If it loses
narrowly, the result is **inconclusive**, not a verdict against it.

## Transformer runs: two ways a fixed epoch budget lies (2026-07-18/19)

**1. Three epochs was not enough, and the run said so.** roberta-base went
0.7462 → 0.7916 → 0.7978 and was **still rising by +0.0062 on its final epoch**.
Its 0.7978 is a lower bound, not a result. Had roberta-large been run against it
at the same 3-epoch budget, the larger model — which needs *more* epochs, not
fewer — would very likely have "lost", and the conclusion would have been the
exact opposite of the truth. `scripts/check_convergence.py` flags this
automatically from the per-epoch history, and `gpu_queue.sh` now excludes
truncated runs from the ranking.

Budget raised to 6 epochs with early stopping (patience 2), so each model stops
at **its own** plateau rather than at a shared arbitrary ceiling.

**2. The same epoch number means different things under different budgets.** At
epoch 3, the 3-epoch run scored 0.7978 and the 6-epoch run only 0.7798 — the
*same model on the same data*. Nothing regressed: the learning rate decays
across the whole planned schedule, so at 3-of-3 the LR had annealed to ~0 while
at 3-of-6 it is still near half its peak. A model mid-schedule is not a model
that has finished.

Practical rule: **never compare two runs epoch-by-epoch when their total budgets
differ.** Only the final, post-decay score is comparable — and only if both runs
converged.

## Negative result: no classical hyper-parameter beats the default (2026-07-18)

11 configs, one change at a time from the current default, full 217k data,
fit 152k / select 32.6k / report 32.6k. **All 11 converged** (`n_iter_` below
`max_iter`), so every comparison is between finished fits — the check that was
missing when the 2020 project published a retracted comparison.

| config | select | report |
|---|---:|---:|
| baseline (C=1, min_df=5, char 2–5, sublinear) | 0.7526 | 0.7608 |
| C=0.5 | **0.7554** | 0.7603 |
| C=2 | 0.7492 | 0.7583 |
| C=4 | 0.7469 | 0.7550 |
| min_df=2 | 0.7530 | 0.7609 |
| min_df=10 | 0.7498 | 0.7588 |
| char 3–5 | 0.7523 | 0.7607 |
| char 2–6 | 0.7529 | **0.7619** |
| word 1–3 | 0.7525 | 0.7601 |
| no sublinear_tf | 0.7468 | 0.7555 |
| char hash 2²¹ | 0.7534 | 0.7607 |

`C=0.5` wins the selection split by **+0.0028** over the baseline and then loses
to it by **−0.0005** on the report split: a **selection bias of +0.0034**, on a
single free parameter. The honest verdict is that **the default configuration is
already at a local optimum** — nothing here is worth changing.

`char 2–6` is the best on the report split (+0.0011), but it ranked 4th on the
selection split. Picking it *because* it tops the report column is precisely the
error this three-way split exists to prevent, so it stays unshipped.

**The most useful number in this table is not in it.** The same baseline config
scores **0.7526 on one split and 0.7608 on another — a gap of +0.0083**, while
the entire 11-config sweep spans only **0.0069**. Which random split you evaluate
on moves the score more than any hyper-parameter tested. Consequence for the rest
of the project: **treat any classical difference below ~0.005 as unmeasurable
without multiple seeds**, and be suspicious of any single-split result that
claims less.

## ⚠️ The disparate-impact metric is gameable — read before optimising it

Found while designing the λ-sweep threshold tuner (ROADMAP §3 item 16), and
pinned by `tests/test_di_edge_cases.py`.

`macro_disparate_impact` averages `max(M,F)/min(M,F)` per predicted job. For a
job predicted for a **single gender**, the other column is NaN, and pandas'
`max`/`min` skip NaN — so both return the same count and the ratio is **1.0,
i.e. scored as perfect parity**. Concretely:

| predictions | DI |
|---|---:|
| job a = 10 M / 1 F, job b = 5 M / 5 F | **5.50** |
| job a = 11 M / **0 F**, job b = 5 M / 5 F | **1.00** |

Removing every woman from a job *improves* the fairness score by 4.5 points.
(A job never predicted at all simply drops out of the mean — that case is
harmless.)

This is inherited from the organisers' reference notebook and **must not be
"fixed"**: the metric is pinned to their published 3.898171170378378 and is what
the competition scores. But it has two consequences:

1. **Any procedure that optimises DI directly will find this** — the proposed
   λ-sweep tuner, adversarial debiasing, or a manual post-processing rule. It
   would produce an excellent DI that is maximally unfair.
2. Every fairness number must therefore be reported with
   `count_single_gender_jobs`. **Current status: 0 of 28 predicted jobs are
   single-gender on the holdout**, so none of the results in this file are
   affected — the trap is latent, not sprung.

The docstring in `metrics.py` previously claimed such jobs "drop out of the
mean", which is not what the code does; corrected without touching the
computation.

## Fairness track — Pareto front, full data (2026-07-18)

All five variants fit on the same 184.6k train split and scored on the same
32.6k holdout, so the numbers are directly comparable. No variant was used to
select anything, so none carries selection bias.
(`scripts/fairness_pareto.py`, results in `reports/fairness_pareto.json`.)

**3 seeds (7, 42, 2024)**, deltas **paired** — each variant compared against the
no-mitigation baseline *of its own seed*, so the seed's intrinsic difficulty
cancels instead of inflating the spread. Full table:
`reports/fairness_pareto_multiseed.md`.

| variant | Macro-F1 | ΔF1 (paired) | DI | ΔDI (paired) | F1 cost per DI point |
|---|---:|---:|---:|---:|---:|
| none | 0.7591 ± 0.0049 | — | 3.828 ± 0.094 | — | — |
| name masking (NER) | 0.7575 ± 0.0051 | −0.0016 ± 0.0015 | 3.816 ± 0.051 | −0.012 ± 0.073 | — |
| **gender scrubbing** | 0.7566 ± 0.0039 | −0.0025 ± 0.0010 | 3.462 ± 0.024 | −0.366 ± 0.071 | **0.0068** |
| scrubbing + masking | 0.7563 ± 0.0032 | −0.0028 ± 0.0027 | 3.464 ± 0.059 | −0.365 ± 0.100 | 0.0077 |
| **counterfactual training** | 0.7522 ± 0.0060 | −0.0069 ± 0.0019 | **3.281 ± 0.061** | **−0.547 ± 0.105** | 0.0126 |

Pairing matters: it shrank the uncertainty on the scrubbing effect enough to
make it unambiguous, where the raw per-seed DI values overlap heavily.

**Settled:**

- **Name masking does nothing measurable.** ΔDI **−0.012 ± 0.073** — the effect
  is smaller than its own spread. This closes a *todo* open since the start of
  the project: the hypothesis was that first names leak gender as strongly as
  pronouns, and on this data they do not, once pronouns are already there.
  `scrub+mask` (−0.365) is indistinguishable from `scrub` alone (−0.366), which
  is the same conclusion reached a second way. Drop the spaCy dependency.
- **On one seed, `scrub` beat `scrub+mask` on DI; on another the order flipped.**
  The earlier "dominated" verdict was noise, and declining to act on it was
  correct.

**Neither survivor dominates the other — they are both on the front:**

| | reaches | costs |
|---|---|---|
| gender scrubbing | DI 3.462 | 0.0068 Macro-F1 per DI point — **best rate** |
| counterfactual training | **DI 3.281** — furthest | 0.0126 per DI point |

Counterfactual buys 0.18 more DI than scrubbing, at roughly double the price per
point. For reference, threshold tuning at λ=0.05 costs **0.026** per DI point —
so scrubbing is ~4× and counterfactual ~2× more efficient than post-hoc
thresholding. Changing what the model learns beats redistributing fixed scores.

**Shipping recommendation:** **counterfactual training** for the fairness-track
submission — that submission exists to minimise DI, and it goes furthest, while
*adding* signal rather than removing it (the test text stays untouched).
`scrub` is the pick if Macro-F1 on the fairness submission matters too. The
accuracy track keeps no mitigation.

Both mitigations are worth re-testing on the transformer: it can exploit context
that a bag-of-words cannot, so the accuracy cost may well be smaller there.

## Threshold-only fairness front, and a caveat on the threshold gain itself

`scripts/threshold_fairness_front.py` runs the per-class bias search against
`Macro-F1 − λ·DI` and sweeps λ. Bias fitted on one half of the holdout (16.3k),
every number below read off the other half (16.3k). **0 single-gender jobs at
every λ**, so the gaming vector above was not triggered and these DI values are
honest.

| | Macro-F1 | ΔF1 | DI | ΔDI |
|---|---:|---:|---:|---:|
| argmax | 0.7653 | — | 4.105 | — |
| λ = 0 (pure Macro-F1) | 0.7623 | **−0.0030** | 4.444 | +0.339 |
| λ = 0.005 | 0.7663 | +0.0010 | 4.395 | +0.289 |
| λ = 0.01 | 0.7658 | +0.0005 | 4.337 | +0.231 |
| λ = 0.02 | 0.7611 | −0.0042 | 4.122 | +0.016 |
| **λ = 0.05** | 0.7509 | −0.0144 | **3.554** | **−0.552** |
| λ = 0.1 | 0.7219 | −0.0434 | 3.610 | −0.495 |

**1. The threshold gain depends on how much calibration data it gets.** At λ=0
this run measures **−0.0030** — the technique *hurts* — where the nested audit
measured **+0.0032**. The difference is the calibration set: 16.3k rows here
versus 32.6k there. Halve the data behind 28 free parameters and the gain flips
sign. Single seed, so the exact value is soft, but the direction matters.

There is an asymmetry worth being precise about: an honest *measurement* must
hold half the validation data back, while *deployment* refits the bias on all of
it. So the measured figure is a **lower bound** on what ships, not an estimate
of it. The practical reading: the deployed gain is somewhere between −0.003 and
+0.003, i.e. **indistinguishable from zero**, and certainly not the +0.008
originally published.

**2. Thresholds are a poor fairness lever compared with counterfactual
training.** Both reach roughly the same DI, at very different prices:

| lever | ΔDI | ΔF1 | F1 cost per DI point |
|---|---:|---:|---:|
| counterfactual training | −0.546 | −0.0052 | **0.010** |
| threshold tuning, λ=0.05 | −0.552 | −0.0144 | 0.026 |

Counterfactual training is about **2.6× more efficient**. It changes what the
model learns; thresholding only redistributes a fixed set of scores, and pays
for every DI point with real accuracy. Ship the counterfactual model for the
fairness track.

**3. The front is not monotone** — λ=0.1 is worse than λ=0.05 on *both* axes.
Coordinate ascent is greedy and the objective gets harder as λ grows, so it
lands in worse local optima. Do not read the λ=0.1 row as a reachable
trade-off; it is a search failure, and a reminder that the front's shape is a
property of the optimiser as much as of the problem.

## Next candidates

- Classical tuning: `min_df`, `C`, `sublinear_tf`, char range (3–5), TF-IDF
  `max_features`; add a Multinomial-NB feature (NB-SVM trick).
- Fairness track: quantify the `--scrub-gender` accuracy/DI trade-off.
- Transformer (Step C): DeBERTa-v3-base fine-tune on GPU.

## Ablation study (technique deltas)

Measured on a classical testbed (hashed word+char LinearSVC, 72k/18k stratified
subsample, zero GPU) while the transformer waits on GPU. Deltas transfer
qualitatively; augmentation is expected to help the transformer, not the linear
model.

| Technique | Macro-F1 | delta |
|---|---:|---:|
| baseline (argmax) | 0.7420 | — |
| + per-class threshold tuning | 0.7502 | **+0.0082** |
| + rare-class augmentation | 0.7392 | -0.0028 |
| + augmentation + thresholds | 0.7474 | +0.0054 |

**Findings:** (1) threshold tuning is a real, free Macro-F1 lever (+0.8pt) and
should transfer to the transformer. (2) EDA/counterfactual augmentation *hurts*
the bag-of-words model (paraphrase diversity is noise to TF-IDF) — it must be
tested on the contextual transformer, where it is expected to help the rare
classes, not on the linear model.

> ⚠️ **Finding (1) above is retracted — see the audit below.** The +0.8 pt was
> measured on the same rows that chose the thresholds. The real gain is +0.3 pt.

## ⚠️ Audit: the threshold-tuning gain was measured in-sample (2026-07-18)

`tune_thresholds.py:68-69` and `ablation_classical.py:87-88` learned the
per-class bias on the holdout and then reported Macro-F1 **on that same
holdout**. The search has 28 free parameters over a 33-point grid for up to 12
rounds — ample capacity to fit one split's noise. Every "+0.8 pt" figure in this
file and in ROADMAP §2 carried that optimism.

Re-measured honestly with a nested split (`scripts/audit_threshold_tuning.py`):
the classical model is fit on 152k rows, the bias is tuned on a 32.6k
**calibration** set, and the score is reported on a 32.6k **eval** set that
neither the model nor the tuner ever saw. Three seeds.

| | Macro-F1 delta | sd (3 seeds) |
|---|---:|---:|
| tuned on calib, judged on eval — **the real gain** | **+0.0032** | 0.0009 |
| tuned on eval, judged on eval — the old methodology | +0.0071 | 0.0014 |
| what the tuner believes it achieved, on its own calib rows | +0.0074 | — |

The old method **overstates the gain by ×2.2**. The tuner's self-report
(+0.0074) lands on the ROADMAP's published +0.0082, which corroborates that the
published figure is the self-reported one rather than a generalising estimate.

**The gain is real but small**: +0.0032 with a seed-to-seed sd of 0.0009 is
consistently positive, so this is not noise — it is simply a third of what was
claimed.

**It also costs fairness, which was never measured**: the old script computed no
disparate impact for the tuned predictions.

| | DI (lower = fairer) |
|---|---:|
| argmax | 3.87 |
| + tuned thresholds | 4.14 (**+0.262**, sd 0.062) |

The mechanism is now measured rather than assumed (`reports/error_analysis.md`):
**rare jobs are the gender-skewed ones.**

| | mean per-job DI ratio |
|---|---:|
| 10 rarest jobs | **5.36** |
| 10 most frequent jobs | **3.03** |

Spearman correlation between support and DI ratio is **−0.387**. The worst
offenders are `dietitian` (12.0), `nurse` (10.1), `rapper` (9.9), `model` (6.7),
`dj` (6.7) — mostly rare, all strongly stereotyped. Threshold tuning exists
precisely to push more predictions into rare classes, so it feeds the most
skewed jobs and drags the average of the ratios up. The +0.26 DI is not a
side-effect to be engineered away; it is what the technique does.

Since DI is the **tie-break for the top 10**, a +0.3 pt Macro-F1 gain bought
with +0.26 DI is a trade-off to decide deliberately, not a free lever. ROADMAP
§3 item 16 proposes the fix: run the same coordinate ascent against
`Macro-F1 − λ·DI` and sweep λ, which traces the whole front from one set of
saved logits at zero GPU cost.

**Consequences:**
- `submissions/classical_tuned.csv` should be expected to score ≈ **0.767**, not
  the 0.7714 logged in row C-thr.
- Threshold tuning stays worth applying to the transformer (Tier 1), but budget
  +0.3 pt, and re-check the DI cost there.
- `tune_thresholds.py` now splits the validation set in half — tunes on one,
  reports on the other — and refits on everything only for the deployed bias.
