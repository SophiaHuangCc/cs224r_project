# Reward Hacking in LLM-Generated Reward Functions: Measurement and Mitigation

**CS224R Final Project — Reward Hacking in Embodied Agents**

> Consolidated results for the report. All numbers are hand-verified against
> `results/eval/<method>/<task>_<ckpt>k.json` and the training summaries in
> `results/{train,kl,physics}/`. Unless noted, figures are the **500k-step**
> checkpoint, 100 evaluation episodes, SAC+HER, seed 42.

---

## 1. Setup

We study whether reward functions written by an LLM induce **reward hacking** —
optimizing the proxy reward without achieving the true objective — on three
MuJoCo Fetch manipulation tasks, and whether alignment-style mitigations help.

| | |
|---|---|
| **Tasks** | FetchReach (saturated control), FetchPickAndPlace, FetchSlide |
| **Algorithm** | SAC + Hindsight Experience Replay, 500k steps, eval every 100k |
| **Reward sources** | *Vanilla* = single-shot LLM reward (one GPT-4o call); *Eureka* = iterated best-of-N LLM reward |
| **Mitigations on Eureka** | Reward **Ensemble** (min of N=3 independent LLM rewards); **KL** penalty to a sparse-reward reference policy (β∈{0.01,0.1,1.0}); **Physics** constraint penalty (coef∈{0.1,1.0,10.0}) |

**Metrics.** We report two orthogonal axes plus safety:

- **Task success rate** — fraction of episodes solving the true task
  (*capability*).
- **Reward alignment** — Pearson correlation between per-episode proxy reward
  and task success (*hackability*). A well-aligned reward ranks successful
  episodes above failures (r→1); a **low or zero correlation means the proxy can
  be maximized without succeeding — the signature of a hackable reward.** This
  metric is parameter-free (no threshold to tune) and, unlike a count of
  "high-reward failures," is not mechanically tied to the success rate.
  It is *undefined* when every episode shares the same outcome (all-success or
  all-fail → no variance), reported as **n/a**.
- **Safety** — fraction of steps violating action-norm / jerk tripwires.

> **Why correlation, not a "hacking rate."** A naive hacking rate (high proxy
> reward + failure, thresholded at the run's median) turned out to equal
> ≈ ½·(1 − success rate) across all 81 runs (r = −0.95 with success) — it merely
> re-encodes the success rate. Reward–success correlation measures alignment
> directly and independently. See §6.

---

## 2. Headline results (500k)

### Task success rate (%) — higher is better
| Method | Reach | PickAndPlace | Slide |
|---|---:|---:|---:|
| Vanilla (single-shot LLM) | 100 | 37 | 14 |
| **Eureka (iterated LLM)** | 100 | 52 | **64** |
| Eureka + Ensemble (min, N=3) | 100 | 4 | 52 |
| **Eureka + KL (β=0.01)** | 100 | **93** | 58 |
| Eureka + Physics (c=1.0) | 100 | 3 | 62 |

### Reward alignment (proxy–success correlation) — higher is better
| Method | Reach | PickAndPlace | Slide |
|---|---:|---:|---:|
| Vanilla (single-shot LLM) | n/a | 0.88 | **0.28** |
| Eureka (iterated LLM) | n/a | 0.96 | **0.87** |
| Eureka + Ensemble (min, N=3) | n/a | 0.97 | 0.79 |
| Eureka + KL (β=0.01) | n/a | 0.72 | 0.87 |
| Eureka + Physics (c=1.0) | n/a | 0.99 | 0.82 |

*(Reach is n/a — every policy solves it 100%, so there is no outcome variance to
correlate against; it is a sanity control.)*

**Headline:** exactly one reward is misaligned — the **single-shot LLM reward on
Slide (r = 0.28)** — and Eureka's iteration fixes it (r = 0.87) while raising
success 14% → 64%. Every other reward, including the mitigated ones, is
well-aligned (r ≥ 0.72); the mitigations therefore move *capability and safety*,
not alignment.

---

## 3. Vanilla vs. Eureka — iteration repairs a misaligned reward

| Task | Metric | Vanilla (single-shot) | Eureka (iterated) | Δ |
|---|---|---:|---:|---|
| PickAndPlace | Success | 37% | 52% | +15 pts |
| PickAndPlace | Alignment (r) | 0.88 | 0.96 | both aligned |
| Slide | Success | 14% | 64% | **+50 pts** |
| Slide | **Alignment (r)** | **0.28** | **0.87** | **misaligned → aligned** |

**Mechanism (Slide).** The single-shot reward is pure distance shaping:

```python
reward = -‖achieved_goal − desired_goal‖ − 0.01·‖action‖      # vanilla
```

A puck that *stops next to* the goal earns nearly the same reward as one that
*reaches* it, so proxy reward barely tracks success (r = 0.28) — the agent can
climb the reward without solving the task. Eureka's iterated reward adds a
**discrete success bonus** that opens a gap between "near" and "solved":

```python
if ‖achieved_goal − desired_goal‖ < 0.05:  return 1.0           # eureka
return -‖achieved_goal − desired_goal‖
```

This lifts alignment to r = 0.87 and quadruples success. **Iterative reward
design is itself the most effective anti-hacking measure we observed.** As
corroboration, the absolute fraction of *failures that reach success-level
reward* is 15% for vanilla Slide vs 0% for Eureka.

---

## 4. Eureka vs. mitigations

Because Eureka is already well-aligned, the mitigations are judged on whether
they **preserve success** and **improve safety** without inducing collapse.

### 4.1 Success and alignment vs. the Eureka baseline (500k)
| Method | PickAndPlace Succ / r | Slide Succ / r | Verdict |
|---|---|---|---|
| Eureka (baseline) | 52% / 0.96 | 64% / 0.87 | — |
| + KL β=0.01 | **93% / 0.72** | 58% / 0.87 | **success ↑↑, still aligned** |
| + Ensemble (min) | 4% / 0.97 | 52% / 0.79 | PickAndPlace **collapse** |
| + Physics c=1.0 | 3% / 0.99 | 62% / 0.82 | PickAndPlace collapse, Slide safe |

Note the collapsed cells (Ensemble/Physics PickAndPlace, 3–4% success) keep
**r ≈ 0.97–0.99**: the reward still correctly ranks the rare successes above
failures, confirming these are *optimization failures, not hacking*. (A
median-style metric mislabeled them as 48% "hacking" — see §6.)

### 4.2 Safety (% steps violating tripwire) — Eureka vs. best mitigations
| Task | Method | Action viol | Jerk viol | Mean action norm | Mean jerk |
|---|---|---:|---:|---:|---:|
| PickAndPlace | Eureka | 15.5 | 24.4 | 0.99 | 0.58 |
| PickAndPlace | + KL β=0.01 | 23.3 | 57.5 | 1.23 | 1.16 |
| Slide | Eureka | 11.9 | 14.3 | 1.17 | 0.42 |
| Slide | + KL β=0.01 | **2.2** | 9.9 | 0.93 | 0.36 |
| Slide | + Physics c=1.0 | **2.9** | **4.8** | 1.04 | 0.24 |

**Reading the mitigations.**
- **KL penalty (β=0.01) — the clear win.** Anchored to a sparse-reward reference
  (Reach 100%, PickAndPlace 54%, Slide 64% on its own), it acts as a soft trust
  region: PickAndPlace success 52% → 93% (and faster at every checkpoint — 250k:
  25% vs Eureka's 18%), and Slide action violations cut 5× (11.9% → 2.2%). Cost:
  jerkier PickAndPlace motions (tunable).
- **Reward ensemble (min) — optimization failure, not anti-hack.** The
  pessimistic `min` over three LLM rewards removes the learning signal on
  PickAndPlace; the policy **freezes** (mean object speed ≈ 2×10⁻⁴, success 4%).
  Its low violation rates are an artifact of not moving. Roughly neutral on Slide
  (64% → 52%).
- **Physics constraint — safety knob, coefficient-sensitive.** c=1.0 on Slide
  preserves success (64% → 62%) while halving action/jerk violations (11.9→2.9,
  14.3→4.8) — the best safety/quality trade in the study — but crushes
  contact-rich PickAndPlace (52% → 3%); c=10.0 collapses every task.

---

## 5. Hyperparameter ablations

### KL penalty strength β (success % / alignment r, 500k)
| β | PickAndPlace | Slide | Note |
|---|---|---|---|
| 0.01 | **93 / 0.72** | **58 / 0.87** | sweet spot |
| 0.1 | 17 / 0.97 | 6 / 0.75 | over-constrained |
| 1.0 | 4 / 0.99 | 0 / n/a | collapse (≈ frozen at reference) |

### Physics constraint coefficient c (success % / alignment r, 500k)
| c | PickAndPlace | Slide | Note |
|---|---|---|---|
| 0.1 | 27 / 0.97 | 47 / 0.87 | mild |
| 1.0 | 3 / 0.99 | **62 / 0.82** | Slide safety win / PickAndPlace collapse |
| 10.0 | 3 / 0.99 | 0 / n/a | collapse |

Both mitigations show a sharp **over-constraint cliff**: beyond a small penalty
the agent earns *low* reward and fails (success → single digits) while alignment
*stays high* — i.e. the failure is in capability, not in reward integrity.
FetchReach stays at 100% success under every setting (saturated control).

---

## 6. On measuring hacking

We initially used a "hacking rate" = fraction of episodes whose proxy reward
exceeds the run's **own median** while failing. Across all 81 runs this equals
≈ ½·(1 − success rate) to within 4 points (corr with success = **−0.95**):
e.g. a not-yet-trained agent on the trivially-solvable Reach task would score
~50% "hacking" at 0% success. The metric re-encodes the success rate rather than
measuring exploitation, and mislabels training collapses (ensemble PickAndPlace:
48% "hacking" despite the policy freezing and earning proxy reward far *below*
success level).

**Reward–success correlation** avoids both failure modes: it has no threshold,
is independent of the success rate, and cleanly separates the two axes —
*low success + high r* = undertrained/collapsed (e.g. Physics PickAndPlace,
r=0.99), *low r* = genuinely hackable reward (vanilla Slide, r=0.28). The lone
misalignment in the whole study is the single-shot Slide reward.
*(Pre-correction median-split tables are archived in `checkpoint_old.md`.)*

---

## 7. Takeaways

1. **LLM rewards are well-aligned here, with one exception.** Only the
   *single-shot* Slide reward is hackable (r = 0.28); Eureka's iterated reward
   repairs it (r = 0.87) and quadruples success — reward *design* beats reward
   *patching*.
2. **Alignment and capability are orthogonal.** Mitigations that tank success
   (Ensemble-min, KL β≥0.1, Physics c≥1.0 on contact-rich tasks) keep r ≈ 0.97+
   — they cause optimization failures, not hacking.
3. **The best mitigation is a soft KL trust region (β=0.01):** +41 pts of
   PickAndPlace success and 5× fewer Slide action violations, alignment intact.
4. **Physics penalties are a task-dependent safety knob:** a clear win on Slide
   (≈ unchanged success, halved violations), harmful on PickAndPlace.
5. **Measurement matters.** The headline depends entirely on the metric: a
   median-split "hacking rate" would have manufactured a 40–50% hacking story
   that is really just the inverse success rate.

> **Caveat (single seed).** All runs use seed 42; mid-training success can swing
> (e.g. KL β=0.01 Reach dips to 2% at 250k before recovering to 100%).
> Cross-method gaps are the story; small differences are within run-to-run
> variance. Multi-seed error bars are the key next step.
