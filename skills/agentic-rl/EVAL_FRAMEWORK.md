# Evaluation Framework — LLM-guided curriculum RL for Isaac Lab

**Status:** v1 (2026-07-09) · Applies to the Phase-1 ON-vs-OFF pilot and every experiment after.

This document pre-registers *what we measure, how we decide, and what we may claim* — written
BEFORE looking at a run's outcome so the analysis can't be steered to a conclusion. It exists
because the prior SONIC system's headline claim did not survive adversarial review; the bar
here is **trustworthiness**, not "the manager won."

---

## 0. The one-sentence question

> Does an LLM/rule outer loop that **adaptively** adjusts curriculum knobs (gated on a protected
> held-out metric) beat (a) fixed defaults [**control**] and (b) an **open-loop replay of its own
> realized knob ladder** [**scripted**] — by more than run-to-run noise?

The scripted arm is the load-bearing comparator: beating *control* only shows "moving the knob
helps"; beating *scripted* is the only thing that shows "**reading the digest to decide WHEN to
move helps**" — the actual adaptivity claim.

---

## 1. Metrics

### 1.1 Primary metric (the headline, protected)
**`heldout_success_rate`** — fraction of steps tracking a FROZEN, salted-hash-split grid of
in-envelope velocity commands within a fixed tolerance (0.25). Protected: the manager cannot
see the grid composition, reweight it, re-threshold it, or add it to training (holdout.py +
process boundary). Reported as the **final-segment** value and the **mean of the last 2
segments** (less noisy).

Rationale for "final/last-2" not "area under curve": all arms share the warm-start, so early
segments are near-identical; the signal is where they diverge (the end).

### 1.2 Secondary metrics (diagnostic, never the headline)
- **Per-condition held-out** (`heldout_per_condition`): per-command success. Guards the
  SONIC failure mode where **one dominating condition** drove 51–96% of apparent gains. We
  report the *worst-condition* success and the spread, not just the mean.
- **Sample efficiency**: held-out at a fixed iteration budget (all arms run equal iters, so
  final held-out already controls for this).
- **Standard `success_rate`** (unprotected eval): reported only to show it tracks the held-out
  metric; NEVER used to compare arms (Goodhart — it's the manager's own decision input surrogate).
- **`mean_tracking_error`**: read jointly with success (a policy can lower error on the easy
  commands while failing hard ones — per-condition catches this).

### 1.3 Metrics we deliberately do NOT use
- Aggregate training reward (`Episode/rew_mean`) as a headline — it's the surrogate the manager
  steers via, and it's not the protected metric.
- Any metric the manager can influence through its action space.

---

## 2. The noise band (chaos floor) — the crux of a trustworthy verdict

A difference between arms is only real if it exceeds run-to-run noise. RL training is chaotic;
identical configs diverge. We MUST measure this floor for the Isaac Lab/skrl engine — the
`E5B_CHAOS_FLOOR_MEAN=1.31e-2` in `equivalence.py` is a SONIC/A10G measurement and
`equivalence.py` explicitly warns to re-measure when the engine/env-count/horizon change (all
differ here).

### 2.1 The probe (pre-experiment, one-time per config)
Run the SAME config (warm-start, segment schedule, knobs) **≥3 times** varying only the seed
(and once with an epsilon loss perturbation, E5b-style). Measure the **std of the final
`heldout_success_rate`** across runs → `sigma_noise`. Calibrate the equivalence tolerance:

    tau = calibrate_tau(chaos_floor_dev = sigma_noise / mean_heldout, safety_factor = 3)

(`calibrate_tau` lives in `runmanager/core/equivalence.py`; it also refuses a τ that would
swallow the smallest effect we care about.)

### 2.2 The verdict gate
Compare each arm pair's held-out trajectory with `EquivalenceGate(tau).compare(a, b)`:
- `within_tau` → **EQUIVALENT** (the arms are the same run within noise; the classic null result).
- `diverged` → a real difference; sign says which arm is better.
- `incomparable` → length/NaN mismatch (a bug, not a result).

**A manager-beats-scripted claim requires `diverged` in the manager's favor, reproduced across
seeds.** Anything else is reported as "equivalent within noise" (which, per the SONIC prior, is
the expected outcome and still a valid, publishable finding).

---

## 3. Decision rule (pre-registered)

| Observation (on held-out, after the noise band is known) | Verdict |
|---|---|
| manager − scripted **> τ**, same sign across ≥3 seeds | adaptivity helps (the strong claim) |
| \|manager − scripted\| **≤ τ** (within_tau) | **no measured benefit of adaptivity** (the SONIC-consistent null) |
| manager − scripted **< −τ** | adaptivity HURTS (also a real finding) |
| manager − control **> τ** but manager ≈ scripted | the KNOB helps, not the digest-reading (curriculum works; adaptivity doesn't add) |
| any arm shows a rollback / segment_failed | run is invalid for comparison — fix and rerun |

Sample-size honesty: **n=1 (the pilot) supports NO row above.** The pilot's only claims are:
(i) the machinery runs end-to-end and clean, (ii) the manager makes real gated decisions,
(iii) the held-out metric is protected + discriminating. The comparative verdict needs the
multi-seed experiment (§2.1).

---

## 3a. The lever–metric coupling prerequisite (learned from the v2 pilot)

**A null result is only meaningful if the knob CAN move the metric.** The v2 manager arm made 3
real hardening moves (command range 1.0→1.25→1.5→1.75) yet held-out barely moved
(0.34→0.39→0.37, range 0.047 — indistinguishable from drift). Reason: the held-out grid is FIXED
in-envelope (|vx|≤1); widening the *training* command range changes the training distribution but
is evaluated on the same in-envelope commands, so its effect on in-envelope held-out is
theoretically ambiguous and empirically ~zero here.

**Consequence:** an ON-vs-OFF null on THIS lever+metric pair would be uninformative — it can't
distinguish "adaptivity doesn't help" from "this knob doesn't move this metric for anyone." Before
running the multi-seed experiment we MUST establish **lever sensitivity**: does the knob move the
metric by more than τ for SOMEONE (e.g. control-at-widest vs control-at-narrowest)? If not, either
(a) pick a lever the metric responds to (optimizer knobs if the run is under-converged; reward
weights that trade off tracking directly; or a HARDER-tolerance held-out so the policy has room to
improve), or (b) pick a metric the lever moves (evaluate held-out at command magnitudes the range
knob actually gates — but keep it protected). This is a **precondition**, added to the validity
gates below:

- **(G7) Lever sensitivity:** the manager's knob, swept across its range on the control policy,
  changes the primary metric by > τ. Else the experiment cannot detect adaptivity on this pair —
  report that and re-pair before claiming a null.

---

## 4. Validity gates (must all pass, else the run is void — enforced by `verify_gate.py`)

1. All arms share the identical warm-start checkpoint and segment schedule.
2. The scripted ladder **exactly** equals the manager's realized applied (tick→knob→value).
3. Held-out metric populated every segment, in [0,1], **non-degenerate** (per-condition spread > 0;
   not floored at 0 by an unachievable grid, not saturated at 1).
4. No `segment_failed`, no crash-induced rollback mislabeled as success, `config_verify: ok`.
5. Eval runs at the pinned, manager-untouchable settings (`-Play-v0`, no knob leak).
6. Held-out grid is IN-ENVELOPE (protection is "manager can't see/steer it," not "it's
   impossibly hard").

---

## 5. What the Phase-1 pilot can and cannot conclude

**Can:** the closed loop trains, evals, verifies config, gates on a protected metric, makes
competence-gated curriculum moves, derives a faithful scripted replay, and the guardrails
(tripwire/rollback/pending-gate) behave. It also reveals the **operating regime** (held-out
ceiling ~0.40 on rough Anymal-C at this budget) needed to calibrate any real experiment.

**Cannot:** say whether adaptivity beats scripted. That is a multi-seed comparison against a
measured noise band (§2), which is the next experiment, not the pilot.

---

## 6. The real experiment (post-pilot)

1. Chaos-floor probe: 3 seeds × the control config → `sigma_noise` → `tau` (§2.1).
2. 3 seeds × {control, manager, scripted}, identical warm-start per seed.
3. Per-seed: `EquivalenceGate(tau)` on manager-vs-scripted and manager-vs-control held-out.
4. Aggregate: report the verdict row (§3) with the per-seed spread and per-condition breakdown.
5. Honest write-up: state n, τ, the null-result prior, and what would change the verdict.
