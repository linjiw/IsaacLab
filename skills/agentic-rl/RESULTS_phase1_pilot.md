# Phase-1 pilot results (v2, single seed) — 2026-07-10

**Scope:** a single-seed 3-arm pilot on `Isaac-Velocity-Rough-Anymal-C-v0` (skrl, warm-started
from a 400-iter policy, 6 segments × 150 iters, band 0.20/0.33, sustain 2, in-envelope held-out
grid). Per `EVAL_FRAMEWORK.md`, **n=1 supports no adaptivity claim** — this pilot validates the
machinery and generates hypotheses. The comparative verdict needs the aligned cross-seed
experiment (`NEXT_EXPERIMENT.md`).

## What ran (all validity gates passed)
`verify_gate.py`: all HARD gates PASS — no `segment_failed`, held-out populated + in [0,1] +
non-degenerate (per-condition spread 0.50–0.64) for every arm, `config_verify: ok`, no rollbacks,
scripted faithfully replays the manager ladder. The same-seed degeneracy is flagged as a soft note.

## Held-out trajectories (protected metric, per segment)
| arm | s1 | s2 | s3 | s4 | s5 | s6 | AUC | final |
|---|---|---|---|---|---|---|---|---|
| control  | 0.341 | 0.347 | 0.397 | 0.398 | 0.374 | 0.456 | **0.386** | 0.456 |
| manager  | 0.341 | 0.347 | 0.389 | 0.379 | 0.371 | 0.372 | **0.367** | 0.372 |
| scripted | 0.341 | 0.347 | 0.389 | 0.379 | 0.371 | 0.372 | **0.367** | 0.372 |

Manager ladder (realized): t2 range→1.25, t4→1.5, t6→1.75 (3 competence-gated hardenings).

## Findings

1. **Machinery validated end-to-end on real GPU training.** Host-driven driver → docker-exec
   train/eval, config-drift verification (`ok` every segment), protected held-out eval, tripwire
   arming, journal — all worked across 18 real training segments.

2. **Same-seed manager == scripted, bit-identical** (AUC diff +0.0000; equivalence gate =
   `bit_identical`). Empirical proof that skrl is bit-deterministic and that the ON-vs-OFF
   ablation MUST be cross-seed (`EVAL_FRAMEWORK §0a`). The pilot's manager-vs-scripted delta is a
   tautology, not a result.

3. **The command-range lever HURTS this metric (real, directional).** manager − control =
   **−0.019 AUC** (−0.084 final); the equivalence gate calls manager-vs-control **`diverged`**
   (mean-rel-dev 0.049 > placeholder τ 0.039). Directional over segments: control beats manager
   by +0.008 (s3) → +0.019 (s4) as widening grows. Mechanism: widening the *training* command
   range spends capacity on fast commands the *narrow fixed* held-out never tests → negative
   transfer. **This is the lever–metric misalignment** (`GAPS.md A0`), now an empirical effect
   exceeding even the placeholder noise band.

4. **AUC vs final matters.** Control's s6 jumped to 0.456 (a genuine late improvement, not an
   artifact — per-condition well-formed), inflating its *final* far more than its *AUC*. The AUC
   gap (−0.019) is the stable read; the final gap (−0.084) overstates it. Confirms the
   sample-efficiency reframing (`EVAL_FRAMEWORK §1.1`).

## Consequences for the next experiment (`NEXT_EXPERIMENT.md`)
- The comparison must be **cross-seed** (scripted replays a donor seed's fixed ladder).
- The held-out grid must be **aligned** to the lever: a WIDE grid that demands the competence
  widening builds, so widening is rewarded rather than penalized. (Else, switch to an optimizer
  lever + narrow grid.)
- τ must be **measured** (seed variance; training is bit-deterministic so fp-chaos ≈ 0).
- Score on **AUC**, report per-condition worst/spread, honest n.

Raw journals + full gate/analysis text: `_runs/pilot/` and `_runs/pilot_v2_{GATE,ANALYSIS}.txt`
(gitignored — regenerate with `verify_gate.py _runs/pilot` / `analyze_pilot.py _runs/pilot`).
