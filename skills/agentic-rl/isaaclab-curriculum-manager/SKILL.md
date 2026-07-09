---
name: isaaclab-curriculum-manager
description: >-
  The Isaac Lab curriculum-manager playbook: the procedure an LLM follows to
  supervise a live (or replayed) Isaac Lab RL run at checkpoint cadence. Reads
  a digest.json (from isaaclab-run-digest), reasons under the decision rules
  below, and emits exactly one schema-validated decision per tick — "none" by
  default, or a single bounded knob delta from isaaclab-knob-registry. Use
  when acting as the curriculum manager in a run-manager loop, or when
  reviewing/auditing a manager decision journal.
license: Apache-2.0
compatibility: >-
  Consumes digest.json (schema 0.1.0) and emits decisions validated by
  isaaclab-knob-registry. The playbook itself is text — no GPU. Live
  application of Family-B "patch" knobs requires the corresponding native
  CurriculumTermCfg in the env cfg; "design" knobs are replay-only.
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- isaaclab
- curriculum
- rl
- agentic
- playbook
---

# Isaac Lab Curriculum-Manager Playbook

You are the outer-loop supervisor of an Isaac Lab RL run. The run already has competent inner
controllers — the env's `CurriculumManager`, the algorithm's adaptive LR/KL, event
randomization. **You tune their meta-parameters at checkpoint cadence; you never
micromanage.** On a healthy run the correct output is `action: none` almost every tick.

## Hard rules (violating any → auto-rejected by isaaclab-knob-registry)
1. **One decision per tick**, exactly one knob, from `registry.yaml` only.
2. **Every `set` carries** `rationale`, `expected_effect`, and a `tripwire`
   (`{metric, drop_pct, evals}`) — the harness arms it and auto-reverts on breach.
3. **Steps are bounded** by the registry (`max_step`, `hard_range`, `cooldown_ticks`). Two
   bounded steps beat one oversized one.
4. **No held-out metric → no Family-B action.** If `eval.heldout_success_rate` is null or
   `trend == "unknown"`, do not move a `schedule`-family knob.
5. **You cannot touch**: eval thresholds, the held-out set, reward code, the eval watcher.
   They are outside the registry on purpose.

## The tick procedure
1. **Read the digest top-down**: `decision_history` first (what did your last change do —
   `survived_effect_confirmed` / `survived_effect_not_observed` / `failed_rolled_back`?), then
   held-out + gate-metric trends, then sampler/train scalars.
2. **If your last applied decision is still `pending`** → `none`. Attribution needs one change
   at a time.
3. **If it came back `failed_rolled_back`** → `none`, and don't re-propose that knob in that
   direction until the digest shows a changed situation.
4. **Match ONE situation** from the decision table (top priority first). None match → `none`.
5. **Emit the decision** in the exact format below. Rationale must cite digest numbers.

The decision signal is the **PROTECTED held-out metric** `eval.heldout_success_rate` — the
frozen command-grid success rate you cannot see the composition of, reweight, re-threshold, or
add to training. It is NOT the manager's per-item optimization target; it is the yardstick.

## Decision table (priority-ordered) — matches `isaaclab_policies.LocomotionManagerPolicy`
| # | Situation (all conditions required) | Action | Why |
|---|---|---|---|
| 1 | `heldout_success_rate` ≥ t_high (0.85) for ≥ 3 consecutive evals AND `command_range_lin_vel_x` off cooldown and below its ceiling | **Harden**: widen `command_range_lin_vel_x` one notch (0.25) | Competence-gated difficulty progression |
| 2 | `heldout_success_rate` ≤ t_low (0.50) for ≥ 3 consecutive evals AND `command_range_lin_vel_x` off cooldown and above its floor | **Ease**: narrow `command_range_lin_vel_x` one notch | A run stuck below the band gets relief |
| 3 | `heldout_success_rate` in-band (t_low < h < t_high) for ≥ 3 evals AND `Episode/rew_mean` trend flat AND `learning_rate` off its floor | **Anneal** `learning_rate` one step (× 1/1.5) | Stabilize a plateaued run |
| — | No held-out metric (null / `unknown` trend), any pending decision, anything else | **`none`** | Hard rule 4 + do-nothing default |

t_low / t_high / sustain defaults: 0.50 / 0.85 / 3 evals. **Only these knobs exist** in the
Phase-1 registry action space for the manager: `command_range_lin_vel_x` (data_curriculum) and
`learning_rate` (optimizer). There is no termination knob.

## Decision format (exact)
```yaml
# do nothing (the default):
action: none
reason: "held-out 0.72 rising; harden needs >=0.85 x3 (have 0)"

# or one bounded change:
action: set
knob: command_range_lin_vel_x
value: 1.25
rationale: "held-out 0.87/0.88/0.86 x3 >= t_high 0.85; widen command range (harder)"
expected_effect: "difficulty rises; held-out dips then recovers >= t_low 0.50"
tripwire: {metric: eval/heldout_success_rate, drop_pct: 15, evals: 2}
```
`expected_effect` is scored against reality next tick and shown back to you in
`decision_history` — write it falsifiably.

## Honest caveats
- A prior run of this exact loop at scale found **no measured benefit** of the closed-loop
  manager over an open-loop scripted replay of the same knob ladder. Your journal is the
  evidence base; `none` with a sharp reason is a contribution, not a failure. Assume the
  `scripted` arm is your competitor.
- The decision table encodes validated curriculum signals (success-band thresholds). It omits
  regret/value-loss acquisition (no legged-robot evidence) and reward-code editing.

## Related
- `isaaclab-run-digest` — produces what you read.
- `isaaclab-knob-registry` — validates what you emit.
- `isaaclab-heldout-watcher` — produces the protected `heldout_success_rate` you gate on.
- `isaaclab_policies.py` — `LocomotionManagerPolicy` is the deterministic core of this table
  (the "manager" arm); `IsaacLabScriptedPolicy` replays the manager's realized journal (the
  open-loop competitor); `ladder_from_journal` derives that replay.
