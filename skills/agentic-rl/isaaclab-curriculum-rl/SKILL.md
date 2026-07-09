---
name: isaaclab-curriculum-rl
description: >-
  Design/scaffold guide for LLM-guided curriculum reinforcement learning in
  Isaac Lab: how an LLM outer loop supervises a live RL run at checkpoint
  cadence and applies bounded curriculum/hyperparameter changes on top of
  Isaac Lab's native CurriculumManager. Explains the write surfaces (reward
  weights, terrain levels, command ranges, domain randomization, optimizer
  meta-params) and the guardrail architecture. Use when planning or reasoning
  about adding LLM-guided curriculum learning or automatic difficulty
  scheduling to Isaac Lab RL training, or when wiring the agentic-rl skills
  together.
license: Apache-2.0
compatibility: >-
  Design/reference skill — pure text, no GPU. The concepts map onto Isaac Lab
  manager-based envs (ManagerBasedRLEnv) and the rl_games / skrl training
  scripts under scripts/reinforcement_learning/. The runnable pieces live in
  the sibling agentic-rl skills and the vendored runmanager/ core.
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read
tags:
- isaaclab
- curriculum
- rl
- agentic
- design
---

# Isaac Lab Curriculum-RL (design & scaffold)

This is the conceptual parent of the `agentic-rl` skill family. It describes how an
**LLM outer loop** supervises a live Isaac Lab RL run — reading progress at checkpoint
cadence and emitting bounded, schema-validated "knob" changes — riding on top of Isaac Lab's
**native curriculum machinery** rather than replacing it.

Read `../DESIGN.md` for the full plan and phasing. Read `references/seams.md` for exact
Isaac Lab file:line integration points.

## The one finding that shapes this work

A prior project (`groot-tao-agentic-rl-curriculum`) built and ran this exact system against
NVIDIA's SONIC humanoid and reached a **documented null result**: an open-loop *scripted
replay* of the manager's own knob ladder matched-or-beat the closed-loop LLM manager in both
seeds. The durable asset is the **scaffold** (guardrails, held-out protection, tripwire/
rollback, config-drift verification, equivalence gate), not the adaptivity claim. Therefore:
**always run a `scripted` arm as the bar the `manager` arm must clear**, and treat LLM
adaptivity as a hypothesis under test, not an assumption.

## Two-tier control

- **Outer loop (this system, LLM):** acts only at checkpoint cadence (every K iterations /
  one "segment"). Reads a `digest.json`, emits one bounded knob delta or `none`.
- **Inner controllers (Isaac Lab, always-on):** the env's `CurriculumManager` terms, the RL
  algorithm's own adaptive LR/KL controllers, event randomization. The LLM tunes the
  **meta-parameters** of these fast controllers — it never micromanages per-step.

## Write surfaces (knob families) — all native to Isaac Lab

| Family | Examples | Isaac Lab seam |
|---|---|---|
| **A — data/task curriculum** | command velocity ranges, object-pose ranges, terrain difficulty | `env.commands.*.ranges.*`; terrain generator; `modify_term_cfg` |
| **B — competence-gated schedules** | reward-weight ramps, termination thresholds, DR push scale | native `CurriculumTermCfg`: `modify_reward_weight`, `modify_term_cfg`, `modify_env_param`, `terrain_levels_vel` |
| **C — optimizer meta-params** | learning rate, entropy coef, desired KL | agent cfg (`agent.params.config.*` for rl_games, `agent.*` for skrl) |

Everything is applied as a **Hydra CLI override** at segment (re)launch — no file edits.

## The guardrail architecture (what the LLM is NOT trusted with)

The driver (`runmanager/core/loop.py`) owns all guardrails; the LLM only proposes:
1. **Knob whitelist + static validator** (`isaaclab-knob-registry`) — hard range, max step,
   cooldown, one-change-at-a-time.
2. **Held-out metric** (`isaaclab-heldout-watcher`) — a frozen eval subset the manager can't
   see, reweight, or filter.
3. **Tripwire + rollback** — auto-revert a knob if its guard metric drops beyond threshold.
4. **Config-drift verification** — reconcile believed knob values against the run's resolved
   `params/{env,agent}.yaml` before any decision.
5. **Decision journal** — every decision with rationale, expected effect, and scored outcome.

## The skill family

- `isaaclab-job-adapter` — the `EngineAdapter` (launch/eval/rollback/parse) for rl_games/skrl.
- `isaaclab-knob-registry` — the action space (`registry.yaml`) + validator.
- `isaaclab-run-digest` — logs → trend-annotated `digest.json`.
- `isaaclab-heldout-watcher` — the protected metric.
- `isaaclab-curriculum-manager` — the LLM playbook.
- `isaaclab-task-catalog` — per-task gate metric + knob subset + framework.
- `runmanager/` — the vendored engine-agnostic core (loop, registry, tripwire, digest, journal, equivalence).

## Related
- `../DESIGN.md` — full design plan and phasing.
- `references/seams.md` — Isaac Lab integration points (file:line).
