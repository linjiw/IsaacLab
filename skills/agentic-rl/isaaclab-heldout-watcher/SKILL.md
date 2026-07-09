---
name: isaaclab-heldout-watcher
description: >-
  Produce the held-out success metric the Isaac Lab curriculum manager cannot
  see, reweight, or filter — a frozen, salted-hash split of tasks/terrains/
  commands/goals evaluated at fixed thresholds. Implements "protect the
  manager's metric from the manager" as a structural (process/manifest)
  boundary, not a prompt rule. Use when defining the held-out evaluation set
  for a curriculum-RL campaign, or when computing/auditing the protected
  heldout_success_rate that gates Family-B (schedule) knob changes.
license: Apache-2.0
compatibility: >-
  Split selection, manifest, and integrity checking are pure Python 3.9+
  stdlib (CPU-testable). Reuses the vendored runmanager as the consumer.
  Running the held-out eval itself requires the isaaclab-job-adapter + a GPU.
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- isaaclab
- curriculum
- rl
- agentic
- evaluation
---

# Isaac Lab Held-out Watcher (the protected metric)

Produces `heldout_success_rate` — the metric the manager **cannot** see in its knob-decision
inputs, reweight, or filter. Protection is **structural**: the held-out subset is chosen by a
deterministic salted hash split, recorded in an integrity-checked manifest, and excluded from
every knob's reach. Family-B (`schedule`) knob decisions are machine-gated on this metric
(no held-out metric → no action).

## What "held-out" means in Isaac Lab
The frozen subset depends on the task family:
- **Locomotion** — a fixed set of command velocities / terrain tiles never used to drive
  curriculum decisions.
- **Manipulation / dexterous** — a fixed set of goal poses / object configs.
- **Factory / contact-rich** — a fixed set of assembly configurations.

Eval runs at **fixed relaxed thresholds** deliberately absent from `registry.yaml`, so the
manager cannot tighten/loosen the very bar it is judged against.

## Guarantees (mirrors the SONIC watcher)
- `select_holdout(keys, fraction, salt)` — deterministic salted SHA-256 split, **stable under
  library growth** (adding tasks never reshuffles the existing split).
- `write_manifest` / `load_manifest` — integrity digest; load **fails on tamper**.
- `curriculum_keys()` — the training-side allowlist (everything the manager may use), which
  **excludes** the held-out keys.
- The record emitter **refuses** to emit a metric if the eval touched motions/tasks outside
  the manifest — a mis-wired eval can't silently feed a wrong number.

## Files (bundled)
- `holdout.py` — split / manifest / integrity / record emission + CLI (adapt SONIC `holdout.py`).
- `test_holdout.py` — CPU tests (split determinism, growth stability, integrity, refusal).

## Quick start
```bash
cd skills/agentic-rl/isaaclab-heldout-watcher
python3 holdout.py select --keys tasks.txt --fraction 0.2 --salt <salt> --out manifest.json
python3 -m pytest test_holdout.py -q
```

## Related
- `runmanager/core/loop.py` — consumes the held-out record via the `heldout_hook`; requires
  an explicit standard-eval pin alongside it (structural guard in `LoopConfig.__post_init__`).
- `runmanager/core/registry.py` — enforces the held-out gate for `schedule`-family knobs.
- `isaaclab-curriculum-manager` — hard rule: no held-out metric, no Family-B action.
