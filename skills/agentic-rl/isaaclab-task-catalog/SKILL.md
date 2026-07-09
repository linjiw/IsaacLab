---
name: isaaclab-task-catalog
description: >-
  Enumerate and describe Isaac Lab's ~179 registered robotics tasks for
  curriculum-RL, and declare per-task the gate metric, applicable knob subset,
  and RL framework. Wraps scripts/environments/list_envs.py and adds a
  tasks.yaml that resolves per-task-family differences — critically the gate
  metric, since locomotion logs tracking error (not a success rate) while
  manipulation/dexterous/factory log success more directly. Use when choosing
  which task to run a curriculum-RL campaign on, or when wiring a task's gate
  metric and knobs into the run-manager loop.
license: Apache-2.0
compatibility: >-
  tasks.yaml + the catalog CLI are pure Python 3.9+ stdlib + PyYAML
  (CPU-testable). Enumerating the live gym registry via list_envs.py requires
  an Isaac Lab + Isaac Sim install.
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- isaaclab
- curriculum
- rl
- agentic
- catalog
---

# Isaac Lab Task Catalog

The goal is LLM-guided curriculum RL for **all** Isaac Lab robotics tasks — not one robot.
Task families differ in ways the run-manager loop must know about, so this skill declares them
in `tasks.yaml` and wraps the built-in enumerator.

## Why this skill exists: the gate-metric problem
The run-manager loop needs a scalar **gate metric** per segment (the tripwire/promotion
signal). But manager-based **locomotion** tasks log tracking **error** (`Metrics/base_velocity/*`),
not a success rate — so the gate metric must be *derived* (e.g. fraction of held-out commands
tracked within tolerance). Manipulation / dexterous / factory tasks log success more directly.
`tasks.yaml` records, per task, how to compute its gate metric.

## What tasks.yaml declares (per task)
```yaml
Isaac-Velocity-Rough-Anymal-C-v0:
  family: locomotion
  workflow: manager_based          # manager_based | direct
  frameworks: [rsl_rl, rl_games, skrl]
  play_task: Isaac-Velocity-Rough-Anymal-C-Play-v0
  gate_metric:
    kind: tracking_within_tol       # success_rate | tracking_within_tol | progress_rate
    source: Metrics/base_velocity/error_vel_xy
    tolerance: 0.25
  native_curriculum: [terrain_levels]     # CurriculumTermCfg terms present in the env cfg
  knob_subset: [command_range_lin_vel_x, terrain_difficulty, learning_rate, entropy_coef]
```

## Task families (from the Isaac Lab catalog)
Velocity/locomotion, manipulation (Lift, Reach, Stack, Open, PickPlace), dexterous (Repose,
Shadow-Hand), factory/contact-rich (Factory, Forge, AutoMate), navigation, classic (Cartpole,
Ant, Humanoid, Quadcopter). ~179 unique `Isaac-*` IDs.

## Files (bundled)
- `tasks.yaml` — the per-task declarations (seeded for the Phase-1 target family, grown over time).
- `catalog.py` — wrap `scripts/environments/list_envs.py`; join live registry against tasks.yaml;
  report which tasks are "curriculum-ready" (have a gate metric + knob subset).
- `test_catalog.py` — CPU tests over a fixture registry dump + tasks.yaml.

## Quick start
```bash
cd skills/agentic-rl/isaaclab-task-catalog
# enumerate the live catalog (needs Isaac Sim)
./isaaclab.sh -p scripts/environments/list_envs.py --keyword Velocity
# which declared tasks are curriculum-ready?
python3 catalog.py ready --tasks tasks.yaml
```

## Related
- `scripts/environments/list_envs.py` — the built-in registry enumerator.
- `isaaclab-job-adapter` — reads `gate_metric` to compute the eval scoreboard; `frameworks`/`play_task` for launch.
- `isaaclab-knob-registry` — `knob_subset` selects which registry knobs apply to a task.
- `isaaclab-heldout-watcher` — the held-out split is drawn from a task's command/goal/terrain space.
