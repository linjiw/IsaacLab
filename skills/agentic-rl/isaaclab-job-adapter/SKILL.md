---
name: isaaclab-job-adapter
description: >-
  Launch/observe/eval/rollback lifecycle for Isaac Lab RL run-segments — the
  curriculum manager's hands. Implements the 6-method EngineAdapter for the
  rl_games and skrl training scripts: builds the train.py invocation with
  knob Hydra overrides and checkpoint resume, snapshots checkpoints for
  rollback, parses TensorBoard scalars into digest train streams, and runs a
  pinned play.py eval pass returning a gate metric. Use when launching a
  managed Isaac Lab run-segment, converting training logs into digest inputs,
  running an eval scoreboard pass, or rolling back a bad knob change.
license: Apache-2.0
compatibility: >-
  Command building, log parsing, and rollback bookkeeping are pure Python
  3.9+ stdlib + PyYAML (CPU-testable offline). launch/wait/eval require a
  working Isaac Lab + Isaac Sim install and a GPU (./isaaclab.sh -p
  scripts/reinforcement_learning/{rl_games,skrl}/{train,play}.py).
metadata:
  author: Isaac Lab Project
  version: "0.1.0"
allowed-tools: Read Bash Write
tags:
- isaaclab
- curriculum
- rl
- agentic
- infra
---

# Isaac Lab Job Adapter (the manager's hands)

Implements `runmanager/core/protocols.py::EngineAdapter` for Isaac Lab. This is the **only
heavily engine-coupled** skill in the family; the vendored core imports no engine.

## Where it runs (host, not container)

Isaac Lab lives only inside the `isaac-lab-base` Docker container, so the adapter builds
`docker exec isaac-lab-base bash -c "..."` commands. **The driver (`run_curriculum.py`) and
this adapter therefore run ON THE HOST** — the host has the `docker` binary and the pure
Python the driver needs (vendored core + PyYAML); it does NOT need Isaac Sim. Running the
driver *inside* the container fails (`docker` not found there). The whole `/workspace` tree is
bind-mounted (`/workspace -> /workspace`), so the container's `/workspace/isaaclab/logs` is the
same files the host sees — journals, checkpoints, and per-segment logs live in that shared
tree. `run_curriculum.py` preflights `shutil.which("docker")` and fails loud if run in the
wrong place.

## The segment model

The manager's unit of change is a **run-segment**: a knob-constant stretch of training.
Apply a decision = end segment N, snapshot its latest checkpoint, launch segment N+1 from
that checkpoint (`--checkpoint`) with the new knob Hydra overrides. Rollback = relaunch from
segment N's **input** checkpoint with segment N-1's knob values.

```
seg1 (knobs A) ──ckpt──▶ snapshot_seg1.pt ──▶ seg2 (knobs B, resume=snapshot_seg1)
                              │ tripwire fires ▼
                     seg2_rollback (knobs A, resume=snapshot_seg1)   ← pre-change state
```

## The 6 EngineAdapter methods → Isaac Lab

| Method | Isaac Lab implementation |
|---|---|
| `launch_segment` | spawn `./isaaclab.sh -p scripts/reinforcement_learning/<fw>/train.py --task <ID> --headless --checkpoint <ckpt_in> --max_iterations <seg_len> <hydra knob overrides>` (background process). |
| `wait` | poll the process; on finish set `status`, `experiment_dir` (from the "Exact experiment name" stdout line), and `snapshot` (copy latest checkpoint to `snapshot_<name>.pt`). |
| `parse_segment` | read TensorBoard event files under `experiment_dir`; emit `train[]` records keyed by injected `train_scalar_keys`; count tracebacks from stderr. |
| `eval_segment` | `play.py --task <ID>-Play-v0 --num_envs N --checkpoint <snapshot>` at pinned settings; derive the task's **gate metric**. |
| `resolved_config_text` | read the run's `params/env.yaml` + `params/agent.yaml`. |
| `knob_to_config_path` | the registry `config_path` map, framework-prefixed. |

## Framework deltas (rl_games vs skrl) — verified

| Concern | rl_games | skrl |
|---|---|---|
| Log dir | `logs/rl_games/<config.name>/<ts>/` | `logs/skrl/<directory>/<ts>_<algo>_<framework>/` |
| Checkpoints | `nn/*.pth` | `checkpoints/*.pt` |
| Knob prefix | `agent.params.config.*`, `agent.params.seed` | `agent.*`, `agent.seed` |
| Max-iters | `--max_iterations` → `params.config.max_epochs` | indirect: `trainer.timesteps = max_iterations × agent.rollouts` |
| Resume | `--checkpoint` → `load_path` + `load_checkpoint=True` | `--checkpoint` → `runner.agent.load(path)` |
| Resolved config | dumps `params/{env,agent}.yaml` ✓ | dumps `params/{env,agent}.yaml` ✓ |

**Pin-the-scoreboard (the M1 lesson):** eval at fixed settings the manager can't touch. Isaac
Lab `-Play-v0` variants already reduce num_envs and disable noise/DR, which gets us most of
the way; the adapter still asserts no knob override leaks into the eval invocation.

## Files (to implement in this skill)
- `isaaclab_adapter.py` — `IsaacLabAdapter` (the 6 methods) + `KNOB_TO_HYDRA` (raises on
  unmapped knobs — never invents config paths) + command builders + TB parser + CLI/dry-run.
- `test_isaaclab_adapter.py` — CPU tests for command building + parsing against real log/TB
  excerpts in `testdata/`.

## Quick start
```bash
cd skills/agentic-rl/isaaclab-job-adapter
python3 -m pytest test_isaaclab_adapter.py -q
# dry-run: print the exact launch command for a knob override
python3 isaaclab_adapter.py command --framework skrl --task Isaac-Velocity-Rough-Anymal-C-v0 \
  --name seg1 --iterations 50 --knob learning_rate=0.0008
```

## Related
- `runmanager/core/protocols.py` — the `EngineAdapter` protocol this satisfies.
- `runmanager/adapters/mock.py` — reference adapter pattern to mirror.
- `isaaclab-run-digest` — consumes the streams this produces.
- `isaaclab-knob-registry` — validates decisions before they become segments.
- `isaaclab-task-catalog` — supplies the per-task gate metric + framework.
