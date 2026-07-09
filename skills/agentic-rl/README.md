# agentic-rl — LLM-guided curriculum RL for Isaac Lab

An agentic-skill system where an outer-loop policy (a rule-based band-stepper OR an LLM)
supervises a live Isaac Lab RL run at checkpoint cadence, applying bounded, held-out-gated
curriculum-knob changes on top of a vendored, engine-agnostic control core. It is built to
**honestly test** whether adaptive curriculum beats a scripted replay and a fixed control — the
prior SONIC system reached a null result, so the durable asset here is the guardrail /
observability / verification scaffold, not an adaptivity claim.

Start with **`DESIGN.md`** (architecture + plan), **`EVAL_FRAMEWORK.md`** (what we measure and
what we may claim), and **`GAPS.md`** (what's left, ranked).

## Layout
```
runmanager/                vendored engine-agnostic core (loop, registry, tripwire, digest,
                           equivalence gate, journal) — NVIDIA Apache-2.0; 138-test suite.
isaaclab-job-adapter/      the 6-method EngineAdapter for skrl/rl_games (docker-exec into the
                           isaac-lab-base container) + eval_rollout.py (headless metrics eval).
isaaclab-knob-registry/    the action space (registry.yaml) + validator.
isaaclab-run-digest/       TB-scalar -> digest-record normalization (tb_reader.py).
isaaclab-heldout-watcher/  the PROTECTED metric: frozen salted-hash command grid + integrity.
isaaclab-curriculum-manager/  the LLM decision playbook (SKILL.md).
isaaclab-task-catalog/     per-task gate metric + knob subset + framework.
isaaclab-curriculum-rl/    design/scaffold guide + Isaac Lab seams.
isaaclab_policies.py       the arms: LocomotionManagerPolicy (rule), IsaacLabScriptedPolicy
                           (replay), LLMPolicy (claude -p), ladder_from_journal.
run_curriculum.py          host-driven arm driver (control/manager/scripted/llm).
run_pilot.sh               3-arm ON-vs-OFF orchestrator (manager -> derive ladder -> scripted -> control).
run_chaos_probe.sh + compute_tau.py    measure the Isaac Lab noise band -> equivalence tau.
run_sensitivity_probe.sh   prove the knob moves the metric > tau (gate G7) before a real verdict.
verify_gate.py             Phase-1 validity gates on the journals.
analyze_pilot.py           ON-vs-OFF analysis (AUC + iters-to-threshold + equivalence gate).
```

## Architecture (where things run)
Isaac Lab lives ONLY inside the `isaac-lab-base` Docker container. The **driver runs on the HOST**
(pure Python — vendored core + PyYAML — and it needs the host `docker` to exec into the
container); the adapter builds `docker exec` commands for all GPU work. `/workspace` is
bind-mounted whole, so the container's `/workspace/isaaclab/logs` == the host's. Run the driver
with the host test venv (`/tmp/rmc-venv/bin/python`), NEVER `/isaac-sim/python.sh`.

## Run the CPU test suite (no GPU)
```bash
python3 -m venv /tmp/rmc-venv && /tmp/rmc-venv/bin/python -m pip install -q pytest pyyaml
cd skills/agentic-rl
/tmp/rmc-venv/bin/python -m pytest runmanager/tests \
  isaaclab-job-adapter/test_isaaclab_adapter.py isaaclab-task-catalog/test_catalog.py \
  isaaclab-heldout-watcher/test_holdout.py isaaclab-run-digest/test_tb_reader.py \
  test_smoke_loop.py -q          # expect: 199 passed, 3 skipped
```

## Reproduce the Phase-1 pilot (needs the isaac-lab-base container + GPU)
```bash
C=isaac-lab-base
# 1) warm-start a competent-enough policy (~400 iters); note its final checkpoint
docker exec $C bash -c "cd /workspace/isaaclab && /isaac-sim/python.sh \
  scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 \
  --headless --max_iterations=400 --num_envs=1024 --seed=42 \
  agent.agent.experiment.experiment_name=warmstart"
WARM=/workspace/isaaclab/logs/skrl/anymal_c_rough/<ts>_ppo_torch_warmstart/checkpoints/agent_9600.pt

# 2) build the PROTECTED held-out manifest (in-envelope command grid)
/tmp/rmc-venv/bin/python isaaclab-heldout-watcher/holdout.py make-manifest \
  --salt phase1 --out _phase1_manifest.json

# 3) run the 3-arm pilot (band calibrated to the achievable ceiling; see EVAL_FRAMEWORK)
AGENTIC_RL_PY=/tmp/rmc-venv/bin/python \
  bash run_pilot.sh "$WARM" "$(pwd)/_phase1_manifest.json" 6 150 1024 0.20 0.33 2

# 4) validity gates + ON-vs-OFF analysis
/tmp/rmc-venv/bin/python verify_gate.py _runs/pilot
/tmp/rmc-venv/bin/python analyze_pilot.py _runs/pilot
```

## Before any adaptivity CLAIM (EVAL_FRAMEWORK §2, §3a)
```bash
# measure the Isaac Lab noise band -> tau
bash run_chaos_probe.sh "$WARM" "$(pwd)/_phase1_manifest.json" 3 4 150 1024
/tmp/rmc-venv/bin/python compute_tau.py _runs/chaos
# prove the lever moves the metric > tau (else re-pair lever/metric)
bash run_sensitivity_probe.sh "$WARM" "$(pwd)/_phase1_manifest.json" 1.0 3.0 4 150 1024
```

## Status (2026-07-09)
Phase-0 built + reviewed twice; Phase-1 pilot running (skrl + locomotion-velocity only; rl_games
fails loud). The comparative adaptivity verdict awaits a measured Isaac Lab tau + a multi-seed
run, scored on sample efficiency (AUC). See GAPS.md for the ranked remaining work.
