#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Phase-1 PILOT orchestrator: run the full 3-arm ON-vs-OFF comparison from a
# shared warm-start checkpoint, with the protected held-out metric.
#
# Order (SONIC methodology — the scripted ladder MUST come from a real manager
# run, so the manager arm runs FIRST):
#   1. manager  arm  -> journal_manager.json   (derives the scripted ladder)
#   2. scripted arm  (replays the manager's realized journal, same warm-start)
#   3. control  arm  (fixed defaults, same warm-start)
# All three: same seed, same segment schedule, same held-out manifest, same
# warm-start. Held-out success_rate is the reported headline for all arms.
#
# ARCHITECTURE: the driver runs ON THE HOST (it is pure Python — vendored core
# + PyYAML — and needs the host's `docker` binary to exec into the container).
# The adapter shells into isaac-lab-base for all GPU work. The whole /workspace
# is bind-mounted (/workspace -> /workspace), so the container's
# /workspace/isaaclab/logs IS the host's — journals land in the shared tree.
# Usage: run_pilot.sh <warm_ckpt> <heldout_manifest> [segments] [iters] [num_envs] [t_low] [t_high]
set -euo pipefail

WARM="${1:?warm-start checkpoint path (container-visible) required}"
MANIFEST="${2:?held-out manifest path required}"
SEGMENTS="${3:-4}"
ITERS="${4:-150}"
NUM_ENVS="${5:-2048}"
T_LOW="${6:-0.50}"     # calibrate to the task's achievable held-out ceiling
T_HIGH="${7:-0.85}"
SUSTAIN="${8:-3}"       # lower (e.g. 2) so the manager acts within a short run
TASK="Isaac-Velocity-Rough-Anymal-C-v0"
DRIVER="/workspace/IsaacLab/skills/agentic-rl/run_curriculum.py"
# journals go to a HOST-WRITABLE dir (the driver runs as the host user;
# /workspace/isaaclab/logs is root-owned — writable only by the adapter's
# docker-exec'd, root-running train/eval). This dir is under the bind-mounted
# /workspace so it's container-visible too. Override with AGENTIC_RL_OUT.
OUT="${AGENTIC_RL_OUT:-/workspace/IsaacLab/skills/agentic-rl/_runs/pilot}"
# host Python with the vendored core + PyYAML (NOT the in-container kit python)
PY="${AGENTIC_RL_PY:-/tmp/rmc-venv/bin/python}"

mkdir -p "$OUT"
common=(--task "$TASK" --framework skrl --segments "$SEGMENTS" --iterations "$ITERS"
        --num-envs "$NUM_ENVS" --eval-envs 128 --seed 42
        --warm-start "$WARM" --heldout-manifest "$MANIFEST"
        --t-low "$T_LOW" --t-high "$T_HIGH" --sustain "$SUSTAIN")

echo "=== [1/3] MANAGER arm ==="
$PY "$DRIVER" --arm manager "${common[@]}" --journal-out "$OUT/journal_manager.json"

echo "=== [2/3] SCRIPTED arm (ladder derived from the manager journal) ==="
$PY "$DRIVER" --arm scripted "${common[@]}" \
    --scripted-from-journal "$OUT/journal_manager.json" \
    --journal-out "$OUT/journal_scripted.json"

echo "=== [3/3] CONTROL arm ==="
$PY "$DRIVER" --arm control "${common[@]}" --journal-out "$OUT/journal_control.json"

echo "=== pilot complete; journals in $OUT ==="
ls -la "$OUT"/journal_*.json
