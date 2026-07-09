#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Chaos-floor probe (EVAL_FRAMEWORK §2.1): measure the Isaac Lab/skrl run-to-run
# noise band on the held-out metric, so the equivalence gate's tau is calibrated
# for THIS engine (not borrowed from SONIC/A10G). Without this, no manager-vs-
# scripted claim is defensible.
#
# Runs the SAME control config N times varying ONLY the seed (control arm =
# fixed defaults, no manager, no scripted). The std of the final
# heldout_success_rate across runs is sigma_noise. calibrate_tau turns that into
# the equivalence tolerance.
#
# ARCHITECTURE: host-driven (like run_pilot.sh). Usage:
#   run_chaos_probe.sh <warm_ckpt> <heldout_manifest> [n_seeds] [segments] [iters] [num_envs]
set -euo pipefail

WARM="${1:?warm-start checkpoint required}"
MANIFEST="${2:?held-out manifest required}"
N_SEEDS="${3:-3}"
SEGMENTS="${4:-4}"
ITERS="${5:-150}"
NUM_ENVS="${6:-1024}"
TASK="Isaac-Velocity-Rough-Anymal-C-v0"
DRIVER="/workspace/IsaacLab/skills/agentic-rl/run_curriculum.py"
OUT="${AGENTIC_RL_OUT:-/workspace/IsaacLab/skills/agentic-rl/_runs/chaos}"
PY="${AGENTIC_RL_PY:-/tmp/rmc-venv/bin/python}"

mkdir -p "$OUT"
SEEDS=(42 1234 7)   # first N used
for i in $(seq 0 $((N_SEEDS - 1))); do
  SEED="${SEEDS[$i]}"
  echo "=== chaos probe: control arm, seed $SEED ($((i+1))/$N_SEEDS) ==="
  $PY "$DRIVER" --arm control --task "$TASK" --framework skrl \
      --segments "$SEGMENTS" --iterations "$ITERS" --num-envs "$NUM_ENVS" \
      --eval-envs 128 --seed "$SEED" --warm-start "$WARM" \
      --heldout-manifest "$MANIFEST" --t-low 0.20 --t-high 0.33 --sustain 2 \
      --journal-out "$OUT/journal_seed${SEED}.json"
done

echo "=== chaos probe complete; computing sigma_noise + tau ==="
$PY "$(dirname "$0")/compute_tau.py" "$OUT"
