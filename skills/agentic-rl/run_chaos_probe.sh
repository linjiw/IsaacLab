#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Noise-band probe (EVAL_FRAMEWORK §2.1): measure the Isaac Lab/skrl held-out
# noise band, so the equivalence gate's tau is calibrated for THIS engine (not
# borrowed from SONIC/A10G). Without this, no manager-vs-scripted claim is
# defensible.
#
# IMPORTANT (corrected after the v2 pilot): skrl training here is BIT-
# DETERMINISTIC — same seed + config reproduces held-out to ~9 sig figs (manager
# and scripted arms were bit-identical). So the fp-"chaos" floor SONIC measured
# (same-seed re-runs diverging on nondeterministic kernels) is ~0 here, and the
# relevant noise for a CROSS-SEED ON-vs-OFF verdict is the SEED-to-seed variance
# of held-out. This probe therefore varies the SEED (control arm, fixed
# defaults) across N runs; sigma over the per-seed held-out is sigma_noise, and
# calibrate_tau turns it into the equivalence tolerance. (Named "chaos" for
# continuity with the SONIC E5B methodology; here it measures seed variance.)
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
