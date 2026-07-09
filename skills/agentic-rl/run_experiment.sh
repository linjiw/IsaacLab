#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The REAL multi-seed ON-vs-OFF experiment (EVAL_FRAMEWORK §0a, §3).
#
# Fixes the degeneracy the v2 pilot exposed: because skrl training is
# bit-deterministic, a same-seed scripted arm that replays that seed's own
# manager ladder is IDENTICAL to the manager (manager - scripted ≡ 0 by
# construction). The ablation is only meaningful CROSS-SEED:
#
#   1. run the manager on a DONOR seed -> extract its fixed ladder
#   2. for each EVAL seed: run manager (adapts to its OWN digest), scripted
#      (replays the DONOR's fixed ladder, blind to this seed), control, llm
#   3. compare manager vs scripted on AUC across the eval seeds
#
# Prereq: measure tau first (run_chaos_probe.sh + compute_tau.py) and prove
# lever sensitivity (run_sensitivity_probe.sh). Host-driven.
#
# Usage: run_experiment.sh <warm_ckpt> <manifest> <donor_seed> "<eval_seeds>" \
#                          [segments] [iters] [num_envs] [t_low] [t_high] [sustain]
set -euo pipefail

WARM="${1:?warm-start checkpoint required}"
MANIFEST="${2:?held-out manifest required}"
DONOR_SEED="${3:-42}"
EVAL_SEEDS="${4:-1234 7 99}"
SEGMENTS="${5:-6}"; ITERS="${6:-150}"; NUM_ENVS="${7:-1024}"
T_LOW="${8:-0.20}"; T_HIGH="${9:-0.33}"; SUSTAIN="${10:-2}"
TASK="Isaac-Velocity-Rough-Anymal-C-v0"
DRIVER="/workspace/IsaacLab/skills/agentic-rl/run_curriculum.py"
OUT="${AGENTIC_RL_OUT:-/workspace/IsaacLab/skills/agentic-rl/_runs/experiment}"
PY="${AGENTIC_RL_PY:-/tmp/rmc-venv/bin/python}"
mkdir -p "$OUT"

common=(--task "$TASK" --framework skrl --segments "$SEGMENTS" --iterations "$ITERS"
        --num-envs "$NUM_ENVS" --eval-envs 128 --warm-start "$WARM"
        --heldout-manifest "$MANIFEST" --t-low "$T_LOW" --t-high "$T_HIGH" --sustain "$SUSTAIN")

echo "=== [donor] manager on seed $DONOR_SEED -> fixed ladder ==="
$PY "$DRIVER" --arm manager --seed "$DONOR_SEED" "${common[@]}" \
    --journal-out "$OUT/donor_manager_seed${DONOR_SEED}.json"
DONOR_LADDER="$OUT/donor_manager_seed${DONOR_SEED}.json"

for SEED in $EVAL_SEEDS; do
  echo "=== [eval seed $SEED] control ==="
  $PY "$DRIVER" --arm control --seed "$SEED" "${common[@]}" \
      --journal-out "$OUT/control_seed${SEED}.json"
  echo "=== [eval seed $SEED] manager (adapts to its OWN digest) ==="
  $PY "$DRIVER" --arm manager --seed "$SEED" "${common[@]}" \
      --journal-out "$OUT/manager_seed${SEED}.json"
  echo "=== [eval seed $SEED] scripted (replays DONOR seed $DONOR_SEED ladder, blind) ==="
  $PY "$DRIVER" --arm scripted --seed "$SEED" "${common[@]}" \
      --scripted-from-journal "$DONOR_LADDER" \
      --journal-out "$OUT/scripted_seed${SEED}.json"
done

echo "=== experiment complete; journals in $OUT ==="
echo "Analyze cross-seed with: analyze_experiment.py $OUT (manager vs scripted AUC across eval seeds)"
ls "$OUT"/*.json
