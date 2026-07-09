#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Lever-sensitivity probe (EVAL_FRAMEWORK §3a / gate G7): prove the manager's
# knob CAN move the primary metric by more than the noise band tau — otherwise
# an ON-vs-OFF null is uninformative (can't tell "adaptivity doesn't help" from
# "this knob doesn't move this metric for anyone").
#
# Runs the CONTROL arm (no proposals) twice, pinning the command-range knob at
# its NARROW floor vs its WIDE ceiling for the whole run, from the same warm
# start, and compares the final held-out. If |wide - narrow| <= tau the
# lever+metric pair is INSENSITIVE and the experiment must be re-paired.
#
# Host-driven. Usage:
#   run_sensitivity_probe.sh <warm_ckpt> <manifest> [lo] [hi] [segments] [iters] [num_envs]
set -euo pipefail

WARM="${1:?warm-start checkpoint required}"
MANIFEST="${2:?held-out manifest required}"
LO="${3:-1.0}"          # narrow command range (half-width)
HI="${4:-3.0}"          # wide command range (knob hard_range ceiling)
SEGMENTS="${5:-4}"
ITERS="${6:-150}"
NUM_ENVS="${7:-1024}"
TASK="Isaac-Velocity-Rough-Anymal-C-v0"
DRIVER="/workspace/IsaacLab/skills/agentic-rl/run_curriculum.py"
OUT="${AGENTIC_RL_OUT:-/workspace/IsaacLab/skills/agentic-rl/_runs/sensitivity}"
PY="${AGENTIC_RL_PY:-/tmp/rmc-venv/bin/python}"

mkdir -p "$OUT"
for LEVEL in "$LO" "$HI"; do
  echo "=== sensitivity probe: control arm, command_range=$LEVEL ==="
  $PY "$DRIVER" --arm control --task "$TASK" --framework skrl \
      --segments "$SEGMENTS" --iterations "$ITERS" --num-envs "$NUM_ENVS" \
      --eval-envs 128 --seed 42 --warm-start "$WARM" \
      --heldout-manifest "$MANIFEST" --t-low 0.20 --t-high 0.33 --sustain 2 \
      --base-knob "command_range_lin_vel_x=$LEVEL" \
      --journal-out "$OUT/journal_range${LEVEL}.json"
done

echo "=== sensitivity result ==="
$PY - "$OUT" "$LO" "$HI" <<'PY'
import json, sys, os
d, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3]
def final_ho(f):
    j = json.load(open(f))
    v = [(e.get("heldout") or {}).get("heldout_success_rate") for e in j if "segment" in e]
    v = [x for x in v if x is not None]
    return v[-1] if v else None
lo_v = final_ho(os.path.join(d, f"journal_range{lo}.json"))
hi_v = final_ho(os.path.join(d, f"journal_range{hi}.json"))
delta = abs(hi_v - lo_v) if (lo_v is not None and hi_v is not None) else None
print(f"  narrow(range={lo}) final held-out = {lo_v}")
print(f"  wide(range={hi})   final held-out = {hi_v}")
print(f"  |delta| = {delta}")
print("  Compare |delta| to the MEASURED tau (compute_tau.py). If |delta| <= tau,")
print("  the command-range lever does NOT move held-out beyond noise → re-pair")
print("  the lever/metric before running the ON-vs-OFF experiment (G7).")
PY
