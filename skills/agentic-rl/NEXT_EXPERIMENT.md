# The aligned cross-seed experiment (spec)

**Status:** ready to run once the third review clears it + the pre-flight probes pass.
This turns every lesson from the v2 pilot into one turnkey, trustworthy experiment.

## Why this design (what the v2 pilot taught)
1. **Same-seed scripted == manager by construction** (skrl is bit-deterministic; proven: the two
   arms were held-out-identical to 9 sig figs). ⇒ the ablation MUST be **cross-seed**
   (`run_experiment.sh`): scripted replays a FIXED ladder from a *donor* seed.
2. **Convergence domination + equal budget ⇒ null by construction on FINAL held-out.** ⇒ score
   on **sample efficiency (AUC / iters-to-threshold)**, not final (`analyze_pilot`/`analyze_experiment`).
3. **Lever–metric MISALIGNMENT.** command-range widening + a NARROW fixed held-out grid gave
   *negative* transfer (control beat manager by +0.008→+0.019 as widening grew, directional over
   s3–s4). ⇒ **align the metric to the lever**: a WIDE held-out grid that DEMANDS the competence
   widening builds.
4. **No measured noise band.** τ is still SONIC's placeholder. ⇒ measure it (seed variance, since
   training is bit-deterministic) before any verdict.

## The hypothesis (aligned)
> Adaptively widening the training command range *when the policy is competent* (manager) reaches
> high held-out success on a WIDE command grid **faster** (higher AUC) than (a) a fixed widening
> schedule transplanted from another seed (scripted) and (b) never widening (control) — by more
> than the measured cross-seed noise band τ.

## Pre-flight gates (ALL must pass before the multi-seed run — else re-design, don't run)
1. **Wide-grid discriminates + isn't floored.** Build the wide manifest
   (`holdout.py make-manifest --lin-vel-x="-2,-1,-0.5,0.5,1,2" --ang-vel-z="-2,-1,1,2" --salt <s>`).
   Eval the warm-start on it: held-out must be a finite fraction with per-condition spread > 0
   (not 0 everywhere — |vx|=2 must be *reachable with curriculum*, unlike |vx|=3).
2. **Lever sensitivity G7** (`run_sensitivity_probe.sh` with the WIDE manifest): control@range=1.0
   vs control@range=3.0 must differ by **> measured τ** on wide-grid held-out — i.e. widening the
   training range *does* move the wide-grid metric (expected POSITIVE now that they're aligned).
   If it doesn't clear τ, switch the lever to optimizer knobs and keep the narrow grid.
3. **Measured τ** (`run_chaos_probe.sh` 3 seeds + `compute_tau.py`) on the SAME (lever, grid, budget).

## The run (once gates pass)
```bash
WIDE=_runs/wide_manifest.json   # from pre-flight gate 1
# donor seed 42 -> ladder; eval seeds 1234, 7, 99
AGENTIC_RL_PY=/tmp/rmc-venv/bin/python \
  bash run_experiment.sh "$WARM" "$(pwd)/$WIDE" 42 "1234 7 99" 6 150 1024 0.20 0.33 2
/tmp/rmc-venv/bin/python analyze_experiment.py _runs/experiment --tau-json _runs/chaos/tau.json
```

## Verdict (pre-registered, EVAL_FRAMEWORK §3)
`analyze_experiment.py` applies the decision rule to **manager − scripted (AUC)** across the eval
seeds vs measured τ: within-τ ⇒ *no measured adaptivity benefit* (the SONIC-consistent null, still
a valid finding); > τ same-sign ⇒ adaptivity helps; < −τ ⇒ hurts. Report n, τ, per-condition
worst/spread, and the manager − control delta (the knob effect, now expected positive on the wide grid).

## Cost note
4 arms × (donor + 3 eval seeds) ≈ manager×4 + scripted×3 + control×3 + llm×3 segments-worth of
training + 2 evals each, on a shared GPU. Sequence it; the machinery + analysis are all in place.
The LLM arm can be added as a 4th arm in run_experiment.sh once the rule-based result is in.
