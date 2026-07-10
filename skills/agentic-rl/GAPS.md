# Design & Implementation Gap Audit (2026-07-09, mid-v2-pilot)

What is left in the current design/implementation, ranked by whether it blocks a *trustworthy
result* vs. a *broader system*. Grounded in the real state (Phase-0 built + reviewed twice,
Phase-1 pilot running; three integration bugs and one experiment-validity trap fixed).

Legend: **[BLOCK]** blocks a trustworthy Phase-1 verdict · **[SCALE]** needed to generalize
beyond one task/framework · **[POLISH]** quality/robustness.

---

## A0. THE central design question (surfaced by the v2 real data)

- **[BLOCK] The held-out metric is convergence-dominated, which may make the ON-vs-OFF
  comparison structurally null regardless of the lever.** Evidence from real runs:
  training duration moves held-out enormously (60 iter → 0.016, 400 iter → 0.32), while
  widening the command range moved it ~0 (manager arm: 0.34→0.39→0.37, flat). Two consequences:
  1. The **command-range lever is wrong for this metric** (it changes the training distribution,
     evaluated on a fixed in-envelope grid → ~no effect). Use a lever the metric responds to:
     the **optimizer family** (lr/entropy/KL) if segments are under-converged (they are —
     held-out is still rising), or **reward-weight** knobs that directly trade off tracking.
  2. **Deeper:** if the metric is dominated by iteration count, and every arm trains the SAME
     total iterations by design, all arms converge to nearly the same held-out → a null is almost
     guaranteed *by construction*, not by "adaptivity doesn't help." **The comparison only has
     power if the lever can change the metric MORE than the shared iteration budget does.**
  **Resolution path (must settle before the multi-seed [BLOCK] run):** (a) run the
  lever-sensitivity probe (`run_sensitivity_probe.sh`) for the CANDIDATE lever; (b) if
  command-range fails G7, switch the manager+scripted to an optimizer or reward-weight lever and
  re-probe; (c) consider a tighter held-out tolerance (0.15 not 0.25) for more dynamic range; (d)
  if NO lever beats the iteration-budget effect, the honest finding is "at this task/budget,
  curriculum knobs don't move the protected metric beyond training-duration" — a valid negative
  result, reported as such. This is the top question for the third review.

- **[BLOCK] LEVER–METRIC ALIGNMENT (the root fix, from the v2 control arm).** Real data:
  control (range fixed at 1.0) scored held-out 0.397 at s3 vs manager (widened to 1.25) 0.389 —
  i.e. **widening the training command range made the FIXED in-envelope held-out slightly WORSE**
  (−0.008, 1 seed, within noise but the SIGN is mechanistically sensible: widening spends policy
  capacity on fast commands the narrow held-out never tests → negative transfer). The lever and
  metric are **misaligned**: we widen TRAINING but evaluate on a NARROW fixed grid, so widening
  can only hurt. **Principled fix for the next experiment:** make the held-out grid span the
  command envelope the manager can REACH (e.g. graduated up to |vx|≈2, keeping headroom — ±3 is
  physically unachievable and floors the metric at 0). Then "widen training" is *rewarded* by
  better generalization to a wider held-out set, and the curriculum question becomes answerable:
  "does adaptively deciding WHEN to widen beat a fixed widening schedule, measured on a held-out
  set that DEMANDS the wider competence?" This is the preferred resolution over (b)/(c) above
  because it aligns the curriculum's ACTION with the metric's DEMAND rather than switching levers.

## A. Evaluation & scientific validity

- **[BLOCK] No measured Isaac Lab noise band.** The equivalence gate uses SONIC's τ
  (E5B_CHAOS_FLOOR_MEAN, A10G/256-env). `equivalence.py` itself says re-measure when
  engine/env-count/horizon change — all differ. **Action:** 3-seed (+1 ε-perturbed) chaos probe
  of the control config → `sigma_noise` → `calibrate_tau`. Until done, NO manager-vs-scripted
  claim is defensible. (EVAL_FRAMEWORK §2.)
- **[BLOCK] Single seed.** The pilot is n=1. The comparative verdict needs ≥3 seeds × 3 arms.
- **[POLISH] Eval determinism unverified.** `eval_rollout` uses a fixed seed, but I haven't
  confirmed two evals of the SAME checkpoint reproduce the same held-out score (SONIC's eval was
  byte-identical; ours is a fresh rollout — needs a 2-run check). If eval itself is noisy, that
  noise must fold into τ.
- **[POLISH] Held-out eval length (300 steps) vs. episode length.** If episodes reset mid-eval,
  the "fraction of steps within tol" mixes transients with steady-state. Should confirm the
  window covers ≥1 full command-hold period per env.

## B. The manager / decision logic

- **[SCALE] Action space is effectively 1-D in practice.** Both arms only moved
  `command_range_lin_vel_x`. `learning_rate` (lr-anneal) only fires on an in-band *plateau with
  flat reward* — which didn't occur. The reward-weight schedule knobs are `status:design`
  (held-out-gated, and we proved the gate path but never exercise them). A richer test needs the
  manager to actually use ≥2 knobs, or we scope the claim to "command-range curriculum" (honest).
- **[SCALE] No LLM in the loop yet.** The "manager" is `LocomotionManagerPolicy` — a deterministic
  band-stepper (the validated core of the SONIC playbook rows). The actual LLM policy (shell out
  to `claude -p` with the digest + playbook, parse fenced JSON, fail-safe to `none`) is designed
  (SKILL.md) but not implemented. The deterministic policy is the right FIRST comparator (it *is*
  the "adaptivity" the scripted arm ablates), but "LLM-guided" in the goal needs the LLM policy
  eventually — and a 4th arm (LLM vs rule-based manager vs scripted vs control).
- **[POLISH] Thrash / ceiling behavior untested.** If held-out sits just above t_high, the
  manager ramps command range to the hard_range ceiling (3.0) — is that desirable, or should
  there be a "stop hardening past competence" rule? The tripwire catches regressions but not
  "hardened too far, metric plateaus." Consider a max-hardening-per-run cap.

## C. Adapter / engine coverage

- **[SCALE] rl_games unimplemented** (eval_rollout raises; train experiment-name needs `+`
  prefix; run_curriculum fails loud). Only skrl works. Fine for Phase-1 (skrl-only), but the
  registry/catalog/DESIGN advertise rl_games.
- **[SCALE] One task family.** Only locomotion velocity (Anymal-C). The gate metric
  (`tracking_within_tol`) is locomotion-specific; manipulation/dexterous/factory need their own
  gate-metric derivation in `eval_rollout` + `tasks.yaml` entries. `isaaclab-task-catalog` is
  built for this but only 3 tasks are declared and only 1 is exercised.
- **[POLISH] Checkpoint resume fidelity assumed, not measured.** skrl `load` restores
  `checkpoint_modules` (incl. optimizer/preprocessor), so segmenting *should* ≈ one continuous
  run. Not empirically verified (train 2 segments vs 1 double-length run, compare via the
  equivalence gate). If resume loses state, each segment is a mini-restart and the "curriculum
  over one run" framing weakens.
- **[POLISH] Eval `num_envs` < grid coverage warning exists but no hard guard.** If
  `num_envs < len(heldout_keys)` some commands are never forced; we warn + set
  `heldout_coverage_partial` but still emit a metric. For a real run, assert full coverage.

## D. Held-out watcher / protection

- **[POLISH] Salt is on the CLI / manifest, readable by anyone on the box.** SONIC's threat model
  was "protect the metric from the manager PROCESS." Our manager is a Python object in the same
  process as the driver — the "process boundary" is weaker (the policy *could* read the manifest
  if coded to). It doesn't (LocomotionManagerPolicy only reads the digest), but this is
  convention, not enforcement. For an LLM policy that's fine (it only sees the digest text). Worth
  a note that protection is by construction, not sandbox.
- **[POLISH] Grid is fixed at manifest-creation.** No check that the training command range never
  grows to *include* held-out commands (the manager can widen to [-3,3], which spans the whole
  in-envelope grid). Widening the training range to cover held-out commands is arguably fine
  (it's generalization, and held-out is never used to *drive* decisions per-command), but it
  means "held-out" is held-out from DECISIONS, not from TRAINING DATA after enough hardening.
  This is a subtle semantic worth documenting: our protection is "manager can't see/steer the
  metric," NOT "policy never trains on these commands."

## E. Orchestration / operational

- **[POLISH] No resume for a crashed pilot.** If the box dies mid-run, `run_pilot.sh` restarts
  from the manager arm. The run-manager loop has no checkpoint/resume of its own journal. For
  long multi-seed runs, add resume-from-last-completed-arm.
- **[POLISH] GPU sharing is manual.** We downsize envs by hand to be a good neighbor. No
  automated back-off if another job appears. Fine for now; note it.
- **[POLISH] Logs split across host (`_runs/`) and container (`/workspace/isaaclab/logs`).**
  Journals in `_runs/`, per-segment train/eval logs + checkpoints in the root-owned container
  tree. Analysis reads both. Works, but a single collected artifact dir per run would be cleaner.

## F. Documentation / reproducibility

- **[POLISH] The three integration bugs + the vacuous-experiment finding aren't in DESIGN.md.**
  They're in memory + this file. DESIGN should get a "Phase-1 lessons" section (host-vs-container,
  YAML tags, root-owned logs, operating-point calibration, do-nothing-manager-on-healthy-run).
- **[POLISH] No single "reproduce the pilot" runbook.** The exact commands (make warm-start →
  make manifest → run_pilot → verify_gate → analyze_pilot) are scattered. Add a README.

---

## Priority for reaching a TRUSTWORTHY Phase-1 verdict (the immediate goal)
1. Finish v2 pilot → `verify_gate.py` + `analyze_pilot.py` (validates machinery + shows the ON ladder). **[in progress]**
2. Third adversarial review over design+impl+the real experiment. Loop fixes.
3. **[BLOCK]** Chaos-floor probe (3 seeds control) → measure Isaac Lab τ.
4. **[BLOCK]** 3-seed × 3-arm real experiment → apply the equivalence gate → pre-registered verdict.
5. Then, and only then, a defensible statement about adaptivity vs scripted.

## Priority for the BROADER goal ("all robotics tasks, LLM-guided")
6. Implement the actual LLM policy (`claude -p`) as a 4th arm.
7. Second task family (a manipulation task) with its own gate metric.
8. rl_games support.
