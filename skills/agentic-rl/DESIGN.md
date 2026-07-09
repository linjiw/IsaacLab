# LLM-Guided Curriculum RL Skills for IsaacLab — Design Plan

**Status:** design (pre-implementation) · **Date:** 2026-07-08 · **Target repo:** `/workspace/IsaacLab`
**Location:** `skills/agentic-rl/` (self-contained; vendors an engine-agnostic core)
**First backend:** `rl_games` / `skrl` (broad task coverage) · **Stance:** scaffold-first, honest ON-vs-OFF test

---

## 0. Executive summary & the finding that shapes everything

We are building a set of agentic skills that let an LLM *supervise* a live reinforcement-learning
run in IsaacLab — reading training progress at checkpoint cadence and emitting bounded, schema-validated
curriculum/hyperparameter changes ("knobs") — with guardrails, held-out evaluation, tripwire rollback,
and a decision journal.

**Critical prior-art finding.** The `groot-tao-agentic-rl-curriculum` fork already built and *ran* this
exact system against NVIDIA's SONIC humanoid, and reached a **documented null result**: in a two-seed
ON-vs-OFF experiment, an *open-loop scripted replay* of the manager's own knob ladder **matched or beat**
the closed-loop LLM manager, and a single bistable motion accounted for 51–96% of the apparent gains.
Their own conclusion: *"the adaptive curriculum manager is dead at this scale."*

What survived as durable value was **not** the LLM-adaptivity claim — it was the *scaffold*: the
guardrails, observability (`digest.json`), held-out-metric protection, tripwire/rollback, config-drift
verification, and the equivalence/verification harness. They also already did the hard decoupling work:
`experiments/run-manager-core/` is a backend-agnostic refactor built around a clean 6-method
`EngineAdapter` protocol.

**Therefore this plan:**
1. **Vendor the proven engine-agnostic core** into IsaacLab (loop, registry validator, tripwire, digest,
   equivalence gate, journal). It carries over essentially unchanged.
2. **Implement one `EngineAdapter` for IsaacLab** (`rl_games`/`skrl`) — the only heavily-coupled piece.
3. **Wire knobs onto IsaacLab's *native* curriculum machinery** (`modify_reward_weight`, `modify_term_cfg`,
   `modify_env_param`, `terrain_levels_vel`) instead of the custom patch mechanism SONIC needed.
4. **Design the experiment to honestly test adaptivity** — control / manager / scripted arms from day one.
   Do **not** assume the LLM helps. Reproduce the null-result methodology (per-task decomposition,
   pinned scoreboard, artifact-not-exit-code verification, chaos-floor equivalence).
5. Build toward the proposed **v6 three-tier** design (analytic tier-0 controllers + LLM supervising
   meta-params/gates + analyst/actuator split) only *after* the honest test, to escape the failure mode
   where "the action space collapsed to a schedule."

---

## 1. Background: what IsaacLab gives us

### 1.1 RL training entry points
IsaacLab ships `train.py`/`play.py` for `rsl_rl`, `rl_games`, `skrl`, `sb3`, plus `ray/` for distributed
tuning, under `scripts/reinforcement_learning/<framework>/`. The launcher is `./isaaclab.sh -p <script>`
(or `python <script>`). ~179 unique `Isaac-*` task IDs across manipulation, locomotion, dexterous,
factory/contact-rich, navigation, and classic-control families. Enumerate them with
`scripts/environments/list_envs.py [--keyword ...]`.

### 1.2 Manager-based envs & native curriculum (the key coupling point)
Manager-based envs (`isaaclab.envs.ManagerBasedRLEnv`) assemble config from managers:
`RewardManager`, `ObservationManager`, `TerminationManager`, `CommandManager`, `EventManager`,
and **`CurriculumManager`**. Curriculum terms are declared in the env cfg:

```python
# source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py
@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

# source/isaaclab_tasks/.../manipulation/lift/lift_env_cfg.py
action_rate = CurrTerm(func=mdp.modify_reward_weight,
    params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000})
```

Native curriculum primitives live in `source/isaaclab/isaaclab/envs/mdp/curriculums.py`:
- **`modify_reward_weight`** — step-scheduled reward reweighting.
- **`modify_term_cfg`** — mutate any term cfg via a dotted address (e.g. `commands.object_pose.ranges.pos_x`).
- **`modify_env_param`** — generic runtime mutation of any env attribute (DR ranges, obs ranges…).
- **`terrain_levels_vel`** (locomotion overlay) — promote/demote terrain difficulty by walked distance.

**This is a gift.** SONIC's "Family-B schedule" knobs needed a bespoke patch mechanism. In IsaacLab they
map *directly* onto these native `CurriculumTermCfg` terms. The groot
`stage2-termination-curriculum/config/threshold_tighten.yaml` is literally already an IsaacLab
`CurriculumTermCfg` with `values`/`num_steps`.

### 1.3 Config overrides = Hydra
`register_task_to_hydra` + `@hydra_task_config` (`source/isaaclab_tasks/.../utils/hydra.py`) expose every
leaf as a CLI override — `env.<a>.<b>=value` / `agent.<x>=value`. So **knobs are CLI strings; no file edits**.
Callables settable as `module:attr`; `null`=None; tuples given as quoted lists (`"[-2.0, 2.0]"`).
Legacy `--num_envs/--seed/--max_iterations` take precedence over Hydra.

### 1.4 Logging / outputs (how the LLM observes)
Runs write to `logs/<framework>/<experiment>/<timestamp>[_run]/`. Every run dumps
`params/env.yaml` + `params/agent.yaml` (fully resolved configs → **config-drift verification for free**).
Manager log dicts land in `env.extras["log"]`: `Episode_Reward/<term>`, `Metrics/<command>/<metric>`,
`Curriculum/<term>`, `Episode_Termination/<term>`. TensorBoard event files are the programmatic progress
source; wandb/neptune optional.

### 1.5 Framework-specific deltas (verified from the train.py scripts)
These matter because we chose `rl_games`/`skrl`, not the rsl_rl the bulk of research assumed:

| Concern | `rl_games` | `skrl` |
|---|---|---|
| Log dir | `logs/rl_games/<config.name>/<ts>/` | `logs/skrl/<directory>/<ts>_<algo>_<framework>/` |
| Checkpoints | `nn/*.pth` | `checkpoints/*.pt` |
| Agent cfg | YAML, nested under `agent.params.*` | YAML, under `agent.*` |
| Seed override | `agent.params.seed` | `agent.seed` |
| Max-iters override | `agent.params.config.max_epochs` | **indirect:** `trainer.timesteps = max_iterations × agent.rollouts` |
| Resume | `--checkpoint` → sets `params.load_path`, `load_checkpoint=True` | `--checkpoint` → `runner.agent.load(path)` |
| Algo selection | (single) | `--algorithm PPO\|MAPPO\|IPPO\|AMP\|...` + `--ml_framework torch\|jax` |
| Resolved config dump | `params/{env,agent}.yaml` ✓ | `params/{env,agent}.yaml` ✓ |

**Adapter implication:** `knob_to_config_path()` must emit framework-correct prefixes
(`agent.params.config.*` for rl_games vs `agent.*` for skrl), `launch_segment` must handle skrl's
indirect max-iterations math, and `parse_segment`/checkpoint-snapshot must know the per-framework
checkpoint subdir (`nn/*.pth` vs `checkpoints/*.pt`).

---

## 2. The engine-agnostic core (vendored, ~unchanged)

Source: `groot-tao-agentic-rl-curriculum/experiments/run-manager-core/`. We vendor `core/` + `adapters/`
+ `tests/` into `skills/agentic-rl/runmanager/`. It is pure Python (stdlib + PyYAML), CPU-testable.

| Module | Role | Change for IsaacLab |
|---|---|---|
| `core/loop.py` (`RunManager`) | the digest→decide→validate→apply→watch tick loop | none (all engine specifics injected via `LoopConfig`) |
| `core/registry.py` (`KnobRegistry`) | whitelist + static validator: `hard_range`, `max_step`, `cooldown`, pending-gate, required tripwire fields, `verify_against_config` | new `registry.yaml` content only |
| `core/tripwire.py` (`TripwireWatch`) | rollback when guard metric drops >X% for N evals; abs-drop floor; stale-eval refusal | none |
| `core/digest.py` | logs → trend-annotated `digest.json`; injectable `train_scalar_keys`, term prefixes | inject IsaacLab scalar keys |
| `core/equivalence.py` (`EquivalenceGate`) | "same run?" gate; chaos-floor τ≈3.9e-2 window-mean; bit-identity short-circuit | none |
| `core/journal.py` | decision journal with provenance (`digest_hash`, `applied_at_iter`) | none |
| `core/protocols.py` | `Segment`, `ParsedSegment`, `Decision`, `Tripwire` dataclasses; `EngineAdapter`, `Policy`, `ObservingPolicy` protocols | none |
| `adapters/mock.py` | reference adapters + policies (`TrainSideBandPolicy`, `ScriptedPolicy`, `V4_MANAGER_LADDER`) + `MOCK_REGISTRY_SPEC` | reference for our IsaacLab adapter/policies |

### 2.1 The `EngineAdapter` seam (verbatim signatures — exactly 6 methods)
```python
class EngineAdapter(Protocol):
    def launch_segment(self, name, iterations, knobs, checkpoint_in=None) -> Segment: ...
    def wait(self, seg, poll_s=20, timeout_s=3600) -> Segment: ...   # sets status/experiment_dir/snapshot
    def parse_segment(self, seg) -> ParsedSegment: ...               # -> .train[], .sampler[], .tracebacks
    def eval_segment(self, seg, it, num_envs=64, extra_overrides=None,
                     out_suffix="_eval", raw=False) -> Dict: ...      # eval-only pass on snapshot
    def resolved_config_text(self, seg) -> Optional[str]: ...        # for config-drift verify (optional)
    def knob_to_config_path(self) -> Dict[str, str]: ...             # knob -> dotted resolved-config path
```
`Policy` is `propose(digest, state, registry) -> dict`; `ObservingPolicy` adds `observe(digest)`.

### 2.2 The tick loop (invariants we inherit)
Per tick, `RunManager.run()`: disk gate (≥8GB free) → launch → wait → parse → traceback check →
renumber iters + accumulate streams + snapshot checkpoint → **config-drift verify (before any decision)**
→ standard eval (pinned thresholds) → **held-out protected-metric eval** → checkpoint purge → build digest
→ `policy.observe(digest)` → **tripwire watch judgment** (hold/watching/rollback/survived) →
control-arm branch → **pending gate** (one change under watch at a time) → `policy.propose(...)` →
if `set`: `validate_decision` → capture baseline (refuse if none) → arm tripwire → apply knob →
mark pending → journal provenance → append journal entry.

---

## 3. The skill set (mirrors the six sonic-* skills + one new)

Following TAO skill-bank conventions: each skill = a directory with `SKILL.md` (YAML frontmatter:
`name`, `description` with literal trigger phrases, `license: Apache-2.0`, `metadata.author/version`,
`allowed-tools`, `tags`), optional `references/skill_info.yaml`, and bundled CPU-testable scripts.
Prefix: `isaaclab-`.

| Skill | groot analog | Purpose | Bundles |
|---|---|---|---|
| **`isaaclab-curriculum-rl`** | `tao-curriculum-rl` | Design/scaffold ancestor. Documents IsaacLab's native curriculum seams and how the LLM outer loop rides on them. | `references/seams.md` (exact file:line cites), no code |
| **`isaaclab-job-adapter`** | `sonic-job-adapter` | **The backend interface.** Implements `EngineAdapter` for rl_games/skrl: launch/resume `train.py`, snapshot/rollback checkpoints, `play.py` eval, log parsing, `knob_to_config_path`. | `isaaclab_adapter.py` + tests + `testdata/` (real log excerpts) |
| **`isaaclab-knob-registry`** | `sonic-knob-registry` | IsaacLab `registry.yaml` (the action space) + reuses vendored validator. | `registry.yaml`, thin CLI |
| **`isaaclab-run-digest`** | `sonic-run-digest` | Parse IsaacLab TensorBoard scalars / `extras["log"]` → `digest.json`. | reuses `core/digest.py`; `tb_reader.py` |
| **`isaaclab-heldout-watcher`** | `sonic-heldout-watcher` | Frozen, salted-hash-split held-out eval subset (tasks/terrains/commands/goals) the manager can't see or touch. | `holdout.py` + manifest |
| **`isaaclab-curriculum-manager`** | `sonic-curriculum-manager` | The LLM playbook (text): read digest, emit one schema-validated decision. | `SKILL.md` only |
| **`isaaclab-task-catalog`** *(new)* | — | Enumerate/filter the 179 tasks via `list_envs.py`; declare each task's **gate metric** + knob subset + framework. | `catalog.py`, `tasks.yaml` |

The new `isaaclab-task-catalog` skill exists because the goal is **all** robotics tasks, not one robot.
It resolves per-task-family differences — critically the **gate metric**: manager-based *locomotion* logs
tracking *error* (`Metrics/base_velocity/*`), not a success rate, so `eval_segment` must derive a scalar
(e.g. fraction of held-out commands tracked within tolerance). Manipulation/dexterous/factory log success
more directly.

### 3.1 Orchestration graph
```
                 ┌──────────────────────────────────────────────┐
                 │  RunManager (vendored core/loop.py) — DRIVER   │  owns guardrails
                 │  loop: digest→decide→validate→apply→watch      │
                 └─┬──────────┬───────────┬───────────┬──────────┘
 isaaclab-job-adapter        │           │           └── isaaclab-knob-registry
 (EngineAdapter: launch/     │           │               (validate_decision, verify_against_config)
  eval/rollback/parse)       │           │
        │                    │      isaaclab-curriculum-manager (LLM playbook)
   raw logs ──► isaaclab-run-digest ───►  reads digest.json, emits decision
   (TB scalars)              ▲
 isaaclab-heldout-watcher ───┘   isaaclab-task-catalog feeds task id + gate metric + knob subset
 (protected metric)              isaaclab-curriculum-rl = design ancestor (docs only)
```

---

## 4. Knob registry for IsaacLab (the action space)

Every knob = a Hydra override string; families mirror groot's A/B/C. Illustrative
(`termination_threshold`-style entries map onto IsaacLab **native** `CurriculumTermCfg`):

```yaml
meta:
  global_rules: {max_changes_per_tick: 1, default_action: none}
knobs:
  # Family A — data/task curriculum
  command_range_lin_vel_x:
    family: data_curriculum
    config_path: env.commands.base_velocity.ranges.lin_vel_x   # rl_games/skrl share env.* prefix
    type: float; default: 1.0; hard_range: [0.5, 3.0]
    max_step: {kind: notch, step: 0.25}; cooldown_ticks: 2; status: available
  terrain_difficulty:
    family: data_curriculum
    config_path: env.scene.terrain.terrain_generator.difficulty_range
    type: choice; status: patch    # rides terrain_levels_vel

  # Family B — competence-gated schedules (NATIVE curriculum terms)
  reward_weight_action_rate:
    family: schedule
    config_path: env.curriculum.action_rate.params.weight   # modify_reward_weight
    type: float; default: -0.1; hard_range: [-1.0, 0.0]
    max_step: {kind: multiplicative, factor: 1.5}; cooldown_ticks: 3; status: patch
  push_velocity_dr:
    family: schedule
    config_path: env.events.push_robot.params.velocity_range
    type: float; status: patch

  # Family C — optimizer meta-params (framework-specific prefix!)
  learning_rate:
    family: optimizer
    config_path: agent.params.config.learning_rate   # rl_games; skrl uses agent.agent.learning_rate
    type: float; default: 1.0e-3; hard_range: [1.0e-4, 3.0e-3]
    max_step: {kind: multiplicative, factor: 1.5}; cooldown_ticks: 4; status: available
  entropy_coef:
    family: optimizer
    config_path: agent.params.config.entropy_coef
    type: float; default: 0.005; hard_range: [0.0, 0.02]
    max_step: {kind: additive, step: 0.002}; cooldown_ticks: 4; status: available
```

Each knob keeps `hard_range`, `max_step` (multiplicative/additive/notch), `cooldown_ticks`, `status`
(available/patch/design). Family-B "schedule" knobs require a non-null held-out metric to fire
(machine-enforced by the validator's `heldout_gated_families`). The `config_path` prefix is
**framework-dependent** — the registry is authored per framework (or the adapter's `knob_to_config_path`
rewrites the prefix).

---

## 5. The IsaacLab EngineAdapter (concrete mapping)

The "segment" model transfers directly: a segment = a knob-constant training stretch; applying a decision
= end segment, snapshot the checkpoint, relaunch `--checkpoint <snapshot>` with new Hydra overrides;
rollback = relaunch from the segment's *input* checkpoint with the previous knob value.

| Method | rl_games / skrl implementation |
|---|---|
| `launch_segment` | build & spawn `./isaaclab.sh -p scripts/reinforcement_learning/<fw>/train.py --task <ID> --headless --checkpoint <ckpt_in> --max_iterations <seg_len> <hydra knob overrides>`; run via `Bash(run_in_background)`. For skrl, translate `seg_len → trainer.timesteps = seg_len × rollouts`. |
| `wait` | poll the background process; on finish set `status`, `experiment_dir` (parse the "Exact experiment name" stdout line), and `snapshot` (copy latest `nn/*.pth` or `checkpoints/*.pt` to `snapshot_<name>.pt`). |
| `parse_segment` | read TensorBoard event files under `experiment_dir`; emit `train[]` records with injected scalar keys (`rewards/*`, `episode_lengths/*`, `Episode_Reward/*`, `Curriculum/*`, `Metrics/*`, `Episode_Termination/*`); count tracebacks from stderr. |
| `eval_segment` | `./isaaclab.sh -p scripts/reinforcement_learning/<fw>/play.py --task <ID>-Play-v0 --num_envs N --checkpoint <snapshot>` at **fixed pinned settings** (Play variants already disable noise/DR — the M1 "pin-the-scoreboard" lesson is half-free); derive the task's gate metric (success rate, or tracking-within-tolerance fraction for locomotion). |
| `resolved_config_text` | read the run's `params/env.yaml` + `params/agent.yaml` → config-drift verification for free. |
| `knob_to_config_path` | the registry `config_path` map, framework-prefixed. |

### 5.1 Reference policies (mirror `adapters/mock.py`)
- **Control arm** — proposals disabled (fixed defaults). Baseline.
- **Manager arm** — an `ObservingPolicy` that reads the digest and proposes bounded deltas
  (the IsaacLab analog of `TrainSideBandPolicy`, targeting the binding termination axis / gate metric).
  For the *live LLM* variant: shell out to `claude -p` with the playbook + digest, parse fenced JSON,
  fail-safe to `{"action": "none"}` (mirrors groot's `LLMPolicy`).
- **Scripted arm** — `ScriptedPolicy`: open-loop replay of the manager's own knob ladder,
  reads the digest *only* to pick the tripwire metric. **The arm the manager must beat to justify itself.**

---

## 6. Honest experiment design (the point of the whole thing)

Reproduce the groot ON-vs-OFF methodology so a positive result would be trustworthy:
- **Three arms:** control / manager / scripted, same seeds, same segment schedule.
- **Per-task decomposition:** report per-task (and per-motion/per-terrain) deltas, not just aggregate —
  a single bistable task dominated groot's apparent gains.
- **Pin the scoreboard:** eval at fixed thresholds the manager can't touch; assert no manager override
  leaks into eval (their M1 finding). IsaacLab `-Play-v0` variants help here.
- **Artifact-not-exit-code verification:** confirm the eval produced a real metric artifact; refuse to
  reuse a stale baseline on eval failure (their M3 fix).
- **Chaos-floor equivalence:** use `EquivalenceGate` (window-mean τ≈3.9e-2) to decide if two runs of the
  same config are "the same run" — pointwise per-iter gating is dead (~28% floor).
- **Protect the metric from the manager:** held-out subset is salted-hash-split, integrity-checked, and
  excluded from every knob's reach.
- **Report honestly, incl. null results.** n=2 seeds ⇒ no claim beyond noise. `log()` any dropped coverage.

---

## 7. Phasing

- **Phase 0 — Scaffold port.** Vendor `runmanager/`; write `IsaacLabAdapter` + a mock; get the tick loop
  green against a short `Isaac-Cartpole-v0` run (rl_games or skrl). Port groot's CPU tests
  (~4,350 lines) for parsing/validation. Deliverable: green `pytest`, one real end-to-end short run.
- **Phase 1 — One task family, real loop.** Target a locomotion velocity task (richest native curriculum:
  terrain levels + reward-weight ramps). Author `registry.yaml`, the digest scalar map, and a held-out
  command/terrain subset. Run all three arms. Deliverable: three-arm run + digests + journals.
- **Phase 2 — Honest experiment.** Full ON-vs-OFF with per-task decomposition, pinned scoreboard,
  chaos-floor equivalence, artifact verification. Deliverable: a results report that would survive the
  same adversarial review that killed the SONIC claim.
- **Phase 3 — Generalize across families** via `isaaclab-task-catalog`: manipulation lift, dexterous
  repose, factory contact-rich — each declaring its gate metric + knob subset + framework.
- **Phase 4 — v6 three-tier** (only if warranted): analytic tier-0 controllers + LLM supervising
  meta-params/stage gates + analyst/actuator split, to escape "action space collapsed to a schedule."

---

## 8. Open risks / decisions to revisit
- **Gate metric for locomotion** is derived, not native — must validate it correlates with policy quality.
- **rl_games/skrl checkpoint cadence & resume fidelity** — confirm `--checkpoint` resume preserves
  optimizer state well enough that segmenting ≈ one continuous run (feed the equivalence gate).
- **Registry `config_path` is framework-specific** — decide whether to ship per-framework registries or a
  single registry + adapter-side prefix rewriting.
- **Live-LLM cost/latency** — the checkpoint-cadence outer loop keeps LLM calls rare; still budget them.
- **Expect the null result to reproduce.** The scaffold is the deliverable; adaptivity is a hypothesis
  under test, not an assumption.

---

## Appendix: key source references
**IsaacLab**
- Training: `scripts/reinforcement_learning/{rl_games,skrl}/train.py`, `.../play.py`
- Env cfg w/ curriculum: `source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py`
- Curriculum primitives: `source/isaaclab/isaaclab/envs/mdp/curriculums.py`; manager
  `source/isaaclab/isaaclab/managers/curriculum_manager.py`
- Hydra overrides: `source/isaaclab_tasks/.../utils/hydra.py`; docs `docs/source/features/hydra.rst`
- Task enumeration: `scripts/environments/list_envs.py`
- Log aggregation: `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` (`extras["log"]`)

**groot-tao-agentic-rl-curriculum (reference to port)**
- Engine-agnostic core: `experiments/run-manager-core/core/{protocols,loop,registry,tripwire,digest,equivalence,journal}.py`
- Adapter/policy template: `experiments/run-manager-core/adapters/mock.py`
- Concrete backend adapter to mirror: `skills/agentic/sonic-job-adapter/job_adapter.py`
- Architecture doc: `docs/design/08-curriculum-manager-agent.md`
- Null-result methodology: `experiments/curriculum-manager-phase2/PHASE2_FINAL_REPORT_DRAFT.md`
- v6 redesign: `docs/design/09-teacher-llm-redesign-research.md`

**TAO skill format**
- Authoring spec: `tao-skills-bank/docs/authoring.md`; validator `scripts/validate-skills.sh`
