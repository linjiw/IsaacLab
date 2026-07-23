# Isaac Lab RL Infrastructure Guide

**Audience:** a team designing new RL training methods (specifically curriculum-learning /
MaxRL-style work) on top of this repository. It explains how RL training is wired end-to-end
in Isaac Lab (v2.3.2, this fork), walks through one real robot task as the running example,
and ends with a concrete mapping from a frontier-teacher / curriculum-×-MaxRL design onto
this codebase's integration seams.

All file paths and line numbers were verified against this repo on 2026-07-22.

---

## 1. The one-paragraph mental model

Isaac Lab is a **massively-parallel GPU simulator** (built on Isaac Sim / PhysX) that exposes
robot tasks as **vectorized Gymnasium environments**: one `env` object simulates `num_envs`
(typically 4096) copies of the robot in a single GPU scene, and every quantity — observations,
rewards, dones, commands, curriculum state — is a batched torch tensor with a leading
`num_envs` dimension. The RL algorithm itself (PPO etc.) is **not** part of Isaac Lab: it
lives in external libraries (`rsl_rl`, `rl_games`, `skrl`, `sb3`) that consume the vectorized
env through thin wrappers. The environment side is authored **declaratively**: a task is a
Python config object composed of small "term" functions grouped under managers (observations,
actions, rewards, terminations, commands, events, curriculum). Changing a task — including
adding a curriculum — usually means editing or injecting config terms, not touching the
simulator or the training loop.

```
┌────────────────────────────────────────────────────────────────────────┐
│  scripts/reinforcement_learning/<lib>/train.py   (entry point, Hydra)  │
│        │ gym.make("Isaac-Velocity-Rough-Anymal-C-v0", cfg=env_cfg)     │
│        ▼                                                               │
│  ManagerBasedRLEnv  (source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
│   ├─ Scene (4096 robot clones, terrain, sensors)  ← GPU PhysX          │
│   ├─ CommandManager      what each env is asked to do (goals)          │
│   ├─ ActionManager       action tensor → joint targets                 │
│   ├─ ObservationManager  batched obs vector (+noise)                   │
│   ├─ RewardManager       weighted sum of reward terms                  │
│   ├─ TerminationManager  terminated / time_out flags                   │
│   ├─ EventManager        domain randomization (startup/reset/interval) │
│   └─ CurriculumManager   difficulty updates, fired on resets  ◄── the  │
│        ▲                                            curriculum seam    │
│        │ obs, rew, dones (all [4096, ...] tensors)                     │
│  RslRlVecEnvWrapper (source/isaaclab_rl/isaaclab_rl/rsl_rl/)           │
│        │                                                               │
│  rsl_rl OnPolicyRunner → PPO update   (external pip package)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repo layout (the RL-relevant parts)

| Path | What it is |
|---|---|
| `source/isaaclab/isaaclab/` | Core framework: envs, managers, scene, sensors, terrains, sim |
| `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` | The RL env class: `step()`, `_reset_idx()` |
| `source/isaaclab/isaaclab/managers/` | One file per manager (`reward_manager.py`, `curriculum_manager.py`, …) |
| `source/isaaclab/isaaclab/envs/mdp/` | The shared library of term functions: `rewards.py`, `observations.py`, `terminations.py`, `events.py`, `curriculums.py`, `commands/` |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/` | The task zoo (locomotion, manipulation, …), each task = configs + optional `mdp/` overlay |
| `source/isaaclab_tasks/isaaclab_tasks/direct/` | The "direct workflow" task zoo (hand-written env classes) |
| `source/isaaclab_rl/isaaclab_rl/` | Wrappers + config classes per RL library: `rsl_rl/`, `rl_games/`, `skrl.py`, `sb3.py` |
| `source/isaaclab_assets/` | Robot asset configs (`ANYMAL_C_CFG`, …) |
| `scripts/reinforcement_learning/{rsl_rl,rl_games,skrl,sb3,ray}/` | `train.py` / `play.py` per library |
| `scripts/environments/list_envs.py` | Enumerate all ~180 registered `Isaac-*` task IDs (`--keyword` filter) |
| `scripts/environments/{random_agent,zero_agent}.py` | Smoke-test an env without training |
| `logs/<library>/<experiment_name>/<timestamp>/` | All run outputs: TB events, checkpoints, resolved configs |
| `skills/agentic-rl/` | **This fork's prior curriculum-RL project** (LLM/scripted outer-loop curriculum, see §9.6) |

Two env-authoring workflows exist:

- **Manager-based** (`ManagerBasedRLEnv` + a `ManagerBasedRLEnvCfg`): declarative, term-based.
  This is where the curriculum machinery lives and where you should build. Everything below
  uses this workflow.
- **Direct** (`DirectRLEnv`): you hand-write `_get_observations() / _get_rewards() /
  _get_dones() / _apply_action()` (see `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py`).
  Faster to hack, no managers, no built-in curriculum hooks — you'd wire everything yourself.

---

## 3. Running example: `Isaac-Velocity-Rough-Anymal-C-v0`

The ANYmal-C quadruped must track a commanded base velocity (vx, vy, yaw-rate) while walking
over procedurally generated rough terrain. It's the canonical manager-based task and — usefully
for the curriculum work — it already ships with a native difficulty curriculum (terrain levels).

### 3.1 How a task ID becomes an environment

Tasks are registered with Gymnasium in each task folder's `__init__.py`
(`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/__init__.py`):

```python
gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",        # generic env class
    kwargs={
        "env_cfg_entry_point":    ".rough_env_cfg:AnymalCRoughEnvCfg",   # the task IS this config
        "rsl_rl_cfg_entry_point": ".agents.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "rl_games_cfg_entry_point": "agents:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point":     "agents:skrl_rough_ppo_cfg.yaml",
    },
)
```

Note the pattern: **the environment class is always the same** (`ManagerBasedRLEnv`); the task
identity is entirely in the *env config* class, and each RL library gets its own *agent config*
entry point alongside it. A "task" = (env cfg, agent cfg per library).

The config inheritance chain for our example:

```
LocomotionVelocityRoughEnvCfg          velocity_env_cfg.py   — robot-agnostic task definition
   └─ AnymalCRoughEnvCfg               rough_env_cfg.py      — swaps in self.scene.robot = ANYMAL_C_CFG
        └─ AnymalCRoughEnvCfg_PLAY                           — eval variant: 50 envs, no randomization/curriculum
```

### 3.2 The task definition, term by term

Everything below is from
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`.
This one file is worth reading in full — it is the template for every manager-based task.

**Scene** (`MySceneCfg`): a terrain generator (`ROUGH_TERRAINS_CFG` — a grid of terrain tiles
with difficulty increasing along rows), the robot articulation, a ray-caster height scanner
under the base, and contact sensors on every body. 4096 env clones, 2.5 m spacing.

**Commands** — what each env is *asked to do*:

```python
base_velocity = mdp.UniformVelocityCommandCfg(
    resampling_time_range=(10.0, 10.0),          # new command every 10 s
    ranges=Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), ...),
)
```

The command manager resamples goals on reset and on a timer, and — important for evaluation —
each command term maintains **metrics** (`Metrics/base_velocity/error_vel_xy`,
`error_vel_yaw`): accumulated tracking *error*, not a success rate (see §8's gotcha).

**Actions**: `JointPositionActionCfg(scale=0.5)` — the policy outputs one value per joint,
scaled and added to default joint positions, fed to PD controllers.

**Observations** (`policy` group, concatenated to one vector, with additive uniform noise):
base lin/ang velocity, projected gravity, the current velocity command, joint pos/vel, last
action, and the height-scanner grid. Note: **the command is part of the observation** — the
policy is goal-conditioned. This matters for curriculum design (§9.4).

**Rewards** — a weighted sum of small term functions:

| term | weight | role |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | task: exp(-error²/std²) velocity tracking |
| `track_ang_vel_z_exp` | +0.5 | task: yaw-rate tracking |
| `lin_vel_z_l2`, `ang_vel_xy_l2`, torques, accel, action-rate | −(small) | smoothness/effort penalties |
| `feet_air_time` | +0.125 | gait shaping |
| `undesired_contacts` (thighs) | −1.0 | don't crawl |

This is a **dense, shaped reward** — there is no built-in binary success signal. (Manipulation
tasks are often closer to verifiable success; e.g. the lift task has an
`object_reached_goal` termination available in
`manager_based/manipulation/lift/mdp/terminations.py`.)

**Terminations**:

```python
time_out     = DoneTerm(func=mdp.time_out, time_out=True)     # 20 s episode → truncation
base_contact = DoneTerm(func=mdp.illegal_contact, ...)        # base touches ground → failure
```

The termination manager exposes these as separate batched flags: `terminated` (real failures)
vs `time_outs` (truncations). **For locomotion, "reached time_out without terminating" is the
natural binary survival/success signal** — this is the cleanest Bernoulli evidence stream in
the whole system (§9.3).

**Events** — domain randomization, three modes: `startup` (friction, base mass, CoM), `reset`
(random initial pose/velocities/joints), `interval` (random base pushes every 10–15 s).

**Curriculum**:

```python
@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

One term. Its mechanics are §5.

### 3.3 The runtime loop — where everything fires

`ManagerBasedRLEnv.step()` (`manager_based_rl_env.py:153`):

```
step(action):                                    # action: [4096, num_joints]
  1. action_manager.process_action(action)
  2. for _ in range(decimation):                 # decimation=4, sim dt=0.005 → policy at 50 Hz
        apply action → write to sim → sim.step() → scene.update()
  3. episode_length_buf += 1;  common_step_counter += 1     # the global schedule clock
  4. reset_buf   = termination_manager.compute()            # [4096] bool
  5. reward_buf  = reward_manager.compute(dt=step_dt)       # [4096] float
  6. if any envs done:  _reset_idx(done_env_ids)            # in-place partial reset
  7. command_manager.compute(dt)                            # resample stale commands
  8. interval events (pushes)
  9. obs_buf = observation_manager.compute()
  return obs, reward, terminated, time_outs, extras
```

Two things to internalize:

- **Resets are partial and in-band.** There is no vector-env auto-reset wrapper; envs that
  finish are reset *inside* `step()` and continue with the batch. Episodes are asynchronous
  across the 4096 envs.
- **`common_step_counter`** counts policy steps (shared by all envs) and is the clock every
  step-scheduled curriculum uses (`modify_reward_weight` compares against it). With rsl_rl's
  `num_steps_per_env=24`, one *learning iteration* = 24 common steps = 24 × 4096 ≈ 98k samples.

`_reset_idx(env_ids)` (`manager_based_rl_env.py:349`) — the reset path, **in order**:

```
1. curriculum_manager.compute(env_ids)      ← curriculum fires FIRST, per resetting env
2. scene.reset(env_ids)                     ← uses (possibly just-updated) env_origins
3. event_manager.apply(mode="reset", ...)   ← randomize initial state
4. every manager's .reset(env_ids)          → returns logging dicts into extras["log"]:
     Episode_Reward/<term>, Episode_Termination/<term>,
     Metrics/<command>/<metric>, Curriculum/<term>
5. episode_length_buf[env_ids] = 0
```

Ordering matters: because curriculum runs *before* the scene reset, a curriculum term that
changes an env's difficulty (e.g. its terrain level → spawn origin) takes effect for the
episode that is about to start. This is exactly the seam a per-env difficulty teacher plugs
into (§9).

---

## 4. The manager/term pattern (how you extend anything)

Every manager consumes a config object whose attributes are term configs. A term is:

- a **function** `f(env: ManagerBasedRLEnv, env_ids/..., **params) -> tensor` for stateless
  terms, or
- a **class** inheriting `ManagerTermBase` (gets `__init__(cfg, env)` for state, `__call__`
  with the same signature, and a `reset(env_ids)` hook) for stateful terms.

You never subclass managers; you write terms and declare them in the task config. All the
stock terms live in `source/isaaclab/isaaclab/envs/mdp/`; tasks add local overlays under their
own `mdp/` folder (e.g. `locomotion/velocity/mdp/`). To modify a task without editing its file,
subclass its cfg (as `AnymalCRoughEnvCfg` does) or override leaves via Hydra (§6.3).

---

## 5. The built-in curriculum system

### 5.1 Mechanics

`CurriculumManager` (`source/isaaclab/isaaclab/managers/curriculum_manager.py`):

- `compute(env_ids)` is called **only from `_reset_idx`**, with only the envs being reset.
  A curriculum term's cadence is therefore the *reset stream*, not every step.
- Term signature: `term(env, env_ids, **params) -> state`. Whatever you return is stored and
  auto-logged to TensorBoard as `Curriculum/<term_name>` (scalars) or
  `Curriculum/<term_name>/<key>` (dict of scalars) on subsequent resets.
- Terms have full access to `env` — they can reach into any other manager and mutate it.
  That is the sanctioned pattern, not a hack (see the built-ins below).

### 5.2 The three generic knob-terms (`source/isaaclab/isaaclab/envs/mdp/curriculums.py`)

| term | what it does | example |
|---|---|---|
| `modify_reward_weight` | step-scheduled reward-weight change: when `common_step_counter > num_steps`, set `reward_manager` term weight | lift task ramps `action_rate` weight −1e-4 → −1e-1 after 10k steps (`lift_env_cfg.py`) |
| `modify_env_param` | generic getter/setter on **any dotted address** into the live env (`"event_manager.cfg.physics_material.func.material_buckets"`), transformed by a user `modify_fn`; return `NO_CHANGE` to skip | widen friction randomization ranges over training |
| `modify_term_cfg` | same, with shorthand addresses (`"commands.object_pose.ranges.pos_x"`) | walk a command range: automatic domain-range curriculum |

These three cover most "scripted schedule" curricula: any config leaf of any manager can be
walked over training with ~10 lines and no framework changes. **This is also the natural
actuation mechanism for a learned/adaptive teacher** — the teacher decides values, a curriculum
term writes them.

### 5.3 The adaptive example: `terrain_levels_vel`

`source/isaaclab_tasks/.../locomotion/velocity/mdp/curriculums.py:27` — the one *performance-
adaptive* curriculum in the stock library, and a compact template for anything you'll build:

```python
def terrain_levels_vel(env, env_ids, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    distance = ‖final xy position − spawn origin‖ for each resetting env
    move_up   = distance > terrain_tile_size / 2            # walked far → promote
    move_down = distance < commanded_speed * episode_len/2  # walked <half of asked → demote
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()            # logged as Curriculum/terrain_levels
```

`TerrainImporter.update_env_origins` (`source/isaaclab/isaaclab/terrains/terrain_importer.py:314`)
keeps a per-env integer `terrain_levels[4096]` and maps level → spawn origin on the terrain
grid (rows = difficulty). Envs that beat the max level get re-assigned to a random level
(anti-forgetting replay, built in). Structural takeaways:

1. **Difficulty is per-env state** (a `[num_envs]` tensor), evaluated and updated at each
   env's own reset — 4096 independent curriculum walkers, not one global knob.
2. The promotion signal is a hand-coded **binary threshold on episode outcome** — precisely
   the (bin, success) Bernoulli observation a frontier teacher would consume; the stock code
   just uses it greedily (±1 level) instead of maintaining a posterior.
3. Difficulty actuation = writing per-env state that the *next* episode's reset consumes
   (spawn origin). Command ranges, in contrast, are global cfg (§9.4's caveat).

---

## 6. The RL-algorithm side

### 6.1 Libraries and wrappers

The env is algorithm-agnostic; `source/isaaclab_rl/` holds one thin adapter per library:

| library | agent cfg format | Hydra prefix | checkpoints | notes |
|---|---|---|---|---|
| **rsl_rl** (default for locomotion) | Python `@configclass` (`agents/rsl_rl_ppo_cfg.py`) | `agent.*` | `model_<it>.pt` in run dir | leanest; PPO/distillation; pip pkg `rsl-rl-lib` (≥3.0.1 pinned in train.py) |
| **rl_games** | YAML | `agent.params.*` | `nn/*.pth` | broad task coverage |
| **skrl** | YAML | `agent.*` | `checkpoints/*.pt` | `trainer.timesteps = max_iterations × rollouts` (indirect) |
| **sb3** | YAML | `agent.*` | | slowest at this env count |

The wrapper (`RslRlVecEnvWrapper`, `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`)
just adapts tensor conventions — obs dict → tensor, dones merging, `extras["log"]` pass-through
(that's how `Curriculum/*` etc. reach TensorBoard). **If you write a custom training loop, you
can use these wrappers directly**; there's nothing sacred about the stock `train.py`.

For the example task, the PPO recipe (`agents/rsl_rl_ppo_cfg.py`, `AnymalCRoughPPORunnerCfg`):
actor/critic MLPs [512,256,128], `num_steps_per_env=24`, `max_iterations=1500`, lr 1e-3
adaptive (KL target 0.01), γ=0.99, λ=0.95, entropy 0.005. Training = 1500 iterations ×
~98k samples ≈ 147M env steps, minutes-to-~1h scale on one modern GPU.

### 6.2 The training entry point

`scripts/reinforcement_learning/rsl_rl/train.py` — read it once, it's ~230 lines:

1. `AppLauncher` boots Isaac Sim (before other imports — this ordering is mandatory).
2. `@hydra_task_config(task, agent_entry_point)` resolves the two configs from the gym registry
   and applies CLI overrides.
3. `gym.make(task, cfg=env_cfg)` → optional video wrapper → `RslRlVecEnvWrapper`.
4. `OnPolicyRunner(env, agent_cfg.to_dict(), log_dir).learn(max_iterations)`.
5. Dumps the fully-resolved `params/env.yaml` + `params/agent.yaml` into the run dir —
   **every run is exactly reproducible/diffable from its own log dir**.

### 6.3 Config overrides = Hydra (no file edits needed)

Every leaf of both configs is CLI-overridable
(`source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py`, docs in `docs/source/features/hydra.rst`):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Anymal-C-v0 --headless --num_envs 2048 --seed 42 \
  env.episode_length_s=15 \
  env.rewards.track_lin_vel_xy_exp.weight=1.5 \
  env.curriculum.terrain_levels=null \        # ← disable the curriculum entirely
  agent.algorithm.learning_rate=5.0e-4 agent.max_iterations=800
```

Setting a term to `null` removes it. This gives you ablation arms (curriculum ON/OFF, reward
variants, ranges) with zero code changes — use it for your baselines.

### 6.4 Commands cheat-sheet

```bash
# enumerate tasks
./isaaclab.sh -p scripts/environments/list_envs.py --keyword velocity

# train (headless), then evaluate a checkpoint with the *_PLAY cfg variant
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-Play-v0 --num_envs 32 --checkpoint <path>

# smoke-test an env without any learning
./isaaclab.sh -p scripts/environments/random_agent.py --task <ID> --headless
```

**In this fork's setup**, Isaac Lab runs inside the `isaac-lab-base` Docker container with
`/workspace` bind-mounted (host `logs/` ≡ container `logs/`); GPU commands are
`docker exec isaac-lab-base bash -c "cd /workspace/isaaclab && /isaac-sim/python.sh <script> ..."`.
See `skills/agentic-rl/README.md` for the working invocations.

---

## 7. Logging & metrics map

Runs land in `logs/<library>/<experiment_name>/<timestamp>/` with TensorBoard event files.
The env-side metrics all flow through `extras["log"]`, populated at resets:

| TB tag | producer | meaning |
|---|---|---|
| `Episode_Reward/<term>` | reward manager | per-term episodic mean, normalized by episode seconds |
| `Episode_Termination/<term>` | termination manager | how episodes are ending (e.g. `base_contact` count vs `time_out`) |
| `Metrics/<command>/<metric>` | command terms | e.g. `Metrics/base_velocity/error_vel_xy` — tracking **error** |
| `Curriculum/<term>` | curriculum manager | whatever your curriculum term returns (e.g. mean terrain level) |
| `Train/*`, `Loss/*`, etc. | RL library | learning-side diagnostics |

**Gotcha (important for anyone defining a success-based teacher or gate metric):** locomotion
tasks log tracking *error* and episode statistics — there is **no ready-made success rate**.
You must derive a binary/scalar success signal yourself; the two honest options are the
termination split (`time_outs` = survived, `terminated` = failed) and a thresholded task
predicate (e.g. tracking error below X, or the distance test `terrain_levels_vel` already
uses). Manipulation/dexterous/factory tasks are generally closer to verifiable success
(explicit `*_reached_goal`-style terms).

---

## 8. Vendored fork extras: `skills/agentic-rl/`

This branch carries a prior project that is directly adjacent to the proposed work: an
outer-loop curriculum controller (rule-based or LLM) that supervises live Isaac Lab runs at
checkpoint cadence via bounded knob changes, with a scripted-replay and fixed-control arm for
honest ON-vs-OFF comparison. Reusable pieces regardless of what you build:

- `isaaclab-run-digest/tb_reader.py` — TB-scalar → normalized record extraction (metric plumbing).
- `isaaclab-task-catalog/` — per-task gate-metric + knob-subset registry (the success-metric
  gotcha, solved per task).
- `isaaclab-knob-registry/` + `verify_gate.py` + `run_chaos_probe.sh`/`compute_tau.py` —
  bounded action spaces, validity gates, and a measured noise band (τ) for equivalence testing.
- Its headline caution (from both this project and the prior SONIC system): **adaptive
  curricula have repeatedly failed to beat scripted schedules here — always run a scripted
  arm and a no-curriculum control arm.**

---

## 9. Designing the proposed curriculum × MaxRL work on this infra

This section maps the frontier-teacher / MaxRL / hindsight design (the "estimator is the
curriculum" notes, `frontier_rl` framework) onto Isaac Lab's actual seams. The short version:
**Isaac Lab is the massively-parallel-sim row of your own regime table** — no rollout groups,
a per-reset Bernoulli evidence stream, on-policy PPO, dense rewards — and the natural
integration is *one custom curriculum term (teacher) + optionally one custom command term
(per-env difficulty) + optionally a modified rsl_rl algorithm (estimator weights)*.

### 9.1 Concept → seam mapping

| frontier_rl concept | Isaac Lab realization | where |
|---|---|---|
| task bins / difficulty axis | per-env state consumed at reset: terrain level (exists, per-env), command magnitude bin (needs custom command term, §9.4), event params (object mass etc.) | scene/terrain, command manager |
| `rollout_group(task_id, n)` | **does not exist** — 4096 persistent envs with asynchronous episodes; evidence is a *stream* of per-episode outcomes, no group size N | — |
| success verifier | termination manager: `env.termination_manager.time_outs` (survived) vs `.terminated` (failed) for locomotion; explicit success `DoneTerm` for manipulation | `termination_manager.py` |
| teacher observe + sample | a stateful curriculum term (`ManagerTermBase`): fires in `_reset_idx(env_ids)` *before* scene reset — read outcome of the ending episode, assign the bin for the next | `CurriculumManager.compute` |
| teacher checkpointing | serialize teacher state in the term; log posterior summaries via the term's return dict → free `Curriculum/*` TB telemetry | curriculum term |
| `Policy.update(trajectories, weights)` | the external RL library (rsl_rl PPO). Estimator changes (MaxRL vs GRPO-style advantages) are **library-side**, not env-side | `rsl-rl-lib`, or custom loop on `RslRlVecEnvWrapper` |
| hindsight relabel | hard under on-policy PPO (§9.5): statistics-only occupancy credit to the teacher is the validated fallback; full relabel needs goal-conditioned obs rewrite + off-policy handling | — |
| uniform floor / replay | built-in precedent: `update_env_origins` already randomizes envs that max out; your sampler owns this otherwise | teacher term |

Because there is no group size N, the advantage-mass utility `2(pass@N − pass@1)` has no N to
read off — your design notes already resolve this: **use the `learnability` utility
p̃(1−p̃)** on the reset stream (this is the `FrontierBinTeacher` row of your regime table,
with evidence-scaled decay so the half-life is invariant to env count).

### 9.2 The teacher as a curriculum term (sketch)

Wiring for the ANYmal task, using terrain level as the difficulty axis — this is a complete
integration, no framework changes:

```python
# e.g. source/isaaclab_tasks/.../locomotion/velocity/mdp/frontier_curriculum.py
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

class frontier_terrain_teacher(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n_bins = env.scene.terrain.max_terrain_level          # difficulty rows
        self.teacher = FrontierBinTeacher(n_bins=n_bins, utility="learnability",
                                          decay_half_life=2048)   # episode-equivalents

    def __call__(self, env, env_ids, uniform_floor: float = 0.1):
        terrain = env.scene.terrain
        # 1) OBSERVE: outcome of the episodes that just ended (one Bernoulli obs each)
        bins    = terrain.terrain_levels[env_ids]                        # [k] current bins
        success = env.termination_manager.time_outs[env_ids]            # survived = success
        self.teacher.observe_resets(bins.cpu().numpy(), (~success).cpu().numpy())
        # 2) SAMPLE: assign next-episode difficulty for exactly these envs
        new_bins = torch.as_tensor(self.teacher.sample_bins(len(env_ids)), device=env.device)
        terrain.terrain_levels[env_ids] = new_bins.clamp(0, terrain.max_terrain_level - 1)
        terrain.env_origins[env_ids] = terrain.terrain_origins[
            terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
        # 3) TELEMETRY: dict → Curriculum/frontier/* in TensorBoard, for free
        return {"mean_bin": terrain.terrain_levels.float().mean(),
                "frontier_bin": float(self.teacher.argmax_utility()),
                "dead_frac": float(self.teacher.dead_fraction())}

# in the env cfg (or injected via a subclass), replacing the stock term:
#   curriculum.terrain_levels = CurrTerm(func=frontier_terrain_teacher,
#                                        params={"uniform_floor": 0.1})
```

Why this placement is correct (verified in code): `_reset_idx` calls
`curriculum_manager.compute(env_ids)` at `manager_based_rl_env.py:356`, **before**
`scene.reset(env_ids)` — so the origins you write are the ones the new episode spawns from,
and `termination_manager` flags for `env_ids` still describe the episode that just ended.
This is the same contract `terrain_levels_vel` relies on.

Baselines come free: stock `terrain_levels_vel` = the greedy scripted-adaptive arm;
`env.curriculum.terrain_levels=null` = the no-curriculum arm; a `modify_term_cfg` schedule =
the scripted arm. Given §8's history, run all of them.

### 9.3 Choosing the success signal (per task, decide first)

- **Locomotion (this example):** `time_outs` (survival) is clean but saturates early;
  better: a task predicate at reset time, e.g. the distance-vs-commanded test already inside
  `terrain_levels_vel`, or thresholded `Metrics/base_velocity/error_vel_*`. The teacher should
  see only the binary outcome (your advantage-mass math assumes it); the dense shaped reward
  stays with PPO untouched.
- **Manipulation:** prefer tasks with an explicit success termination
  (`object_reached_goal`-style) — that's a real verifier. Check
  `skills/agentic-rl/isaaclab-task-catalog/` for per-task gate metrics already worked out.

### 9.4 Choosing the difficulty axis (the one real constraint)

Per-env difficulty needs per-env state that reset consumes. Options, in order of effort:

1. **Terrain level** (locomotion): already per-env (`terrain_levels[4096]`), already actuated
   at reset. Zero new machinery — start here.
2. **Command ranges** (goal difficulty — commanded speed, goal distance): the stock
   `UniformVelocityCommand` samples all envs from **one global `cfg.ranges`** — there is no
   per-env range. A curriculum can walk the *global* range (via `modify_term_cfg`), but per-env
   binning requires a small custom `CommandTerm` subclass that samples conditioned on a
   per-env bin tensor your teacher writes. ~50 lines; this is the "goal-conditioned reach"
   pattern from your MountainCar case study.
3. **Physics/event params** (mass, friction, push magnitude): via `modify_env_param`
   addresses; often global buckets, check per-axis whether per-env is possible.

Your shared-policy lesson transfers exactly: all 4096 envs already share one policy, and the
command/goal is already in the observation vector (`velocity_commands` ObsTerm) — so the
"condition on the goal, don't partition by it" contract holds by construction *as long as the
difficulty variable is observable*. If you bin by something the policy cannot observe (e.g.
terrain level is only implicitly visible through the height scanner — fine; commanded speed is
explicit — fine), verify there's a channel for competence to transfer through.

### 9.5 Estimator and hindsight under on-policy PPO — scope honestly

- The teacher (data distribution) is env-side and estimator-agnostic: the sketch above works
  with stock PPO unchanged. This is the low-risk, high-signal first experiment.
- **MaxRL-style success-conditioned weighting** is a change to the *advantage computation*
  inside the RL library. rsl_rl is the easiest to modify (small, class-based, pip-installable
  from source); alternatively write a custom training loop against `RslRlVecEnvWrapper`. Note
  the regime mismatch: PPO here is dense-reward with GAE, not group-based binary-reward — your
  own design table says dense-reward PPO → teacher with `learnability`, usually **skip**
  estimator changes and hindsight (dense reward already carries partial credit).
- **Full hindsight relabeling** (rewrite trajectory conditioning to the achieved goal) breaks
  on-policyness and requires the goal-in-obs rewrite (your contract 2) inside rsl_rl's rollout
  storage. Feasible but the most invasive option. The validated fallback from your own
  IsaacLab adapter design: **statistics-only hindsight** — credit the teacher's posterior for
  the bin actually achieved, don't touch the gradient.

### 9.6 Suggested experimental ladder

1. **Anymal-C rough, terrain-level teacher** (§9.2) vs {no-curriculum, stock
   `terrain_levels_vel`, scripted schedule} — same seeds, fixed wall-clock or fixed
   iterations, gate metric decided up front (§9.3). Pure env-side change.
2. Add the **custom command term** for commanded-speed bins (goal-difficulty axis) on flat
   terrain — the MountainCar-pattern experiment, still stock PPO.
3. Only if 1–2 show signal: estimator-side work (success-conditioned weighting in rsl_rl on a
   sparse-success manipulation task, where the verifier is real).

Log everything through the curriculum term's return dict (free TB `Curriculum/*` tags), and
diff arms via the auto-dumped `params/env.yaml` per run. The τ/noise-band and validity-gate
tooling in `skills/agentic-rl/` (§8) exists precisely to keep these comparisons honest.

---

## 10. Pitfalls checklist

- **Import order:** `AppLauncher` must boot before any `isaaclab.*`/torch-sim imports in entry
  scripts — copy the structure of an existing `train.py`.
- **Everything is batched:** term functions receive and must return tensors over `env_ids`;
  a stray `.item()` or python loop over 4096 envs will destroy throughput.
- **Curriculum cadence = resets, not steps.** With 20 s episodes and asynchronous resets, your
  term is called many times per iteration with small `env_ids` batches; keep it cheap, and use
  evidence-count-based decay (not wall-clock) so behavior is invariant to `num_envs`.
- **`common_step_counter` is policy steps**, not learning iterations (×`num_steps_per_env`).
- **No success metric in locomotion logs** — derive one; don't gate on `Metrics/*` error
  accumulators without thresholding (§7).
- **Command ranges are global** per term — per-env goal difficulty needs a custom command term (§9.4).
- **PLAY cfg variants** disable randomization *and curriculum* — evaluate with them, never
  train with them; conversely, don't report training-time terrain-level as an eval metric.
- **Determinism:** same seed + same GPU count only; `--distributed` perturbs seeds per rank
  (train.py:144). Measure your noise band before claiming a curriculum effect (`skills/agentic-rl/run_chaos_probe.sh`).
- **Prior art here says adaptivity is hard to prove:** SONIC and the Phase-1 pilot both
  reached null-ish results vs scripted schedules. Design the scripted arm first.
