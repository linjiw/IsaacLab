# Isaac Lab integration seams (file:line)

Exact points where the agentic-rl system attaches to Isaac Lab. Verify these still exist
before relying on them (paths are relative to the repo root `/workspace/IsaacLab`).

## Training / eval entry points
- `scripts/reinforcement_learning/rl_games/train.py` — RL-Games training. Key facts:
  - `--task`, `--num_envs`, `--seed`, `--checkpoint`, `--max_iterations`, `--headless`,
    `--agent` (default `rl_games_cfg_entry_point`). (argparse, lines ~17–47)
  - `agent_cfg["params"]["seed"]` / `["config"]["max_epochs"]` set from CLI (lines ~116–119).
  - `--checkpoint` → `params.load_checkpoint=True`, `params.load_path=<resolved>` (lines ~120–124).
  - log dir `logs/rl_games/<config.name>/<full_experiment_name>/` (lines ~142–154).
  - dumps `params/env.yaml` + `params/agent.yaml` (lines ~159–160).
- `scripts/reinforcement_learning/skrl/train.py` — skrl training. Key facts:
  - `--algorithm PPO|MAPPO|IPPO|AMP|...`, `--ml_framework torch|jax` (lines ~43–58).
  - **indirect max-iters:** `agent_cfg["trainer"]["timesteps"] = max_iterations × agent_cfg["agent"]["rollouts"]` (line ~152).
  - `agent_cfg["seed"]` from CLI (line ~164); log dir `logs/skrl/<directory>/<ts>_<algo>_<framework>/` (lines ~168–182).
  - resume via `runner.agent.load(resume_path)` (lines ~231–233); dumps `params/{env,agent}.yaml` (lines ~185–186).
- `scripts/reinforcement_learning/{rl_games,skrl}/play.py` — eval/rollout; use `-Play-v0` task
  variants (reduced num_envs, noise/DR disabled) for the pinned scoreboard.

## Native curriculum machinery
- `source/isaaclab/isaaclab/envs/mdp/curriculums.py` — `modify_reward_weight`,
  `modify_env_param`, `modify_term_cfg`.
- `source/isaaclab/isaaclab/managers/curriculum_manager.py` — `CurriculumManager`; term state
  flows into `env.extras["log"]` under `Curriculum/<term>`.
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
  — `terrain_levels_vel` (the flagship terrain-difficulty curriculum).
- Declared in env cfg as `CurrTerm(func=..., params=...)` — e.g.
  `.../locomotion/velocity/velocity_env_cfg.py` (`CurriculumCfg`, terrain_levels) and
  `.../manipulation/lift/lift_env_cfg.py` (`modify_reward_weight` with `num_steps`).

## Config overrides (Hydra)
- `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py` — `register_task_to_hydra`,
  `@hydra_task_config`. Every leaf overridable as `env.a.b=val` / `agent.x=val` on the CLI.
  Callables as `module:attr`; `null`=None; tuples as quoted lists (`"[-2.0, 2.0]"`).
  Legacy `--num_envs/--seed/--max_iterations` take precedence over Hydra.
- docs: `docs/source/features/hydra.rst`.

## Logging / observation
- `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` — merges each manager's log dict into
  `env.extras["log"]` on reset. Keys: `Episode_Reward/<term>`, `Metrics/<command>/<metric>`,
  `Curriculum/<term>`, `Episode_Termination/<term>`.
- Per-run: `params/env.yaml` + `params/agent.yaml` (resolved configs → config-drift verify),
  TensorBoard event files, checkpoints (`nn/*.pth` for rl_games, `checkpoints/*.pt` for skrl).

## Task enumeration
- `scripts/environments/list_envs.py` — iterates the gym registry; `--keyword` filter.
  ~179 unique `Isaac-*` task IDs.

## MDP term libraries (for authoring new curriculum terms)
- `source/isaaclab/isaaclab/envs/mdp/{rewards,observations,terminations,events,curriculums}.py`
  plus per-task `mdp/` overlays.
