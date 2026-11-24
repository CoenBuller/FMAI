### To-Do

- Test policy on an environment with an increased number of agents. 
- Test policy on an environment with stronger wind/wind gusts.
- Test policy on an environment where the target is moving faster with sharper turns. -> wont do that

# FMAI: robustness-shaped MARL on VMAS Navigation

We train PPO agents (IPPO or MAPPO) on VMAS Navigation and compare reward signals:
the native VMAS reward, the robustness ρ of a qSFO spec, or a mix of both.
Every option is evaluated on the same spec-independent robustness, so the results are comparable.

## The current experiment (LIACS cluster)

One job runs **`IPPO_zones.yaml`** (danger zones + inertia, see below) × **`rho`** (robustness reward) × **seeds 1–3** on one L40S on `ceratanium`,
at most `MAX_PARALLEL` runs at a time.
At the end it builds `compare.png`/`.csv`, the robustness analysis (`analysis.png`/`.csv`), the zero-shot scaling test (`scaling.png`/`.csv`) and the summary (see below).

```bash
bash submit.sh          # from this folder (WSL or Git Bash); copies the code to ~/fmai-ass1 on the cluster and runs sbatch
```

What to run is set in the block at the top of `cluster.sbatch` (see the variables under Flags).

**On the cluster:**
- Code, venv and uv cache are on `/local/$USER/fmai-ass1`.
- Results go to `/data/$USER/fmai-results`.
- Per-run logs are in `~/fmai-ass1/logs/<jobid>/`.

**Follow and fetch** (from WSL use `ssh.exe` / `scp.exe`, because the `l40s` alias lives in the Windows `~/.ssh/config`):

```bash
ssh.exe l40s 'squeue -u $USER; tail -n 20 ~/fmai-ass1/fmai-<jobid>.out'
ssh.exe l40s 'tail -n 2 ~/fmai-ass1/logs/<jobid>/*.log'                 # progress per run
scp.exe -r "l40s:/data/s3774740/fmai-results/luc/traces/<jobid>_*" output_temp/
ssh.exe l40s 'scancel <jobid>'                                         # stop the job
```

### Danger zones + inertia (`IPPO_zones.yaml`, `MAPPO_zones.yaml`)

An augmented Navigation (`src/navigation_sfo.py`):
- **Danger zones:** `N_ZONES: 2` circles (`ZONE_RADIUS: 0.25`) per episode, on the straight path of random agents, never on a spawn or goal. Agents can drive through them. They see zones through a second 12-ray LiDAR (`ZONE_LIDAR_RANGE: 0.6`, appended to the observation; inside a zone every ray reads 0).
- **Catastrophe:** each step inside a zone, with probability `CAT_PROB: 0.002` (1/500), the native reward gets `-CAT_PENALTY` (15). That is the only zone signal in the native reward. The value is balanced with scripted controllers (straight through vs around, γ = 0.99): the expected discounted catastrophe cost of crossing (about 2.8 % per crossing) roughly equals the discounted cost of the detour (about 0.36).
- **Inertia:** `AGENT_MASS: 2.5`, `DRAG: 0.1`. Same top speed as the default (mass 1, drag 0.25), but about 26 instead of 10 steps to reach it and a coasting distance of 0.35 instead of 0.12.
- **Robustness reward** (`GatedSpec` in `sfo.py`): one-step terms per agent, each normalised to [−1, 1] (0 = just satisfied, 1 = comfortably):
  - Prog: progress towards the goal in the last step (1 within `eps_g` of the goal), Agent: distance to the nearest agent, Zone: signed distance to the nearest zone edge.
  - Gate: if every term is ≥ 0 the reward is Prog, otherwise it is the worst term.
  - Scored on the true positions after the rollout (`reward_fn` in `train()`), not in the env.
  - The reward is `RHO_SCALE` (0.1) · ρ, so it stays near the native reward scale (`mixed` = native + 0.1·ρ).
  - Actor and critic gradients are clipped separately (1.0 each).
- **Eval** adds `zone_entry_rate` (episodes in which any agent entered a zone) and `catastrophes` per episode, and the `sfo` robustness gets a third term μ3 (distance to the nearest zone). `analyze.py` adds a Zone panel and the fraction of agents inside a zone over time.
- **GIFs** are always drawn with matplotlib, with zones (red) and catastrophes (black X).

Submit behind a running job: `SBATCH_ARGS="--dependency=afterany:<jobid>" bash submit.sh`.

## Running locally

**Setup** (once): `uv sync`. `pyproject.toml` lists `qsfo` from `../artifact-fm-qsfo`; nothing imports it, but `uv sync`
needs that folder (see `setup.sh`). The cluster job installs everything except qsfo.
Run all commands from this folder with `PYTHONPATH=src`. The zones yamls use `DEVICE: "cuda"`; set `DEVICE: "cpu"` without a GPU.

**The full navigation pipeline** (what `cluster.sbatch` does, but one run after the other):

```bash
export PYTHONPATH=src
# 1. Train: IPPO and MAPPO x native and robustness reward x 3 seeds (one run id per algorithm)
uv run python -m experiments.experiment --exp IPPO_zones.yaml  --modes none rho --seeds 3 --run-id ippo --train true
uv run python -m experiments.experiment --exp MAPPO_zones.yaml --modes none rho --seeds 3 --run-id mappo --train true
# 2. Compare figure + table per run id
uv run python -m experiments.experiment --exp IPPO_zones.yaml --run-id ippo --compare-only --smooth 5
uv run python -m experiments.experiment --exp MAPPO_zones.yaml --run-id mappo --compare-only --smooth 5
# 3. Robustness analysis of the final checkpoints (spec terms over time)
uv run python -m experiments.analyze --traces results/luc/traces/ippo --agents results/luc/agents/ippo
# 4. Zero-shot scaling: the same policies on 2, 4, 6 and 8 agents, without retraining
uv run python -m experiments.scaling --traces results/luc/traces/ippo --n-agents 2 4 6 8
# 5. One summary over both algorithms
uv run python -m experiments.summary --traces results/luc/traces/ippo results/luc/traces/mappo
```

For a quick check that everything runs, copy a zones yaml with `NUM_ENVS: 8`, `MAX_STEPS: 20`, `EPISODES: 16`,
`MINIBATCH: 80`, `EVAL_EPISODES: 4`, `DEVICE: "cpu"` (a few seconds per run).

**Coen's flocking experiments** use the same entry point: `flocking.yaml` / `flocking_wind.yaml` with `--train true`
(see `flocking_no_wind.sh`, `experiments.sh`). Without `--train`, the script runs his stress test of a saved flocking
policy (`stresstest.sh`, `stresstest.yaml`, `plot_stresstest.py`).

## Flags

### `experiments/experiment.py` (training)

`--exp` is a file in `src/experiments/`. Every other flag overrides the yaml value for that call.

| Flag | Config field | Meaning |
|---|---|---|
| `--exp FILE` | – | **Required.** Yaml in `src/experiments/` (`config:` block = Config fields, `modes:` = default options) |
| `--train true` | – | **Needed to train.** Without it, the script runs Coen's (flocking) stress test |
| `--modes ...` | yaml `modes` | Options, trained in this order: `none` (native reward), `rho[:spec]` (reward = spec robustness ρ), `mixed[:spec]` (native + `RHO_COEF`·ρ). Navigation always trains on `GatedSpec` (`SPEC: robustness`, labels `rho_robustness`, `mixed_robustness`). Flocking uses Coen's flocking spec |
| `--seeds N` | `N_SEEDS` | Seeds `SEED … SEED+N−1` per option (default 1) |
| `--seed N` | `SEED` | First seed |
| `--run-id ID` | `RUN_ID` | Output folder name (default: timestamp). Parallel runs share one id |
| `--compare-only` | – | No training: builds `compare.png` / `compare.csv` from the JSONs in `traces/<run-id>` |
| `--smooth K` | `SMOOTH` | Moving-average window (in eval points) for the compare plots; `1` = off |
| `--n_agents N` | `NUM_AGENTS` | Number of agents |
| `--mappo true/false` | `MAPPO` | Centralised critic |
| `--shared_reward true/false` | `SHARED_REW` | Team reward |
| `--results_dir DIR` | `RESULTS_DIR` | Output root |
| `--scenario S` | `SCENARIO` | `navigation` or `flocking` |
| `--mode M` | `REWARD_MODE` | Stress test only (training takes the mode from `--modes` or the yaml) |
| `--wind_strength X`, `--wind_gust_strength X` | `WIND_STRENGTH`, `WIND_GUST_STRENGTH` | Flocking only |
| `--wind_trained true/false` | – | Stress test only: load the policy trained with or without wind |

**Config fields** (`src/config.py`) that matter for navigation, settable in the yaml:
- `GOAL_EPS` (0.1) and `SAFETY_DIST` (0.3): the eval robustness ρ = minᵢ min(maxₜ μ1ᵢ, minₜ μ2ᵢ[, minₜ μ3ᵢ]), i.e. every agent reaches its goal, never comes closer than `SAFETY_DIST` to another agent (and never enters a zone). It is the same for every option. `GatedSpec.from_config` also takes `GOAL_EPS`, `SAFETY_DIST`, `ZONE_RADIUS`, `ZONE_LIDAR_RANGE`, `AGENT_MASS` and `DRAG` from the config.
- `N_ZONES`, `ZONE_RADIUS`, `ZONE_LIDAR_RANGE`, `CAT_PROB`, `CAT_PENALTY`, `AGENT_MASS`, `DRAG`: danger zones and inertia (see above).
- `RHO_SCALE`: the gated reward is `RHO_SCALE`·ρ. `RHO_COEF`: weight of ρ in `mixed`.
- `MAPPO`, `MAPPO_SHARED_REW`: by default MAPPO forces a team reward (`SHARED_REW`). `MAPPO_zones.yaml` sets `MAPPO_SHARED_REW: False`, so IPPO and MAPPO differ only in the critic and every agent gets its own reward and ρ.
- `EVAL_INTERVAL` (every N iterations), `EVAL_EPISODES`, `EVAL_GIF` (a GIF at every eval; slow, and the final GIF is always saved anyway).
- `ENTROPY_COEF`, `LR`, `EPOCHS`, `MINIBATCH` (must be ≤ `NUM_ENVS × MAX_STEPS`, or no update runs), `ACTIVATION` (`Tanh`, `ReLU`, `Sigmoid`), …

### `experiments/analyze.py` (robustness analysis)

Every checkpoint of a run is rolled out on the **same environments** (fixed seed, deterministic policy, all `MAX_STEPS`).
The terms of the robustness spec (Prog, Agent, Zone) and the gated total are plotted over time per option, for the worst
agent per episode, as mean ± 1 std over all episodes of all seeds. Writes `analysis.png` and `analysis.csv` (per option:
mean of each term, fraction of steps and of whole episodes with ρ ≥ 0, all agents at their goal at the end, progress per step).

| Flag | Default | Meaning |
|---|---|---|
| `--traces DIR` | required | `traces/<run_id>` |
| `--agents DIR` | required | `agents/<run_id>` |
| `--episodes N` | 100 | Environments per checkpoint |
| `--seed N` | 12345 | Env seed, the same for every model |
| `--out DIR` | `--traces` | Output folder |
| `--device D` | cuda if available | |

### `experiments/scaling.py` (zero-shot scaling)

Every checkpoint of a run, trained with `NUM_AGENTS` agents, is evaluated **without retraining** on environments with
other numbers of agents. This works because the policy shares its parameters over the agents and an agent's
observation does not depend on the number of agents. The metrics are those of the normal evaluation (success, pass rate,
return, safety violations, collisions, zone entries), as mean ± std over seeds. Writes `scaling.png` (metric vs number
of agents; dashed line = the number it was trained on) and `scaling.csv`.

| Flag | Default | Meaning |
|---|---|---|
| `--traces DIR` | required | `traces/<run_id>` |
| `--n-agents N ...` | required | Numbers of agents to test on, e.g. `2 4 6 8` |
| `--agents DIR` | `traces/<id>` → `agents/<id>` | Checkpoint folder |
| `--episodes N` | 100 | Environments per number of agents |
| `--seed N` | 12345 | Env seed, the same for every model |
| `--out DIR` | `--traces` | Output folder |
| `--device D` | cuda if available | |

### `experiments/summary.py` (summary over algorithms)

All algorithm × reward combinations of one or more run folders in one figure and one table:
- `summary_<ids>.png`: learning curves (eval return, eval ρ, pass rate, success, safety violations, zone entries). Colour = reward option, line style = algorithm (solid IPPO, dashed MAPPO), mean ± std over seeds.
- `summary_<ids>.csv` / `.md`: per combination, the final eval (mean ± std over seeds) and, from rollouts of the final checkpoints on the same envs, the steps until an agent and until all agents reach their goal, and the spec ρ per step.

| Flag | Default | Meaning |
|---|---|---|
| `--traces DIR ...` | required | Run folders `traces/<run_id>` (e.g. IPPO and MAPPO) |
| `--agents DIR ...` | `traces/<id>` → `agents/<id>` | Matching checkpoint folders |
| `--out DIR` | parent of the first `--traces` | Output folder |
| `--smooth K` | 5 | Moving average of the curves |
| `--episodes N`, `--seed N` | 100, 12345 | Rollouts of the final checkpoints |
| `--no-rollouts` | off | Only the curves and the final eval |
| `--device D` | cuda if available | |

### `cluster.sbatch` (variables at the top)

| Variable | Meaning |
|---|---|
| `EXPERIMENTS` | Yamls to run, e.g. `"IPPO_zones.yaml MAPPO_zones.yaml"` |
| `MODES` | Options, e.g. `"none rho"` |
| `SEEDS` | e.g. `"1 2 3"` |
| `SMOOTH` | Moving average of the compare plots |
| `SCALE_AGENTS` | Numbers of agents for the zero-shot scaling test (`""` = skip) |
| `EXTRA` | Extra flags for every training run |
| `MAX_PARALLEL` | Runs at once on the GPU |

## Output

All output goes under `RESULTS_DIR` (`results/luc` in the zones yamls):
- `traces/<run_id>/`: per-run `<scenario>_<option>_seed<N>.json` (config + eval history) and `.png`, plus `compare.png` / `compare.csv`, `analysis.png` / `analysis.csv` and `scaling.png` / `scaling.csv`.
- `agents/<run_id>/`: checkpoints.
- `gifs/<run_id>/`: a rollout GIF per run, saved after training (drawn with matplotlib when there is no display, e.g. on the cluster).
- `error/`: skipped updates.
