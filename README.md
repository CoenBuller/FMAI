### To-Do

- Test policy on an environment with an increased number of agents. 
- Test policy on an environment with stronger wind/wind gusts.
- Test policy on an environment where the target is moving faster with sharper turns. -> wont do that

# FMAI: robustness-shaped MARL on VMAS Navigation

We train PPO agents (IPPO or MAPPO) on VMAS Navigation and compare reward signals:
the native VMAS reward, the robustness ρ of a qSFO spec, or a mix of both.
Every option is evaluated on the same spec-independent robustness, so the results are comparable.

## The current experiment (LIACS cluster)

One job runs **`IPPO_zones.yaml`** (danger zones + inertia, see below) × **`rho:gated3`** × **seeds 1–3** on one L40S on `ceratanium`,
at most `MAX_PARALLEL` runs at a time.
At the end it builds `compare.png`/`.csv`, the robustness analysis (`analysis.png`/`.csv`) and the summary (see below).

```bash
bash submit.sh          # from this folder (WSL or Git Bash); copies the code to ~/fmai-ass1 on the cluster and runs sbatch
```

What to run is set in the block at the top of `cluster.sbatch` (`EXPERIMENTS`, `MODES`, `SEEDS`, `SMOOTH`, `EXTRA`, `MAX_PARALLEL`).

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
- **Spec `gated3`** (`GatedSpec` in `sfo.py`): one-step terms per agent, each normalised to [−1, 1] (0 = just satisfied, 1 = comfortably):
  - Prog: progress towards the goal in the last step (1 within `eps_g` of the goal), Agent: distance to the nearest agent, Zone: signed distance to the nearest zone edge.
  - Gate: if every term is ≥ 0 the reward is Prog, otherwise it is the worst term.
  - Scored on the true positions after the rollout (`reward_fn` in `train()`), not in the env.
  - The reward is `RHO_SCALE` (0.1) · ρ, so it stays near the native reward scale (`mixed` = native + 0.1·ρ).
  - Actor and critic gradients are clipped separately (1.0 each).
- **Eval** adds `zone_entry_rate` (episodes in which any agent entered a zone) and `catastrophes` per episode, and the `sfo` robustness gets a third term μ3 (distance to the nearest zone). `analyze.py` adds a Zone panel and the fraction of agents inside a zone over time.
- **GIFs** are always drawn with matplotlib, with zones (red) and catastrophes (black X).

Submit behind a running job: `SBATCH_ARGS="--dependency=afterany:<jobid>" bash submit.sh`.

## Running locally

```bash
PYTHONPATH=src uv run python -m experiments.experiment --exp IPPO_zones.yaml \
    --modes none rho:gated3 --seeds 3 --smooth 5 --train true
```

`--exp` is a file in `src/experiments/`. Every flag below overrides the YAML value for that run.
Note: `pyproject.toml` lists `qsfo` from `../artifact-fm-qsfo`. Nothing imports it, but `uv sync` needs that folder (see `setup.sh`).
The cluster job installs everything except qsfo.

## Flags (`experiments/experiment.py`)

| Flag | Config field | Options / meaning |
|---|---|---|
| `--exp FILE` | – | **Required.** YAML in `src/experiments/` (`config:` block = Config fields, `modes:` = default options) |
| `--modes ...` | YAML `modes` | Options, trained in this order: `none` (native reward), `rho[:spec]` (reward = spec robustness ρ), `mixed[:spec]` (native + `RHO_COEF`·ρ). Specs: `coen` (past-time, computed in the env at every step) and `gated3` (scored per agent on the finished batch, see `GatedSpec` in `sfo.py`). Labels in plots: `native`, `rho_coen`, `rho_gated3`, … |
| `--seeds N` | `N_SEEDS` | Seeds `SEED … SEED+N−1` per option (default 1) |
| `--seed N` | `SEED` | First seed |
| `--smooth K` | `SMOOTH` | Moving-average window (in eval points) for the compare plots; `1` = off |
| `--train true` | – | **Needed to train.** Without it `experiment.py` runs Coen's stress test of a saved flocking policy |
| `--run-id ID` | `RUN_ID` | Output folder name (default: timestamp). Parallel runs share one ID |
| `--compare-only` | – | No training: builds `compare.png` / `compare.csv` from the JSONs in `traces/<run-id>` |

**Other Config fields** (`src/config.py`), settable in the YAML:
- `GOAL_EPS` (0.1) and `SAFETY_DIST` (0.3): the eval robustness ρ = minᵢ min(maxₜ μ1ᵢ, minₜ μ2ᵢ[, minₜ μ3ᵢ]), i.e. every agent reaches its goal, never comes closer than `SAFETY_DIST` to another agent (and never enters a zone). It is the same for every option.
- `RHO_COEF`: weight of ρ in `mixed`.
- `EVAL_INTERVAL`: evaluate every N iterations, not episodes.
- `EVAL_EPISODES`.
- `EVAL_GIF`: a GIF at every eval. That's slow, and the final GIF is always saved anyway.
- `MAPPO`: centralized critic. By default MAPPO forces a team reward (`SHARED_REW`); `MAPPO_zones.yaml` sets `MAPPO_SHARED_REW: False`, so IPPO and MAPPO differ only in the critic and every agent gets its own reward and ρ.
- `ENTROPY_COEF`, `LR`, `EPOCHS`, `MINIBATCH` (must be ≤ `NUM_ENVS × MAX_STEPS`, or no update runs), …

## Robustness analysis (`experiments/analyze.py`)

After training, every checkpoint of a run is rolled out on the **same 100 environments** (fixed seed, deterministic policy, no early stopping, all `MAX_STEPS`).
The terms of the `gated3` spec (Prog, Agent, Zone) and the gated total are plotted over time per option, for the worst agent per episode, as mean ± 1 std over all episodes of all seeds.

```bash
PYTHONPATH=src uv run python -m experiments.analyze --traces results/luc/traces/<run_id> --agents results/luc/agents/<run_id>
```

Flags: `--episodes` (100), `--seed` (12345, the env seed shared by all models), `--out` (default `--traces`), `--device`.
It writes `analysis.png` and `analysis.csv` (per option: mean of each term, fraction of steps and of whole episodes with ρ ≥ 0, fraction of episodes with all agents on goal at the end, progress per step).

## Summary over algorithms (`experiments/summary.py`)

All algorithm × reward combinations of one or more run folders in one figure and one table (the cluster job runs it at the end):

```bash
PYTHONPATH=src uv run python -m experiments.summary --traces results/luc/traces/<id>_IPPO_zones results/luc/traces/<id>_MAPPO_zones
```

- `summary_<ids>.png`: learning curves for all combinations together (eval return, eval ρ, pass rate, success, safety violations, zone entries). Colour shows the reward option and line style the algorithm (solid IPPO, dashed MAPPO). Each curve is the mean ± std over seeds.
- `summary_<ids>.csv` / `.md`: one row per combination:
  - final eval as mean ± std over seeds;
  - from rollouts of the final checkpoints on the same 100 envs: steps until an agent reaches its goal, steps until all agents have (mean ± std and the fraction that got there), and the spec ρ per step (worst agent, mean ± std).
- Checkpoints come from `agents/<id>`; use `--agents` for another location or `--no-rollouts` to skip them.
- Output goes to the parent of the first `--traces` folder unless you pass `--out`.

## Output

All output goes under `RESULTS_DIR` (default `results/luc`):
- `traces/<run_id>/`: per-run `<scenario>_<option>_seed<N>.json` (config + eval history) and `.png`, plus `compare.png` / `compare.csv` and `analysis.png` / `analysis.csv`.
- `agents/<run_id>/`: checkpoints.
- `gifs/<run_id>/`: a rollout GIF per run, saved after training (drawn with matplotlib when there is no display, e.g. on the cluster).
- `error/`: skipped updates.
