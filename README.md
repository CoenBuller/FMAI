# FMAI: robustness-shaped MARL on VMAS Navigation

We train PPO agents (IPPO or MAPPO) on VMAS Navigation and compare reward signals:
the native VMAS reward, the robustness ρ of a qSFO spec, or a mix of both.
Every option is evaluated on the same spec-independent robustness, so the results are comparable.

## The current experiment (LIACS cluster)

One job runs **IPPO (`IPPO_individual.yaml`) and MAPPO (`experiment_1.yaml`)** × **`none`, `rho:coen`, `mixed:coen`** × **seeds 1–3**.
That's 18 runs in parallel on one L40S on `ceratanium`, about 5M env steps each, and it takes about 2.5 h.

```bash
bash submit.sh          # from this folder (WSL or Git Bash); copies the code to ~/fmai-ass1 on the cluster and runs sbatch
```

What to run is set in the block at the top of `cluster.sbatch` (`EXPERIMENTS`, `MODES`, `SEEDS`, `SMOOTH`, `EXTRA`).

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

## Running locally

```bash
PYTHONPATH=src uv run python -m experiments.experiment --exp IPPO_individual.yaml \
    --modes none rho:coen mixed:coen --seeds 3 --smooth 5 --plot compare
```

`--exp` is a file in `src/experiments/`. Every flag below overrides the YAML value for that run.
Note: `pyproject.toml` lists `qsfo` from `../artifact-fm-qsfo`. Nothing imports it, but `uv sync` needs that folder (see `setup.sh`).
The cluster job installs everything except qsfo.

## Flags (`experiments/experiment.py`)

| Flag | Config field | Options / meaning |
|---|---|---|
| `--exp FILE` | – | **Required.** YAML in `src/experiments/` (`config:` block = Config fields, `modes:` = default options) |
| `--modes ...` | YAML `modes` | Options, trained in this order: `none` (native reward), `rho[:spec]` (reward = spec robustness ρ), `mixed[:spec]` (native + `RHO_COEF`·ρ). Specs: `coen` (`luc` is unfinished). Labels in plots: `native`, `rho_coen`, `mixed_coen` |
| `--steps N` | `STEPS` | Training budget in env steps (rounded down to whole iterations of `NUM_ENVS × MAX_STEPS`). Default: `EPISODES × MAX_STEPS` |
| `--agents N` | `NUM_AGENTS` | Number of agents (default 4) |
| `--seeds N` | `N_SEEDS` | Seeds `SEED … SEED+N−1` per option (default 1) |
| `--seed N` | `SEED` | First seed |
| `--smooth K` | `SMOOTH` | Moving-average window (in eval points) for the compare plots; `1` = off |
| `--plot MODE` | `PLOT_MODE` | `diagnostic` (default): one 6-panel PNG per run (return, ρ, entropy/KL, actor loss, critic loss, grad norm), updated at every eval. `compare`: the same, **plus** at the end `compare.png` (eval return, eval ρ, actor & critic loss; all options overlaid, mean ± std over seeds) and `compare.csv` (final eval, mean ± std over seeds) |
| `--eval-spec S` | `EVAL_SPEC` | Robustness used for evaluation. `sfo` (default): spec-independent, ρ = minᵢ min(maxₜ μ1ᵢ, minₜ μ2ᵢ), i.e. every agent reaches its goal (`GOAL_EPS`) and never comes closer than `SAFETY_DIST`. Or a spec name, e.g. `coen` |
| `--no-early-stop` | `EARLY_STOP` | By default an episode ends once every agent is within `GOAL_EPS` of its goal; with this flag only `MAX_STEPS` ends it |
| `--run-id ID` | `RUN_ID` | Output folder name (default: timestamp). Parallel runs share one ID |
| `--compare-only` | – | No training: builds `compare.png` / `compare.csv` from the JSONs in `traces/<run-id>` |

**Other Config fields** (`src/config.py`), settable in the YAML:
- `GOAL_EPS` (0.1) and `SAFETY_DIST` (0.3): used by early stopping and the `sfo` eval robustness.
- `RHO_COEF`: weight of ρ in `mixed`.
- `EVAL_INTERVAL`: evaluate every N iterations, not episodes.
- `EVAL_EPISODES`.
- `EVAL_GIF`: a GIF at every eval. That's slow, and the final GIF is always saved anyway.
- `MAPPO`, `ENTROPY_COEF`, `LR`, `EPOCHS`, `MINIBATCH`, …

## Output

All output goes under `RESULTS_DIR` (default `results/luc`):
- `traces/<run_id>/`: per-run `<scenario>_<option>_seed<N>.json` (config + eval history) and `.png`, plus `compare.png` / `compare.csv`.
- `agents/<run_id>/`: checkpoints.
- `gifs/<run_id>/`: rollout GIFs.
- `error/`: skipped updates.
