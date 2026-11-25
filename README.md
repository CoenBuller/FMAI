### To-Do

- Test policy on an environment with an increased number of agents. 
- Test policy on an environment with stronger wind/wind gusts.
- Test policy on an environment where the target is moving faster with sharper turns. -> wont do that

# FMAI: robustness-shaped MARL on VMAS Navigation

We train IPPO agents on VMAS Navigation and compare two reward signals:
the native VMAS reward and the robustness ρ of a qSFO spec.
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
scp.exe -r "l40s:/data/s3774740/fmai-results/navigation" results/            # everything of the navigation runs
ssh.exe l40s 'scancel <jobid>'                                         # stop the job
```

### Danger zones + inertia (`IPPO_zones.yaml`)

An augmented Navigation (`src/environment/navigation.py`):
- **Danger zones:** `N_ZONES: 2` circles (`ZONE_RADIUS: 0.25`) per episode, on the straight path of random agents, never on a spawn or goal. Agents can drive through them. They see zones through a second 12-ray LiDAR (`ZONE_LIDAR_RANGE: 0.6`, appended to the observation; inside a zone every ray reads 0).
- **Catastrophe:** each step inside a zone, with probability `CAT_PROB: 0.002` (1/500), the native reward gets `-CAT_PENALTY` (15). That is the only zone signal in the native reward. The value is balanced with scripted controllers (straight through vs around, γ = 0.99): the expected discounted catastrophe cost of crossing (about 2.8 % per crossing) roughly equals the discounted cost of the detour (about 0.36).
- **Inertia:** `AGENT_MASS: 2.5`, `DRAG: 0.1`. Same top speed as the default (mass 1, drag 0.25), but about 26 instead of 10 steps to reach it and a coasting distance of 0.35 instead of 0.12.
- **Robustness reward** (`GatedSpec` in `sfo.py`): one-step terms per agent, each normalised to [−1, 1] (0 = just satisfied, 1 = comfortably):
  - Prog: progress towards the goal in the last step (1 within `eps_g` of the goal), Agent: distance to the nearest agent, Zone: signed distance to the nearest zone edge.
  - Gate: if every term is ≥ 0 the reward is Prog, otherwise it is the worst term.
  - Scored on the true positions after the rollout (`reward_fn` in `train()`), not in the env.
  - The reward is `RHO_SCALE` (0.1) · ρ, so it stays near the native reward scale.
  - Actor and critic gradients are clipped separately (1.0 each).
- **Eval** adds `zone_entry_rate` (episodes in which any agent entered a zone) and `catastrophes` per episode, and the `sfo` robustness gets a third term μ3 (distance to the nearest zone). `analyze.py` adds a Zone panel and the fraction of agents inside a zone over time.
- **GIFs** are always drawn with matplotlib, with zones (red) and catastrophes (black X).

Submit behind a running job: `SBATCH_ARGS="--dependency=afterany:<jobid>" bash submit.sh`.

## Running locally

**Setup** (once): `uv sync`.
Run all commands from this folder with `PYTHONPATH=src`. The zones yamls use `DEVICE: "cuda"`; set `DEVICE: "cpu"` without a GPU.

**The full navigation pipeline** (what `cluster.sbatch` does, but one run after the other):

```bash
export PYTHONPATH=src
# 1. Train: native and robustness reward x 3 seeds, all under one run id
uv run python -m experiments.experiment --exp IPPO_zones.yaml --modes none rho --seeds 3 --run-id ippo --train true
# 2. Compare figure + table (learning curves of both options)
uv run python -m experiments.experiment --exp IPPO_zones.yaml --run-id ippo --compare-only --smooth 5
# 3. Robustness analysis of the final checkpoints (spec terms over time)
uv run python -m experiments.analyze --results results/navigation --run-id ippo
# 4. Zero-shot scaling: the same policies on 2, 4, 6 and 8 agents, without retraining
uv run python -m experiments.scaling --results results/navigation --run-id ippo --n-agents 2 4 6 8
# 5. Summary: curves, final eval, steps to goal and spec rho per step of both options
uv run python -m experiments.summary --results results/navigation --run-id ippo
```

For a quick check that everything runs, copy a zones yaml with `NUM_ENVS: 8`, `MAX_STEPS: 20`, `EPISODES: 16`,
`MINIBATCH: 80`, `EVAL_EPISODES: 4`, `DEVICE: "cpu"` (a few seconds per run).

## Flocking

`SCENARIO: "flocking"` (`src/environment/flocking.py`): the agents follow a target that moves as a random walk. They must
stay in a band around it (`TARGET_RADIUS`, `EPSILON_T`), keep `AGENT_DISTANCE` from each other and match its speed
(`TARGET_REL_SPEED`, `EPSILON_V`, `V_MAX`), optionally under wind (`WIND_STRENGTH`, `WIND_GUST_STRENGTH`). The robustness
reward uses Coen's flocking spec (`build_floc_spec` in `sfo.py`, recovery window `TAU_FLOCK`), not `GatedSpec`.
Flocking uses the same results layout as navigation (see [Output](#output)): `results/flocking/<native|robustness>/`,
with the run id `no_wind` for training without wind and `wind` for training with wind.

| Yaml | Use |
|---|---|
| `flocking.yaml` | Training without wind (`modes: none, rho`) |
| `flocking_wind.yaml` | Training with wind and gusts 0.15 |
| `stresstest.yaml` | Evaluating saved policies under other wind / numbers of agents (`EVAL_EPISODES: 1000`) |

**1. Train.** Use the same entry point as navigation, with `--train true`. One call trains every mode of the yaml
(native and ρ); use run id `no_wind` or `wind`, because the stress test and `plots.py` look for those:

```bash
export PYTHONPATH=src
uv run python -m experiments.experiment --exp flocking.yaml      --seed 30 --seeds 5 --n_agents 3 --run-id no_wind --train true
uv run python -m experiments.experiment --exp flocking_wind.yaml --seed 30 --seeds 5 --n_agents 3 --run-id wind    --train true
uv run python -m experiments.experiment --exp flocking.yaml      --run-id no_wind --compare-only   # compare.png / .csv
uv run python -m experiments.experiment --exp flocking_wind.yaml --run-id wind    --compare-only
```

Each run writes `results/flocking/<native|robustness>/{agents,gifs,traces}/<no_wind|wind>/flocking_<mode>_seed<N>.*`.
`bash flocking_no_wind.sh flocking` and `bash flocking_wind.sh flocking_wind` do the same per seed (seeds 30–34,
3 agents), with a log per run in `local_logs/`.

**2. Stress test** (no `--train`). This loads
`results/flocking/<native|robustness>/agents/<wind|no_wind>/flocking_<mode>_seed<N>.pt` (`--wind_trained true|false`
picks the run id; `--run-id` overrides it), evaluates the policy on `EVAL_EPISODES` new envs with the wind and number
of agents you give, and writes one JSON to
`results/flocking/stresstest/seed_<N>/<mode>_<windtrained|nowindtrained>_n<A>_wind_<w>_gust_<g>.json`:

```bash
uv run python -m experiments.experiment --exp stresstest.yaml --mode rho --seed 30 --n_agents 6 \
    --wind_trained true --wind_strength 0.22 --wind_gust_strength 0.22
```

`bash stresstest.sh stresstest` runs the full grid: modes × seeds × 4–6 agents × trained with/without wind × wind × gusts.

**3. Plots** (both take `--results`, default `results/flocking`):
- `python plot_stresstest.py`: reads `stresstest/seed_*/*.json` and writes the pass-rate and return matrices
  (wind × gusts × number of agents) and the ρ − native deltas to `comparison/stresstest/`. It also prints a summary table.
- `python plots.py [--run-ids no_wind wind]`: sub-formula failure rates over training and final pass rate / return,
  native vs ρ, to `comparison/<run_id>/flocking.png`.
- `python gif_to_png.py <gif> [out.png]`: a contact sheet (a grid of 12 frames) of a GIF, for the report.

`analyze.py`, `scaling.py` and `summary.py` are built for navigation (the gated spec and zones); don't use them on
flocking runs.

## Flags

### `experiments/experiment.py` (training)

`--exp` is a file in `src/experiments/`. Every other flag overrides the yaml value for that call.

| Flag | Config field | Meaning |
|---|---|---|
| `--exp FILE` | – | **Required.** Yaml in `src/experiments/` (`config:` block = Config fields, `modes:` = default options) |
| `--train true` | – | **Needed to train.** Without it, the script runs Coen's (flocking) stress test |
| `--modes ...` | yaml `modes` | Options, trained in this order: `none` (native reward) and/or `rho` (reward = spec robustness ρ). Navigation trains on `GatedSpec` (label `rho_robustness`), flocking on Coen's flocking spec |
| `--seeds N` | `N_SEEDS` | Seeds `SEED … SEED+N−1` per option (default 1) |
| `--seed N` | `SEED` | First seed |
| `--run-id ID` | `RUN_ID` | Output folder name (default: timestamp). Parallel runs share one id |
| `--eval_gif true/false` | `EVAL_GIF` | A 5-env GIF at every evaluation, for the first seed of the call (`cluster.sbatch`: first seed only) |
| `--compare-only` | – | No training: builds `compare.png` / `compare.csv` from the JSONs in `traces/<run-id>` |
| `--smooth K` | `SMOOTH` | Moving-average window (in eval points) for the compare plots; `1` = off |
| `--n_agents N` | `NUM_AGENTS` | Number of agents |
| `--shared_reward true/false` | `SHARED_REW` | Team reward |
| `--results_dir DIR` | `RESULTS_DIR` | Output root |
| `--scenario S` | `SCENARIO` | `navigation` or `flocking` |
| `--mode M` | `REWARD_MODE` | Stress test only (training takes the mode from `--modes` or the yaml) |
| `--wind_strength X`, `--wind_gust_strength X` | `WIND_STRENGTH`, `WIND_GUST_STRENGTH` | Flocking only |
| `--wind_trained true/false` | – | Stress test only: load the policy trained with or without wind |

**Config fields** (`src/config.py`) that matter for navigation, settable in the yaml:
- `GOAL_EPS` (0.1) and `SAFETY_DIST` (0.3): the eval robustness ρ = minᵢ min(maxₜ μ1ᵢ, minₜ μ2ᵢ[, minₜ μ3ᵢ]), i.e. every agent reaches its goal, never comes closer than `SAFETY_DIST` to another agent (and never enters a zone). It is the same for every option. `GatedSpec.from_config` also takes `GOAL_EPS`, `SAFETY_DIST`, `ZONE_RADIUS`, `ZONE_LIDAR_RANGE`, `AGENT_MASS` and `DRAG` from the config.
- `N_ZONES`, `ZONE_RADIUS`, `ZONE_LIDAR_RANGE`, `CAT_PROB`, `CAT_PENALTY`, `AGENT_MASS`, `DRAG`: danger zones and inertia (see above).
- `RESULTS_DIR`, `SPLIT_RESULTS`: where the results go (see [Output](#output)).
- `RHO_SCALE`: the gated reward is `RHO_SCALE`·ρ.
- `SHARED_REW`: `False` (the zones yaml): every agent gets its own reward and ρ; `True`: all agents get the team reward.
- `EVAL_INTERVAL` (every N iterations), `EVAL_EPISODES`, `EVAL_GIF` (a GIF at every eval; slow, and the final GIF is always saved anyway).
- `ENTROPY_COEF`, `LR`, `EPOCHS`, `MINIBATCH` (must be ≤ `NUM_ENVS × MAX_STEPS`, or no update runs), `ACTIVATION` (`Tanh`, `ReLU`, `Sigmoid`), …

### `experiments/analyze.py` (robustness analysis)

Every checkpoint of a run is rolled out on the **same environments** (fixed seed, deterministic policy, all `MAX_STEPS`).
The terms of the robustness spec (Prog, Agent, Zone) and the gated total are plotted over time per option, for the worst
agent per episode, as mean ± 1 std over all episodes of all seeds. Writes `analysis.png` and `analysis.csv` (per option:
mean of each term, fraction of steps and of whole episodes with ρ ≥ 0, all agents at their goal at the end, progress per step).

| Flag | Default | Meaning |
|---|---|---|
| `--results DIR` + `--run-id ID` | – | The run: every `<option>/traces/<ID>` folder under `DIR` (e.g. `results/navigation`), or `traces/<ID>` in a flat layout |
| `--traces DIR ...` | – | Or: the `traces/<run_id>` folder(s) themselves |
| `--out DIR` | `<results>/comparison/<ID>` (or the first `--traces`) | Output folder |
| `--episodes N` | 100 | Environments per checkpoint |
| `--seed N` | 12345 | Env seed, the same for every model |
| `--device D` | cuda if available | |

### `experiments/scaling.py` (zero-shot scaling)

All checkpoints are found next to the traces (`<option>/agents/<run_id>`). Every checkpoint of a run, trained with `NUM_AGENTS` agents, is evaluated **without retraining** on environments with
other numbers of agents. This works because the policy shares its parameters over the agents and an agent's
observation does not depend on the number of agents. The metrics are those of the normal evaluation (success, pass rate,
return, safety violations, collisions, zone entries), as mean ± std over seeds. Writes `scaling.png` (metric vs number
of agents; dashed line = the number it was trained on) and `scaling.csv`.

| Flag | Default | Meaning |
|---|---|---|
| `--results DIR` + `--run-id ID` | – | The run: every `<option>/traces/<ID>` folder under `DIR` (e.g. `results/navigation`), or `traces/<ID>` in a flat layout |
| `--traces DIR ...` | – | Or: the `traces/<run_id>` folder(s) themselves |
| `--out DIR` | `<results>/comparison/<ID>` (or the first `--traces`) | Output folder |
| `--n-agents N ...` | required | Numbers of agents to test on, e.g. `2 4 6 8` |
| `--episodes N` | 100 | Environments per number of agents |
| `--seed N` | 12345 | Env seed, the same for every model |
| `--device D` | cuda if available | |

### `experiments/summary.py` (summary of a run)

Both reward options of a run in one figure and one table:
- `summary.png`: learning curves (eval return, eval ρ, pass rate, success, safety violations, zone entries), one colour per option, mean ± std over seeds.
- `summary.csv` / `.md`: per option, the final eval (mean ± std over seeds) and, from rollouts of the final checkpoints on the same envs, the steps until an agent and until all agents reach their goal, and the spec ρ per step.

| Flag | Default | Meaning |
|---|---|---|
| `--results DIR` + `--run-id ID` | – | The run, as for `analyze.py` |
| `--traces DIR ...` | – | Or: the `traces/<run_id>` folder(s) themselves |
| `--out DIR` | `<results>/comparison/<ID>` (or the first `--traces`) | Output folder |
| `--smooth K` | 5 | Moving average of the curves |
| `--episodes N`, `--seed N` | 100, 12345 | Rollouts of the final checkpoints |
| `--no-rollouts` | off | Only the curves and the final eval |
| `--device D` | cuda if available | |

### `cluster.sbatch` (variables at the top)

| Variable | Meaning |
|---|---|
| `RESULTS` | `RESULTS_DIR` of the yamls (`results/navigation`) |
| `EXPERIMENTS` | Yamls to run (`"IPPO_zones.yaml"`) |
| `MODES` | Options, e.g. `"none rho"` |
| `SEEDS` | e.g. `"1 2 3"` |
| `SMOOTH` | Moving average of the compare plots |
| `SCALE_AGENTS` | Numbers of agents for the zero-shot scaling test (`""` = skip) |
| `EXTRA` | Extra flags for every training run |
| `MAX_PARALLEL` | Runs at once on the GPU |

## Output

Navigation and flocking use the same layout (`SPLIT_RESULTS: True`): one root per scenario, one folder per reward option,
and a `comparison/` folder for everything that puts both options side by side.

```
results/<navigation|flocking>/
├── native/                      # REWARD_MODE none
│   ├── agents/<run_id>/         #   checkpoints: <scenario>_none_seed<N>.pt
│   ├── gifs/<run_id>/           #   a rollout GIF per run, saved after training
│   ├── traces/<run_id>/         #   config + eval history (.json) and diagnostics (.png) per run
│   └── error/<run_id>.txt       #   skipped updates, if any
├── robustness/                  # REWARD_MODE rho: the same folders (<scenario>_rho_seed<N>.*)
├── stresstest/seed_<N>/         # flocking only: one JSON per stress-test setting
└── comparison/
    ├── <run_id>/                # both options of one run: compare (+ navigation: analysis, scaling, summary;
    │                            #   flocking: flocking.png)
    └── stresstest/              # flocking only: matrices and deltas of the stress test
```

Run ids: navigation `ippo` (or the job id on the cluster), flocking `no_wind` and `wind`. The repo holds only the
final run, `25097_IPPO_zones` and `25097_flocking_wind` (cluster job 25097). Without `SPLIT_RESULTS` a run is kept in
`RESULTS_DIR/{agents,gifs,traces,error}/<run_id>/`; the compare, analyze and scaling scripts read that layout too.
