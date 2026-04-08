# Notebook batch runner

Executes the same notebook once per config override from a JSON array. # noqa: E999
It does **not** require you to rewrite the notebook to read external config files.
Instead, it injects a small cell right after the notebook's `CONFIG = {...}` cell and runs the notebook with those overrides.

## Usage

From the repo root:

```bash
python src/utils/experiments/experiments.py \
  --notebook src/lee2019_finetune_sjepa.ipynb \
  --configs src/utils/experiments/configs/test.json \
  --kernel-name dl
```

Run in daemon mode (detached/background):

```bash
python src/utils/experiments/experiments.py \
  --notebook src/lee2019_finetune_sjepa.ipynb \
  --configs src/utils/experiments/configs/hard_subjects_current_keys_sweep.json \
  --kernel-name dl \
  --daemon
```

Daemon mode prints:

- `Run ID`
- `PID`
- `Status file`
- `Daemon log`

Each run writes `experiment_results/<run_id>/status.json` with progress fields like:

- `state`
- `completed_configs`
- `total_configs`
- `failed_configs`
- `pid`
- `updated_at`

## Status helper

Use the helper script to inspect run progress:

```bash
# Most recent run status
python src/utils/experiments/experiments_helper.py status

# Specific run
python src/utils/experiments/experiments_helper.py status --run-id <run_id>

# List only active runs (running/daemonized + live pid)
python src/utils/experiments/experiments_helper.py running

# List all runs that have a status file
python src/utils/experiments/experiments_helper.py list
```

## Config file shape

`utils/experiments/configs/configurations.json` must contain a JSON array of objects.
Each object is passed to `CONFIG.update(...)` inside the notebook.

Example:

```json
[
  {
    "seed": 46,
    "batch_size": 16,
    "learning_rate": 0.001
  }
]
```
