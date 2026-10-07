"""
Smoke-test scripts/train_partition.py end to end, on real data, with
--limit_batches 1 and max_epochs: 1 (one real train/val/test batch, with
logging and checkpointing ACTIVE -- unlike --fast_dev_run, which Lightning
documents as suppressing both) -- the only thing that exercises main() at
all; neither pytest nor tools/equivalence.py ever call it (see
docs/open_issues.md).

Six cases: the three committed loss variants with mode: full, plus
mode: local_only and mode: remote_only with the committed loss
(full_gnll_quantile_v2_landfill), plus land_fill_mode: zero with mode: full.
Every case gets its own temporary yaml (run_name + max_epochs: 1 override,
and mode/land_fill_mode for the three non-base cases) so none of them touch
the real committed yaml files, configs/, or sandbox/.

After each case, checks that output_dir has: a checkpoint, metrics.csv,
resolved_config.yaml, model_config.json.

  python tools/smoke.py

Requires: MHW_DATA_FILE, MHW_CLIM_FILE (real files). Sets its own
MHW_EXPERIMENTS_DIR and WANDB_MODE=disabled (no real WandB writes, local or
remote). Everything -- including each case's ~50 MB checkpoint -- is
written under one temp directory, deleted when this script exits.
"""

import copy
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "scripts" / "train_partition.py"

BASE_CONFIGS = {
    "full_gnll_quantile": REPO
    / "configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml",
    "full_gnll_only": REPO / "configs/partition/full_gnll_v2_landfill/fold0.yaml",
    "full_mse": REPO / "configs/partition/full_mse_v2_landfill/fold0.yaml",
}


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _write(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)


def _run_case(config_path, env):
    start = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(config_path),
            "--limit_batches",
            "1",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    return proc.returncode, elapsed, proc.stdout, proc.stderr


def _last_match(pattern, text):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else "-"


def _extract_losses(stdout):
    train = _last_match(r"train_loss_epoch=([\d.eE+-]+)", stdout)
    val = _last_match(r"val_loss=([\d.eE+-]+)", stdout)
    test = _last_match(r"test_loss\s+([\d.eE+-]+)", stdout)
    return train, val, test


def _check_artifacts(run_dir):
    checks = {
        "checkpoint": any((run_dir / "checkpoints").glob("*.ckpt")),
        "metrics.csv": (run_dir / "metrics.csv").exists(),
        "resolved_config.yaml": (run_dir / "resolved_config.yaml").exists(),
        "model_config.json": (run_dir / "model_config.json").exists(),
    }
    return checks


def main():
    for var in ("MHW_DATA_FILE", "MHW_CLIM_FILE"):
        if not os.environ.get(var):
            raise EnvironmentError(f"{var} must point at the real data file")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = dict(os.environ)
        env["MHW_EXPERIMENTS_DIR"] = str(tmp / "experiments")
        env["WANDB_ENTITY"] = "smoke"
        env["WANDB_PROJECT"] = "smoke"
        env["WANDB_MODE"] = "disabled"

        quantile_cfg = _load(BASE_CONFIGS["full_gnll_quantile"])

        cases = []
        for label, path in BASE_CONFIGS.items():
            cfg = _load(path)
            cfg["max_epochs"] = 1
            cfg_path = tmp / f"{label}.yaml"
            _write(cfg, cfg_path)
            cases.append((label, cfg_path, cfg["run_name"]))

        local_cfg = copy.deepcopy(quantile_cfg)
        local_cfg["run_name"] = "smoke_local_only_fold0"
        local_cfg["mode"] = "local_only"
        local_cfg["max_epochs"] = 1
        local_path = tmp / "local_only.yaml"
        _write(local_cfg, local_path)
        cases.append(("local_only", local_path, local_cfg["run_name"]))

        remote_cfg = copy.deepcopy(quantile_cfg)
        remote_cfg["run_name"] = "smoke_remote_only_fold0"
        remote_cfg["mode"] = "remote_only"
        remote_cfg["max_epochs"] = 1
        remote_path = tmp / "remote_only.yaml"
        _write(remote_cfg, remote_path)
        cases.append(("remote_only", remote_path, remote_cfg["run_name"]))

        zero_cfg = copy.deepcopy(quantile_cfg)
        zero_cfg["run_name"] = "smoke_zero_fill_fold0"
        zero_cfg["land_fill_mode"] = "zero"
        zero_cfg["max_epochs"] = 1
        zero_path = tmp / "zero_fill.yaml"
        _write(zero_cfg, zero_path)
        cases.append(("zero_fill", zero_path, zero_cfg["run_name"]))

        results = []
        for label, config_path, run_name in cases:
            print(f"=== {label} ===", flush=True)
            code, elapsed, stdout, stderr = _run_case(config_path, env)
            train, val, test = _extract_losses(stdout)
            run_dir = tmp / "experiments" / "partition" / run_name
            checks = _check_artifacts(run_dir) if code == 0 else {}
            results.append((label, code, elapsed, train, val, test, checks))
            if code != 0:
                print(stdout[-3000:])
                print(stderr[-3000:])

    print()
    header = (
        f"{'case':<20} {'exit':<5} {'elapsed_s':<10} {'train_loss':<12} "
        f"{'val_loss':<10} {'test_loss':<20} {'artifacts missing'}"
    )
    print(header)
    all_ok = True
    for label, code, elapsed, train, val, test, checks in results:
        missing = [name for name, ok in checks.items() if not ok]
        if code != 0 or missing:
            all_ok = False
        print(
            f"{label:<20} {code:<5} {elapsed:<10.1f} {train:<12} {val:<10} "
            f"{test:<20} {missing or '-'}"
        )

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
