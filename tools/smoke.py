"""
Smoke-test scripts/train_partition.py end to end, on real data, with
--fast_dev_run 1 (one train/val/test batch each) -- the only thing that
exercises main() at all; neither pytest nor tools/equivalence.py ever call
it (see docs/open_issues.md).

Six cases: the three committed loss variants in --mode full, plus
--mode local_only and --mode remote_only with the committed loss
(full_gnll_quantile_v2_landfill), plus land_fill_mode: zero in --mode full.
local_only/remote_only/zero_fill need their own run_name (so their output
dirs don't collide with --mode full's); this script builds those three
temporary configs itself, from the real committed yaml, in a throwaway
directory -- nothing under configs/ or sandbox/ is read-written for them.

  python tools/smoke.py

Requires: MHW_DATA_FILE, MHW_CLIM_FILE (real files). Sets its own
MHW_EXPERIMENTS_DIR (a temp dir, discarded after the run) and
WANDB_MODE=disabled (no real WandB writes, local or remote).
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


def _run_case(config_path, mode, env):
    start = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(config_path),
            "--mode",
            mode,
            "--fast_dev_run",
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

        cases = [(label, path, "full") for label, path in BASE_CONFIGS.items()]

        quantile_cfg = _load(BASE_CONFIGS["full_gnll_quantile"])

        local_cfg = copy.deepcopy(quantile_cfg)
        local_cfg["run_name"] = "smoke_local_only_fold0"
        local_path = tmp / "local_only.yaml"
        _write(local_cfg, local_path)
        cases.append(("local_only", local_path, "local_only"))

        remote_cfg = copy.deepcopy(quantile_cfg)
        remote_cfg["run_name"] = "smoke_remote_only_fold0"
        remote_path = tmp / "remote_only.yaml"
        _write(remote_cfg, remote_path)
        cases.append(("remote_only", remote_path, "remote_only"))

        zero_cfg = copy.deepcopy(quantile_cfg)
        zero_cfg["run_name"] = "smoke_zero_fill_fold0"
        zero_cfg["land_fill_mode"] = "zero"
        zero_path = tmp / "zero_fill.yaml"
        _write(zero_cfg, zero_path)
        cases.append(("zero_fill", zero_path, "full"))

        results = []
        for label, config_path, mode in cases:
            print(f"=== {label} (mode={mode}) ===", flush=True)
            code, elapsed, stdout, stderr = _run_case(config_path, mode, env)
            train, val, test = _extract_losses(stdout)
            results.append((label, code, elapsed, train, val, test))
            if code != 0:
                print(stdout[-3000:])
                print(stderr[-3000:])

    print()
    header = f"{'case':<20} {'exit':<5} {'elapsed_s':<10} {'train_loss':<12} {'val_loss':<10} {'test_loss':<12}"
    print(header)
    for label, code, elapsed, train, val, test in results:
        print(
            f"{label:<20} {code:<5} {elapsed:<10.1f} {train:<12} {val:<10} {test:<12}"
        )

    if any(code != 0 for _, code, *_ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
