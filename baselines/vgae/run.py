"""Runner for the vendored VGAE baseline.

Two entry points in ``src/``:
  - ``train.py``            : sparse-simulation training (input: DataGenerator .pkl)
  - ``train_real_data.py``  : real-data training (input: adjacency .npy)
"""
import argparse
import os
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r", type=int, default=None,
                    help="latent dim for sparse-sim setting; omit for real-data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default="data/sparse_sim",
                    help="for sparse-sim: dir with r=<r>/seed=<S>.pkl. "
                         "For real-data: dir with <dataset>/generator/seed=<S>.npy.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")

    if args.r is not None:
        # sparse-simulation: train.py reads pkl with model_Z / model_alpha
        data_pkl = pathlib.Path(args.data_root).resolve() / f"r={args.r}" / f"seed={args.seed}.pkl"
        cmd = [
            sys.executable, "train.py",
            "--model", "VGAE",
            "--data_path", str(data_pkl),
            "--output_dir", str(out),
            "--input_dim", "500",
            "--r", str(args.r),
            "--num_epoch", str(cfg["epochs"]),
            "--learning_rate", str(cfg["learning_rate"]),
        ]
    else:
        # real-data: train_real_data.py reads adjacency .npy
        data_npy = pathlib.Path(args.data_root).resolve() / cfg["dataset"] / "generator" / f"seed={args.seed}.npy"
        cmd = [
            sys.executable, "train_real_data.py",
            "--model", "VGAE",
            "--data_path", str(data_npy),
            "--output_dir", str(out),
            "--input_dim", str(cfg["input_dim"]),
            "--r", str(cfg.get("r", 6)),
            "--hidden2_dim", str(cfg.get("hidden2_dim", cfg.get("r", 6))),
            "--num_epoch", str(cfg["epochs"]),
            "--learning_rate", str(cfg["learning_rate"]),
        ]

    print("[vgae-runner]", " ".join(cmd))
    subprocess.run(cmd, cwd=SRC, env=env, check=True)
    print(f"[vgae-runner] output -> {out}")


if __name__ == "__main__":
    main()
