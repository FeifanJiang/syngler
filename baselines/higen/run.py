"""Runner stub for HiGen — defers to the upstream entry script.

HiGen's upstream pipeline is config-heavy; the simplest integration is to
let it consume its own YAML config. This wrapper copies our flat YAML into
HiGen's expected layout (`<src>/exp/config.yaml`) and shells out to the
upstream `main.py`.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data_root", default="data/real")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"clone upstream HiGen into {SRC} (see {HERE}/README.md)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Stage adjacency where HiGen expects it
    src_npy = pathlib.Path(args.data_root) / cfg["dataset"] / "generator" / "seed=0.npy"
    staged = SRC / "data" / cfg["dataset"]
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_npy, staged / "adjacency.npy")

    higen_cfg = {
        "dataset": {"name": cfg["dataset"], "data_dir": str(staged)},
        "train": {"epochs": cfg["epochs"], "lr": cfg["lr"],
                  "batch_size": cfg["batch_size"]},
        "model": {"num_levels": cfg["num_levels"]},
        "test": {"num_samples": cfg["num_samples"], "out_dir": str(out)},
        "device": cfg["device"],
    }
    cfg_path = SRC / "exp" / f"runtime_{cfg['dataset']}.yaml"
    cfg_path.parent.mkdir(exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(higen_cfg, f)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "main.py", "-c", str(cfg_path)]
    print("[higen-runner]", " ".join(cmd))
    subprocess.run(cmd, cwd=SRC, env=env, check=True)
    print(f"[higen-runner] output -> {out}")


if __name__ == "__main__":
    main()
