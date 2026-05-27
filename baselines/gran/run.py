"""Thin runner wrapping the upstream GRAN entry point.

Reads our YAML config, fills in dataset path / seed / output dir, then
shells out to `<src>/run_exp.py`.
"""
import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--r", type=int, required=True, help="latent dimension (sparse-sim setting)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_root", default="data/sparse_sim",
                    help="dir containing r=<r>/seed=<seed>.pkl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"clone upstream GRAN into {SRC} (see {HERE}/README.md)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = pathlib.Path(args.output).resolve()
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    data_dir = pathlib.Path(args.data_root).resolve() / f"r={args.r}"
    cfg["exp_dir"] = str(out_dir)
    cfg["dataset"]["data_path"] = str(cache_dir) + "/"
    cfg["dataset"]["source_dir"] = str(data_dir) + "/"
    cfg["dataset"]["filename_glob"] = f"seed={args.seed}.pkl"
    cfg["test"]["gen_out_dir"] = str(out_dir / "samples") + "/"
    cfg["test"]["test_model_dir"] = "dummy"

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        cfg_path = f.name

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "run_exp.py", "-c", cfg_path]
    print("[gran-runner] cwd=", SRC, "cmd=", " ".join(cmd))
    subprocess.run(cmd, cwd=SRC, env=env, check=True)
    os.unlink(cfg_path)
    print(f"[gran-runner] samples saved to {out_dir/'samples'}")


if __name__ == "__main__":
    main()
