"""Runner stub for LGD. Defers to upstream training script."""
import argparse, os, pathlib, subprocess, sys
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="path to input adjacency .npy")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"clone upstream LGD into {SRC} (see {HERE}/README.md)")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "train_diffusion.py",
           "--data_path", str(args.data),
           "--out_dir", str(out),
           "--epochs", str(cfg["epochs"]),
           "--lr", str(cfg["lr"]),
           "--num_samples", str(cfg["num_samples"]),
           "--device", cfg["device"]]
    print("[lgd-runner]", " ".join(cmd))
    subprocess.run(cmd, cwd=SRC, env=env, check=True)


if __name__ == "__main__":
    main()
