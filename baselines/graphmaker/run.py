"""Runner stub for GraphMaker (attributed setting only)."""
import argparse, os, pathlib, subprocess, sys
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"clone upstream GraphMaker into {SRC} (see {HERE}/README.md)")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    # GraphMaker provides pre-trained checkpoints for cora/amazon_photo/amazon_computer.
    cmd = [sys.executable, "sample.py",
           "--dataset", cfg["dataset"],
           "--type", cfg["type"],
           "--num_samples", str(cfg["num_samples"]),
           "--out_dir", str(out)]
    print("[graphmaker-runner]", " ".join(cmd))
    subprocess.run(cmd, cwd=SRC, env=env, check=True)


if __name__ == "__main__":
    main()
