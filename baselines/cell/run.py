"""Runner for the CELL baseline.

Translates our config into CELL's `Cell` constructor + training loop.
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data_root", default="data/real")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"clone upstream CELL into {SRC} (see {HERE}/README.md)")
    sys.path.insert(0, str(SRC))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    # Load adjacency (the paper's prepared LSM-fitted .npy is the input)
    A = np.load(pathlib.Path(args.data_root) / cfg["dataset"] /
                "generator" / f"seed={args.seed}.npy")

    from cell.cell import Cell  # upstream import
    import scipy.sparse as sp
    A_sp = sp.csr_matrix(A)
    model = Cell(A_sp, H=cfg["H"], device=cfg["device"])
    model.train(steps=cfg["n_iters"], optimizer_fn="adam", lr=cfg["lr"])
    for k in range(cfg["num_samples"]):
        sample = model.sample_graph(seed=args.seed * 10_000 + k)
        np.save(out / "samples" / f"rep{k}.npy", np.asarray(sample.todense(), dtype=np.uint8))
    print(f"[cell-runner] {cfg['num_samples']} samples -> {out/'samples'}")


if __name__ == "__main__":
    main()
