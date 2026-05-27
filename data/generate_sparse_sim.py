"""Generate sparse-simulation seeds for the paper's GRAN/EDGE comparison.

DGP: latent space model with
  * n = 500 nodes
  * r in {2, 3, 4} latent dimension
  * sparse_level = 0.4  (rho = -log(n) * sparse_level)
  * Z drawn from a clipped-Gaussian mixture, alpha from Uniform[-0.5, 0.5]

Produces two artifacts per (r, seed):
  data/sparse_sim/r=<r>/seed=<S>.pkl     # DataGenerator pickle (P matrix etc.)
  data/sparse_sim/r=<r>/seed=<S>_A.npy   # one sampled binary adjacency

Usage:
  python data/generate_sparse_sim.py --r 2 --seed_start 0 --seed_end 20
  python data/generate_sparse_sim.py --r 3 --seed_start 0 --seed_end 20
  python data/generate_sparse_sim.py --r 4 --seed_start 0 --seed_end 20
"""
import argparse
import json
import os
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from syngler.lsm.source import (  # noqa: E402
    sigmoid,
    symmetrization,
    UniformCovariateSampler,
    ClippedGaussianCovariateSampler,
    DataGenerator,
)


def clipped_gaussian_mixture(n, r):
    sample = ClippedGaussianCovariateSampler(n, mu=np.zeros(r),
                                             Sigma=np.eye(r), low=-2, up=2) / np.sqrt(r)
    v1 = np.random.uniform(-1, 1, size=r)
    v2 = np.random.uniform(-1, 1, size=r)
    mask = np.random.rand(n) < 0.5
    sample += np.where(mask[:, None], v1, v2)
    sample = sample / np.sqrt(np.linalg.norm(sample @ sample.T, "fro") / n)
    return sample


def generate_one(n, r, seed, sparse_level, out_dir):
    pkl_path = out_dir / f"seed={seed}.pkl"
    npy_path = out_dir / f"seed={seed}_A.npy"
    if pkl_path.exists() and npy_path.exists():
        return False, None

    np.random.seed(seed)
    rho = -np.log(n) * sparse_level
    with open(ROOT / "syngler" / "lsm" / "config" / "default.json") as f:
        cfg = json.load(f)
    p = cfg["p"]
    beta = np.zeros(p)
    X = symmetrization(np.zeros((n, n, p)))

    data = DataGenerator(beta, X, Z_enable=True, alpha_enable=True,
                         act=sigmoid, sparsity=rho)
    data.RefreshLatentVar(
        lambda n_: clipped_gaussian_mixture(n_, r),
        lambda n_: UniformCovariateSampler(n_, 1, -0.5, 0.5),
        Z_standardize=True,
    )

    with open(pkl_path, "wb") as f:
        pickle.dump({"data": data, "beta_true": beta}, f)

    rng = np.random.RandomState(seed)
    tril = np.tril(np.ones((n, n), dtype=bool), k=-1)
    A = np.zeros((n, n), dtype=np.uint8)
    A[tril] = (rng.rand(*data.P[tril].shape) < data.P[tril]).astype(np.uint8)
    A = A + A.T
    np.save(npy_path, A)
    return True, int(A.sum() // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--sparse_level", type=float, default=0.4)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--seed_end", type=int, default=20)
    ap.add_argument("--out_base", default=str(ROOT / "data" / "sparse_sim"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_base) / f"r={args.r}"
    out_dir.mkdir(parents=True, exist_ok=True)
    new, skip = 0, 0
    for s in range(args.seed_start, args.seed_end):
        was_new, edges = generate_one(args.n, args.r, s, args.sparse_level, out_dir)
        if was_new:
            new += 1
            if new <= 5 or s % 25 == 0:
                print(f"  n={args.n} r={args.r} seed={s}: edges={edges}")
        else:
            skip += 1
    print(f"\nDone: n={args.n} r={args.r} sparse_level={args.sparse_level} "
          f"seeds [{args.seed_start}, {args.seed_end}); created={new} skipped={skip} "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()
