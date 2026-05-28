"""Runner for the HiGen baseline.

HiGen's pipeline is config-heavy and structured around its own entry script
(`<SRC>/main.py -c configs/<name>.yaml`). This wrapper:

1. Loads the LSM-fitted adjacency at
   `data/real/<dataset>/generator/seed=<S>.npy`, extracts the LCC, and pickles
   it as a single-graph list to `<SRC>/data/syngler_<dataset>_single.pkl`
   (the path HiGen's upstream `higen_syngler_<dataset>_single.yaml` configs
   already point at).
2. Writes a runtime config under `<SRC>/configs/runtime_<dataset>.yaml`
   that overrides `exp_dir` to live under our `--output` directory.
3. Shells out to `python main.py -c configs/runtime_<dataset>.yaml` from
   inside `<SRC>`.

HiGen writes its outputs (generated graphs, snapshots, configs) under
`<output>/higen/<dataset.name>/<subdir>/`.
"""
from __future__ import annotations

import argparse
import copy
import os
import pathlib
import pickle
import subprocess
import sys

import networkx as nx
import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "src"


def _lcc(adj: np.ndarray) -> np.ndarray:
    sym = np.logical_or(adj > 0, (adj > 0).T)
    np.fill_diagonal(sym, False)
    g = nx.from_numpy_array(sym.astype(np.uint8))
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    out = nx.to_numpy_array(g, dtype=np.uint8)
    np.fill_diagonal(out, 0)
    return out


def _stage_pkl(npy_path: pathlib.Path, pkl_path: pathlib.Path) -> int:
    adj = _lcc(np.asarray(np.load(npy_path, allow_pickle=True)))
    g = nx.from_numpy_array(adj)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump([g], f)
    return g.number_of_nodes()


def _write_runtime_config(template: pathlib.Path, out_path: pathlib.Path,
                          exp_dir: pathlib.Path) -> None:
    with open(template) as f:
        cfg = yaml.safe_load(f)
    cfg["exp_dir"] = str(exp_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to baselines/higen/configs/<dataset>.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--data_root", default="data/real")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["dataset"]
    variant = cfg.get("variant", "single")

    template = SRC / "configs" / f"higen_syngler_{dataset}_{variant}.yaml"
    if not template.exists():
        raise SystemExit(f"Missing HiGen config: {template}")

    npy = pathlib.Path(args.data_root) / dataset / "generator" / f"seed={args.seed}.npy"
    if not npy.exists():
        raise SystemExit(f"Missing input adjacency: {npy}")

    pkl = SRC / "data" / f"syngler_{dataset}_{variant}.pkl"
    n = _stage_pkl(npy, pkl)
    print(f"[higen] staged {npy.name} -> {pkl.relative_to(SRC.parent)} "
          f"(LCC n={n})")

    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime_cfg = SRC / "configs" / f"runtime_{dataset}_{variant}.yaml"
    _write_runtime_config(template, runtime_cfg, exp_dir=out)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "main.py", "-c", f"configs/{runtime_cfg.name}"]
    print(f"[higen] cwd={SRC}")
    print(f"[higen] cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=SRC, env=env, check=True)
    print(f"[higen] outputs -> {out}/higen/")


if __name__ == "__main__":
    main()
