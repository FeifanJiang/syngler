"""Thin runner wrapping the upstream EDGE entry point.

Calls `<src>/train.py` for training, then loads the checkpoint and runs
`model.sample(num_generation)` × sample_batches to produce 200 binary
adjacency `.npy` files in `<output>/samples/`.
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
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_root", default="data/sparse_sim")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"clone upstream EDGE into {SRC} (see {HERE}/README.md)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = pathlib.Path(args.output).resolve()
    samples_dir = out / "samples"
    # EDGE crashes if log dir already exists
    for p in (out / "real", samples_dir):
        if p.exists():
            shutil.rmtree(p)
    samples_dir.mkdir(parents=True, exist_ok=True)

    data_npy = pathlib.Path(args.data_root).resolve() / f"r={args.r}" / f"seed={args.seed}_A.npy"
    if not data_npy.exists():
        raise SystemExit(f"missing {data_npy}; run data/generate_sparse_sim.py first")

    name = f"edge_r{args.r}_s{args.seed}"
    env = os.environ.copy()
    env["DGLBACKEND"] = "pytorch"
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")

    # Train
    train_cmd = [
        sys.executable, "train.py",
        "--dataset", "real",
        "--data_path", str(data_npy),
        "--epochs", str(cfg["epochs"]),
        "--lr", str(cfg["lr"]),
        "--num_generation", str(cfg["num_generation"]),
        "--eval_every", str(cfg["eval_every"]),
        "--check_every", str(cfg["check_every"]),
        "--device", cfg["device"],
        "--arch", cfg["arch"],
        "--name", name,
        "--log_home", str(out),
    ]
    if cfg.get("degree"):
        train_cmd.append("--degree")
    print("[edge-runner]", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=SRC, env=env, check=True)

    # Sample
    run_dir = out / "real" / "multinomial_diffusion" / "multistep" / name
    sample_py = f"""
import sys, os, torch, pickle, numpy as np, networkx as nx
sys.path.insert(0, {str(SRC)!r})
os.environ['DGLBACKEND'] = 'pytorch'
with open({str(run_dir / 'args.pickle')!r}, 'rb') as f: args = pickle.load(f)
args.device = {cfg['device']!r}
from datasets.data import get_data
train_loader, *_, initial_graph_sampler, _, _, _ = get_data(args)
args.num_node_classes = 2; args.num_edge_classes = 2; args.has_node_feature = False
from model import get_model
model = get_model(args, initial_graph_sampler=initial_graph_sampler)
ckpt = os.path.join({str(run_dir / 'check')!r}, f'checkpoint_{{ {cfg['epochs']!r}-1 }}.pt')
model.load_state_dict(torch.load(ckpt, map_location={cfg['device']!r}, weights_only=False)['model'])
model.to({cfg['device']!r}); model.eval()
import torch_geometric as pyg
count = 0
with torch.no_grad():
    for _ in range({cfg['sample_batches']!r}):
        gen = model.sample({cfg['num_generation']!r})
        for p in gen.to_data_list():
            G = pyg.utils.to_networkx(p, to_undirected=True)
            np.save(os.path.join({str(samples_dir)!r}, f'rep{{count}}.npy'),
                    nx.to_numpy_array(G).astype(np.uint8))
            count += 1
        torch.cuda.empty_cache()
print(f'Generated {{count}} samples')
"""
    subprocess.run([sys.executable, "-c", sample_py], cwd=SRC, env=env, check=True)
    print(f"[edge-runner] samples saved to {samples_dir}")


if __name__ == "__main__":
    main()
