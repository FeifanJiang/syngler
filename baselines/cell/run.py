"""Runner for the CELL baseline.

Loads a pre-fit LSM adjacency from `data/real/<dataset>/generator/seed=<S>.npy`,
trains CELL on the largest connected component, and dumps 200 sampled
adjacencies to `<output>/samples/rep*.npy`.

Mirrors the rebuttal-era CELL invocation: Adam optimizer with weight decay,
EdgeOverlapCriterion early-stop, rank scales with num_nodes if unset.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent
SRC_CELL_PKG = HERE / "src" / "src"   # CELL uses a src/ layout: <here>/src/src/cell/


def _default_rank(num_nodes: int) -> int:
    if num_nodes <= 1500:
        return 16
    if num_nodes <= 2500:
        return 24
    return 32


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_arg}, but torch.cuda.is_available() is False."
        )
    return device_arg


def _load_lcc_adjacency(path: pathlib.Path) -> tuple[np.ndarray, nx.Graph]:
    adjacency = np.asarray(np.load(path, allow_pickle=True))
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected square adjacency at {path}, got {adjacency.shape}")
    adjacency = adjacency > 0
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)
    graph = nx.from_numpy_array(adjacency.astype(np.uint8))
    if not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    graph = nx.convert_node_labels_to_integers(graph)
    adjacency = nx.to_numpy_array(graph, dtype=np.uint8)
    np.fill_diagonal(adjacency, 0)
    return adjacency, graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data_root", default="data/real")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not SRC_CELL_PKG.exists():
        raise SystemExit(
            f"Expected vendored CELL package at {SRC_CELL_PKG}. "
            f"This release ships CELL — was the src/ pruned?"
        )
    sys.path.insert(0, str(SRC_CELL_PKG))
    from cell.cell import Cell, EdgeOverlapCriterion  # noqa: E402

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = pathlib.Path(args.data_root) / cfg["dataset"] / "generator" / f"seed={args.seed}.npy"
    adjacency, graph = _load_lcc_adjacency(data_path)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    rank = cfg.get("rank") or _default_rank(num_nodes)
    device = _resolve_device(cfg.get("device", "auto"))

    out = pathlib.Path(args.output).resolve()
    (out / "samples").mkdir(parents=True, exist_ok=True)

    print(f"[cell] dataset={cfg['dataset']} nodes={num_nodes} edges={num_edges} "
          f"rank={rank} steps={cfg['steps']} lr={cfg['lr']} device={device}")

    callbacks = [EdgeOverlapCriterion(
        invoke_every=cfg.get("callback_every", 20),
        edge_overlap_limit=cfg.get("edge_overlap_limit", 0.95),
    )]
    model = Cell(A=sp.csr_matrix(adjacency), H=rank, callbacks=callbacks, device=device)
    model.train(
        steps=cfg["steps"],
        optimizer_fn=torch.optim.Adam,
        optimizer_args={"lr": cfg["lr"], "weight_decay": cfg.get("weight_decay", 1e-7)},
    )
    model.update_scores_matrix()

    for rep in range(cfg["num_samples"]):
        sampled = np.asarray(model.sample_graph().todense(), dtype=np.uint8)
        sampled = np.maximum(sampled, sampled.T)
        np.fill_diagonal(sampled, 0)
        np.save(out / "samples" / f"rep{rep}.npy", sampled)

    metadata = {
        "method": "CELL", "dataset": cfg["dataset"], "seed": args.seed,
        "rank": rank, "steps": cfg["steps"], "lr": cfg["lr"],
        "weight_decay": cfg.get("weight_decay", 1e-7), "device": device,
        "num_nodes": num_nodes, "num_edges": num_edges,
        "num_samples": cfg["num_samples"],
        "total_train_time_sec": model.total_time,
        "real_graph_path": str(data_path),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[cell] {cfg['num_samples']} samples -> {out/'samples'}")


if __name__ == "__main__":
    main()
