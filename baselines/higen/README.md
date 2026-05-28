# HiGen baseline (vendored)

HiGen — Hierarchical Graph Generation (Karami, NeurIPS 2024). Original:
<https://github.com/Karami-m/HiGen_main>. License: see `src/LICENSE`.

The implementation under `src/` is the SyNGLER-modified fork. Differences
from upstream:

- `configs/higen_syngler_<dataset>_single.yaml` — per-dataset single-graph
  configs covering `polblogs`, `dblp`, `yelp`, `youtube`. These pin the
  graph-coarsening, node-ordering, and posenc settings used in the paper.
- A training-loop optimization (~60× speedup on the per-iteration
  bottleneck identified during rebuttal) baked into `utils/`. Without
  this, the paper's training runs would not finish in compute budget.
- `utils/dataset/dataset_preprocessing.py::load_custom_graph_list` reads
  arbitrary `List[nx.Graph]` pickles, which is how the LSM-fitted real
  graphs are fed in.

## Install

```bash
cd baselines/higen/src && pip install -r requirement.txt
```

(`requirement.txt` is upstream's; `utils/` lists what HiGen actually imports
at runtime — most extra packages there are optional.)

## Run

```bash
python baselines/higen/run.py \
    --config baselines/higen/configs/<dataset>.yaml \
    --output runs/higen/<dataset>/seed=0
```

`<dataset>` ∈ {`polblogs`, `dblp`, `yelp`, `youtube`}. The runner

1. Loads `data/real/<dataset>/generator/seed=<S>.npy`, extracts the LCC,
   pickles it as `[nx.Graph]` to `src/data/syngler_<dataset>_single.pkl`
   (this is where the upstream config's `raw_graphs_path` points).
2. Writes `src/configs/runtime_<dataset>_single.yaml` that copies the
   upstream HiGen config and rewrites `exp_dir` to the `--output` path.
3. Shells out to `python main.py -c configs/runtime_<dataset>_single.yaml`
   from inside `src/`.

Final outputs (samples, checkpoints, config copies) land under
`<output>/higen/<dataset_name>/`.

## Notes

- HiGen requires a CUDA build of PyTorch with `torch_geometric` installed
  against the same CUDA version. Set `device: cuda:0` in the upstream
  config; CPU runs are not supported.
- Training is slow even after the 60× optimization — multi-day for
  `yelp` / `youtube`. The `single_ml` variant (multi-level) is slower
  still and not used in the main tables. Use `variant: single` for the
  paper setting.
- For paper-metric evaluation, point
  `scripts/reeval_paper_metrics.py --method higen --samples_root
  runs/higen/<dataset>` at the generated samples directory once a run
  completes.
