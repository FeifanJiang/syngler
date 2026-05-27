# EDGE baseline (vendored)

EDGE — Efficient and Degree-Guided graph generation via Discrete
Diffusion (Chen et al., ICML 2023).

The implementation under `src/` is the SyNGLER-modified fork. Beyond the
original EDGE, this adds a `'real'` / `'REAL'` dataset branch in
`datasets/data.py` that loads a single adjacency from
`.edgelist/.txt/.npy/.pkl`. Original EDGE:
<https://github.com/tufts-ml/graph-generation-EDGE>. License:
see `src/LICENSE`.

## Install

```bash
cd baselines/edge/src && pip install dgl prettytable scikit-learn tensorboard torch-geometric tqdm
```

## Run (sparse simulation)

```bash
python baselines/edge/run.py \
    --config baselines/edge/configs/sparse_sim.yaml \
    --r 2 --seed 0 \
    --output runs/edge/r=2/seed=0
```

## Notes

- The stock PyTorch wheel supports CUDA compute capability up to
  `sm_90`. On clusters with newer GPUs constrain SLURM to compatible
  hardware via `SLURM_CONSTRAINT` (see `scripts/launch_sparse_sim.sh`).
- EDGE training is ~10 min on a single n=500 graph (500 epochs).
- Sampling: 200 graphs via `model.sample(5)` × 40 batches, ~12 min.
- Output: 200 `rep*.npy` binary adjacency matrices in `<output>/samples/`.
- For paper-metric evaluation use
  `python scripts/reeval_paper_metrics.py --method edge ...`.
- EDGE crashes if `<output>/real/` already exists — the runner removes it
  before training to handle resubmits.
