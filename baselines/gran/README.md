# GRAN baseline (vendored)

GRAN — Graph Recurrent Attention Network (Liao et al., NeurIPS 2019).

The implementation under `src/` is the SyNGLER-modified fork. Two things
beyond the original GRAN: a `custom_pkl` data loader that reads
LSM-fitted `.pkl` files (with `model_Z`, `model_alpha`), and per-dataset
wrappers (`run_dblp.py`, `run_polblogs.py`, `run_youtube.py`). Original
GRAN: <https://github.com/lrjconan/GRAN>. License: see `src/LICENSE`.

## Install

```bash
cd baselines/gran/src && pip install -r requirements.txt
```

## Run (sparse simulation)

```bash
python baselines/gran/run.py \
    --config baselines/gran/configs/sparse_sim.yaml \
    --r 2 --seed 0 \
    --output runs/gran/r=2/seed=0
```

## Notes

- The stock PyTorch wheel supports CUDA compute capability up to
  `sm_90`. On clusters with newer GPUs constrain SLURM to compatible
  hardware (set `SLURM_CONSTRAINT` to your cluster's feature labels;
  the bundled `scripts/launch_sparse_sim.sh` reads this env var).
- GRAN trains for 500 epochs on a single graph and generates 200 samples.
- For n=500 sparse simulation, training is ~30 sec per seed, generation
  ~14 min (200 samples × ~4 sec each).
- The runner writes 200 `rep*.npy` probability matrices to
  `<output>/samples/`; threshold-sample with `np.random.RandomState(idx)`
  to obtain binary adjacency.
- For paper-metric evaluation, use
  `python scripts/reeval_paper_metrics.py --method gran ...`.
