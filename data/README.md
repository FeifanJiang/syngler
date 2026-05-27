# Datasets

This directory holds the input adjacency matrices used by SyNGLER and the
baselines. Two settings:

## 1. Sparse simulation

Generated on-the-fly from a latent-space model with `n=500` and
`r ∈ {2, 3, 4}` (20 MC reps per `r`). To produce the data:

```bash
python data/generate_sparse_sim.py --r 2 --seed_start 0 --seed_end 20
python data/generate_sparse_sim.py --r 3 --seed_start 0 --seed_end 20
python data/generate_sparse_sim.py --r 4 --seed_start 0 --seed_end 20
```

Each call writes two files per seed under `data/sparse_sim/r=<r>/`:

* `seed=<S>.pkl` — `DataGenerator` pickle with the true edge-probability
  matrix `P` and latent parameters. Required by **GRAN** (which trains on
  `P`-sampled adjacencies) and by **SyNGLER**'s LSM-fitting pipeline.
* `seed=<S>_A.npy` — one binary adjacency matrix sampled from `P` (the
  reference graph for that rep). Required by **EDGE** and by paper-metric
  evaluation as the reference.

DGP: `rho = -log(n) * sparse_level` with `sparse_level=0.4`; `Z` drawn from
a clipped-Gaussian mixture; `alpha` from Uniform`[-0.5, 0.5]`.

## 2. Real datasets

LSM-fitted adjacency matrices for the four real benchmarks are packaged
directly (see `data/real/`):

| dataset  | n     | edges    |
|----------|-------|----------|
| polblogs | 1,222 | ~17k     |
| dblp     | 2,879 | ~9k      |
| youtube  | ~1.1k | ~3k      |
| yelp     | 4,580 | ~46k     |

Each `data/real/<dataset>/generator/seed=0.npy` is the **single graph**
used as the LSM training input. SyNG-D/R then resample/diffuse over the
fitted latents to produce synthetic graphs.

To re-fit the LSM yourself instead of using the packaged adjacency, see
`syngler/lsm/runners/run_cora_syngr.py` for a worked example.
