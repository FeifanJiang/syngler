# SyNGLER

**Efficient Synthetic Network Generation via Latent Embedding Reconstruction.**

SyNGLER is a two-stage framework for generating synthetic graphs that
preserve the structural properties of an observed network (degree
heterogeneity, clustering, spectral characteristics, sparsity). The first
stage fits a latent-space model (LSM) on the input graph; the second
stage generates new latent embeddings and reconstructs the graph from
them. This release provides both variants from the paper plus the
evaluation utilities, data, and baseline-runner stubs needed to
reproduce every experiment.

| Variant     | Stage-2 generator                                  | Notes                                |
|-------------|----------------------------------------------------|--------------------------------------|
| **SyNG-R**  | Bootstrap-resample the fitted latents `(Z, α)`     | Distribution-free, instant inference |
| **SyNG-D**  | Diffusion model over `(Z, α)`                      | ForestDiffusion (default) or MLP DDPM (GPU) |

The link function used at reconstruction is

```
P_ij = σ(Z_i^T Z_j + α_i + α_j + ρ),
```

and the adjacency is then sampled element-wise from `Bernoulli(P_ij)`.

## Layout

```
syngler/                     SyNGLER core (importable as `syngler`)
├── lsm/                     LSM fitting + simulated-data DGP
│   ├── source.py              Core PGD optimizer, DataGenerator,
│   │                          covariate samplers, sigmoid
│   ├── generator.py           Synthetic-graph generator utilities
│   ├── config/default.json    Default optimization hyperparameters
│   └── runners/               Top-level training scripts
│       ├── run_sim.py             LSM fit on sparse-sim seeds
│       ├── run_cora_syngr.py      SyNG-R on Cora (real-data demo)
│       └── run_cora_syngler_attr.py
│                                  SyNGLER-Attr (Algorithm 3, Appendix E)
├── res/                     SyNG-R API
│   ├── bootstrap.py           bootstrap_latents() / generate_graphs()
│   └── _scripts/              Original batch runners (`python -m …`)
├── diff/                    SyNG-D API
│   ├── forest.py              ForestDiffusion backend (paper default)
│   ├── mlp.py                 Residual-MLP DDPM backend (GPU)
│   └── _scripts/              Original batch runners
├── utils/source.py          reconstruct_adjacency(Z, α, ρ),
│                            bootstrap_alpha_Z, LCC helpers, …
└── evaluation/
    ├── metrics.py             Paper metrics:
    │                            triangle_density,
    │                            global_clustering_coefficient,
    │                            degree_centrality,
    │                            eigenvalues (Laplacian),
    │                            energy_distance, compute_mmd.
    └── orbit.py               ORCA-based orbit-MMD distance

baselines/                   Runner wrappers + (selectively) bundled source
├── README.md                Convention + per-baseline upstream URLs
├── gran/, edge/             VENDORED: SyNGLER-modified fork — has custom_pkl
│                            loader + real-data branch
├── vgae/                    VENDORED: paper's own implementation
├── cell/                    VENDORED: upstream snapshot, unmodified
├── higen/                   VENDORED: SyNGLER-modified fork — adds the
│                            higen_syngler_*_single configs + a 60× training
│                            optimization on the per-iter bottleneck
└── lgd/, graphmaker/        Stubs: clone upstream into src/ to use

data/
├── real/<dataset>/generator/   Packaged LSM-fitted adjacency (~40 MB)
│                               dblp, yelp, youtube, polblogs
└── generate_sparse_sim.py      Reproduce sparse-sim seeds

experiments/                 Reproduction recipes
├── sparse_simulation/         Appendix `tab:sparse-modern-baselines`
├── real_data/                 Main real-data tables
└── orbit_mmd/                 Orbit-MMD evaluation

scripts/
├── reeval_paper_metrics.py    Per-sample eval with paper metric defs
├── aggregate_paper_metrics.py Aggregate to a markdown / JSON table
└── launch_sparse_sim.sh       Local or SLURM-parallel launcher
```

## Install

```bash
git clone <this-repo> syngler && cd syngler
python -m pip install -e .
# optional, only for SyNG-D's forest backend:
python -m pip install ForestDiffusion
```

Python ≥ 3.9, PyTorch ≥ 2.0. The stock PyTorch wheel supports CUDA
compute capability up to `sm_90`; on newer GPUs (e.g. `sm_120`) you'll
see "no kernel image" unless you build PyTorch from source. The bundled
`scripts/launch_sparse_sim.sh` accepts a `SLURM_CONSTRAINT` env var (set
to your cluster's feature labels) to filter incompatible nodes.

## Quickstart

End-to-end on a single sparse-simulation seed:

```bash
# 1. Generate one input graph
python data/generate_sparse_sim.py --r 2 --seed_start 0 --seed_end 1

# 2. Run SyNG-R + SyNG-D (forest) on it
python experiments/sparse_simulation/run_syngler.py \
    --r 2 --seeds 0 --num_samples 200 --methods res,diff

# 3. Evaluate with paper metrics
for m in syngr syngd; do
  python scripts/reeval_paper_metrics.py \
      --method $m --r 2 --seeds 0 \
      --samples_root runs --out_dir runs/eval_paper
done

# 4. Aggregate
python scripts/aggregate_paper_metrics.py \
    --eval_root runs/eval_paper --methods syngr,syngd --rs 2 --max_n 1
```

Or library style:

```python
import numpy as np, torch, pickle
from syngler.res import generate_graphs as syngr
from syngler.diff.forest import generate_graphs as syngd
from syngler.evaluation.metrics import triangle_density

# Load a pre-fit LSM (e.g. from data/sparse_sim/r=2/seed=0.pkl)
with open("data/sparse_sim/r=2/seed=0.pkl", "rb") as f:
    dg = pickle.load(f)["data"]
Z, alpha, rho = dg.Z, dg.alpha, dg.sparsity

# Draw 200 SyNG-R graphs
samples = list(syngr(Z, alpha, n_reps=200, rho=rho, seed=0))
batch = torch.from_numpy(np.stack(samples).astype(np.float32))
print(triangle_density(batch, device="cpu").mean().item())
```

## Reproducing paper experiments

### 1. Sparse-simulation appendix (`tab:sparse-modern-baselines`)

Setting: `n=500`, `r ∈ {2, 3, 4}`, 20 Monte Carlo reps per `(r, method)`.

```bash
# Step 1: generate input data (~1 min total)
for r in 2 3 4; do
  python data/generate_sparse_sim.py --r $r --seed_start 0 --seed_end 20
done

# Step 2: install GRAN/EDGE deps (vendored under baselines/<name>/src/)
pip install -r baselines/gran/src/requirements.txt
pip install dgl prettytable scikit-learn tensorboard torch-geometric tqdm  # EDGE

# Step 3: launch all runs locally (GNU parallel) or via SLURM
MODE=local  bash scripts/launch_sparse_sim.sh         # uses GNU parallel
MODE=slurm  bash scripts/launch_sparse_sim.sh         # one slurm job per (method,r,seed)

# Step 4: paper-metric eval + aggregate
for r in 2 3 4; do
  SEEDS=$(seq -s, 0 19)
  for m in syngr syngd gran edge vgae; do
    python scripts/reeval_paper_metrics.py \
        --method $m --r $r --seeds $SEEDS \
        --samples_root runs --out_dir runs/eval_paper
  done
done
python scripts/aggregate_paper_metrics.py \
    --eval_root runs/eval_paper \
    --methods syngr,syngd,gran,edge,vgae --rs 2,3,4
```

The resulting `runs/eval_paper/summary.md` matches the paper's
`tab:sparse-modern-baselines` for the SyNG-D, SyNG-R, GRAN, and EDGE
rows (verified bit-for-bit on `r=2` against the values in the paper).

### 2. Real-data experiments

The four real datasets ship pre-fit (LSM already applied; see
`data/real/<dataset>/generator/seed=0.npy`).

```bash
# Fit cached LSM with SyNG-R + SyNG-D on one dataset
python experiments/real_data/run_syngler.py \
    --dataset polblogs \
    --fitted_pkl path/to/fitted_lsm_r=6.pkl \
    --output runs/syngler/polblogs

# Evaluate
python scripts/reeval_paper_metrics.py \
    --method syngr --r 6 --seeds 0 \
    --samples_root runs/syngler/polblogs --out_dir runs/eval_paper/polblogs
```

Re-fitting the LSM from raw input requires more compute; see
`syngler/lsm/runners/run_cora_syngr.py` for a worked example.

### 3. Orbit-MMD evaluation

```bash
# One-time: compile ORCA
g++ -O2 -std=c++11 -o syngler/evaluation/orca/orca \
                       syngler/evaluation/orca/orca.cpp

python syngler/evaluation/orbit.py \
    --reference data/real/dblp/generator/seed=0.npy \
    --generated runs/syngler/dblp/syngr/samples \
    --output runs/eval_paper/dblp_orbit.json
```

## Baselines

Other than VGAE (vendored as the paper's own reimplementation), each
baseline expects its upstream source under `baselines/<name>/src/`. See
`baselines/README.md` for upstream URLs and per-baseline run
instructions. Wrapper runners read a YAML config and shell out to the
upstream entry script; pass `--r/--seed` for sparse simulation or
`--dataset` for real data.

| Name        | Used in              | Upstream                                                   |
|-------------|----------------------|------------------------------------------------------------|
| **GRAN**    | Sparse sim (appendix)| bundled at `baselines/gran/src/` (SyNGLER fork; upstream: https://github.com/lrjconan/GRAN) |
| **EDGE**    | Sparse sim (appendix)| bundled at `baselines/edge/src/` (SyNGLER fork; upstream: https://github.com/tufts-ml/graph-generation-EDGE) |
| **VGAE**    | Main + appendix      | bundled at `baselines/vgae/src/` (paper's own implementation) |
| **CELL**    | Real data (rebuttal) | bundled at `baselines/cell/src/` (upstream: https://github.com/hheidrich/CELL) |
| **HiGen**   | Real data (rebuttal) | bundled at `baselines/higen/src/` (SyNGLER fork; upstream: https://github.com/Karami-m/HiGen_main) |
| **LGD**     | Real data (rebuttal) | https://github.com/zhouc20/LatentGraphDiffusion — OOMs on n≳1000, not vendored |
| **GraphMaker** | Appendix F note   | https://github.com/Graph-COM/GraphMaker                   |

End-to-end smoke tests have been run against GRAN and EDGE on the sparse-sim
setting (`r=2, seed=0`, 200 samples produced). CELL and HiGen are vendored with
working runners — both were exercised during the rebuttal real-data
experiments but have not been re-smoke-tested for this release; runners follow
the same arguments as those rebuttal invocations. LGD and GraphMaker wrappers
remain stubs (no `src/`); clone upstream into `baselines/<name>/src/` to use.

## Known limitations

- **SyNG-D forest** requires the `ForestDiffusion` package, which is
  CPU-only and trains via XGBoost; throughput is dominated by tree
  fitting. Use `syngler.diff.mlp` for a GPU-friendly DDPM alternative.
- **LSM fitting on large real graphs** (e.g. Yelp `n≈4.6k`) takes hours
  on CPU. We ship pre-fit adjacency artifacts (`data/real/...`) so users
  can run downstream steps without re-fitting.
- **The "Tri." metric in the paper** is *triangle density*
  `(# triangles) / C(n, 3)`, not transitivity. The
  `syngler.evaluation.metrics` module gives both; use
  `triangle_density(...)` for the paper convention.
- **`reconstruct_adjacency`** uses the full LSM link including
  `α` and `ρ`. An earlier internal `load_synthetic_data` used
  `P = Z @ Z.T` (no `α`, no `ρ`); that path has been removed from
  this release — always go through `reconstruct_adjacency`.

## Citation

```bibtex
@inproceedings{syngler2026,
  title     = {Efficient Synthetic Network Generation via Latent Embedding Reconstruction},
  author    = {Jiang, Feifan and Bu, Yinan and Wu, Shihao and Xu, Gongjun and Zhu, Ji},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

## License

MIT — see `LICENSE`. Baseline implementations remain the property of
their respective authors; please honour their licenses when cloning the
upstream repositories.
