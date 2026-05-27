# LGD (Latent Graph Diffusion) baseline

## Install

```bash
cd baselines/lgd
git clone https://github.com/zhouc20/LatentGraphDiffusion src
cd src && pip install -r requirements.txt
```

## Caveat

LGD uses dense attention (O(N²) memory). It OOMs on graphs of n ≳ 1000
nodes — including most of the paper's real datasets. The paper rebuttal
reports this limitation. Treat this baseline as a smoke test on small
graphs (e.g., PolBlogs, n = 1222 — already OOMs on an A100; only the
smallest sparse-sim setting fits).

## Run

```bash
python baselines/lgd/run.py \
    --config baselines/lgd/configs/default.yaml \
    --output runs/lgd/seed=0
```
