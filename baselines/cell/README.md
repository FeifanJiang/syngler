# CELL baseline

CELL — Cross-Entropy Low-rank Logits — Rendsburg et al., ICML 2020.

A bilinear low-rank graph generator: $W = W_{\text{down}} W_{\text{up}}$,
softmax row-wise to get a transition matrix, then sample edges from
random walks.

## Install

```bash
cd baselines/cell
git clone https://github.com/hheidrich/CELL src
cd src && pip install -e .
```

## Run (real data)

```bash
python baselines/cell/run.py \
    --config baselines/cell/configs/<dataset>.yaml \
    --output runs/cell/<dataset>/seed=0
```

## Notes

- CELL does **not** support node covariates; the comparison in the paper
  is structure-only.
- Discussion of CELL vs SyNGLER appears in the rebuttal as a structural
  comparison. See `experiments/real_data/README.md`.
