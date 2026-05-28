# CELL baseline (vendored)

CELL — Cross-Entropy Low-rank Logits (Rendsburg et al., ICML 2020). A
bilinear low-rank graph generator: `W = W_down W_up`, row-softmaxed into a
random-walk transition matrix, then sampled to a graph.

The implementation under `src/` is a snapshot of the public CELL repo
(<https://github.com/hheidrich/CELL>), unmodified. License: see
`src/LICENSE`.

## Install

```bash
cd baselines/cell/src && pip install -e .
# or just: pip install torch scipy networkx
```

## Run (real data)

```bash
python baselines/cell/run.py \
    --config baselines/cell/configs/<dataset>.yaml \
    --output runs/cell/<dataset>/seed=0
```

`<dataset>` ∈ {`polblogs`, `dblp`, `yelp`, `youtube`}. The runner reads
`data/real/<dataset>/generator/seed=<S>.npy` (the LSM-fitted adjacency),
extracts the largest connected component, trains CELL with `Adam(lr,
weight_decay)`, and writes 200 sampled adjacencies to `<output>/samples/rep*.npy`
plus a `metadata.json`.

## Notes

- CELL does **not** use node covariates; the comparison in the paper is
  structure-only.
- Default rank scales with graph size: 16 for `n ≤ 1500`, 24 for `n ≤ 2500`,
  32 otherwise. Override with `rank:` in the YAML.
- `EdgeOverlapCriterion(edge_overlap_limit=0.95)` provides early stopping;
  most runs converge well before `steps=400`.
- Output adjacencies are symmetric `uint8` matrices with zero diagonal —
  compatible with `scripts/reeval_paper_metrics.py --method cell`.
