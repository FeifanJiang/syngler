# GraphMaker baseline

GraphMaker — attributed-graph generator (Li et al., 2024).

## Install

```bash
cd baselines/graphmaker
git clone https://github.com/Graph-COM/GraphMaker src
cd src && pip install -r requirements.txt
```

## Caveat — applicability

GraphMaker is an **attributed** graph generator: it generates node
features and adjacency jointly. The paper's main sparse-simulation
setting (LSM-generated graphs without node features) is **not directly
applicable** to GraphMaker; only the attributed setting (Cora etc.)
yields a fair comparison.

This baseline is referenced only in Appendix F of the paper for the
link-prediction protocol.

## Run (attributed)

```bash
python baselines/graphmaker/run.py \
    --config baselines/graphmaker/configs/cora.yaml \
    --output runs/graphmaker/cora/seed=0
```
