# HiGen baseline

HiGen — Hierarchical Graph Generation (Karami, NeurIPS 2024).

## Install

```bash
cd baselines/higen
git clone https://github.com/Karami-m/HiGen_main src
cd src && pip install -r requirements.txt
```

## Run

```bash
# Real data
python baselines/higen/run.py \
    --config baselines/higen/configs/<dataset>.yaml \
    --output runs/higen/<dataset>/seed=0
```

## Notes

- Training on real data is slow (~60× the original bottleneck before our
  optimization; see the paper rebuttal notes). Expect multi-day runs on
  large datasets like Yelp / YouTube.
- HiGen requires pre-processed multilevel decompositions; the runner
  triggers them automatically given the input adjacency.
