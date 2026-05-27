# VGAE baseline

Vendored: the paper's own re-implementation of Variational Graph
Auto-Encoders (Kipf & Welling, 2016) adapted for the SyNGLER eval
protocol. Source lives in `src/`. No external clone required.

## Run

```bash
# Sparse simulation
python baselines/vgae/run.py \
    --config baselines/vgae/configs/sparse_sim.yaml \
    --r 2 --seed 0 \
    --output runs/vgae/r=2/seed=0

# Real data
python baselines/vgae/run.py \
    --config baselines/vgae/configs/<dataset>.yaml \
    --output runs/vgae/<dataset>/seed=0
```

## Notes

- VGAE is much faster than GRAN/EDGE; the paper uses 200 MC reps.
- Output: a probability matrix `P.npy` per rep, plus 200 binary
  adjacency `rep*.npy` samples.
