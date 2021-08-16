# Cover 5 examples

## Training (in the cluster environment)
```bash
./counting-trees/scripts/train_eval.py --config \
    ds/sahara \
    cover/5/a1 cover/5/metrics cover/5/plots \
    opt/adam1 \
    --train 80 --seed 0
```

## Evaluation (in the cluster environment)
```bash
./counting-trees/scripts/train_eval.py --config \
    ds/sahara \
    cover/5/a1 cover/5/metrics \
    opt/adam1 \
    --eval --seed 0 --slurm-minutes 5
```
