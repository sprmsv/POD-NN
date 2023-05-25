# POD-NN

After setting up the environment, the code can be run with such a command:
```bash
python run.py \
    --datadir ./data/230517/base --n_trn 2048 --n_val 512 \
    -L 40 \
    -D 4 -W 500 \
    --epochs 4e03 --lr 5e-04 \
    --normalize-output \
    --verbose
```

The normalization is triggered by passing an additional argument `--normalize-outputs`.

Run this command to get more information about the command line options:
```bash
python run.py --help
```
