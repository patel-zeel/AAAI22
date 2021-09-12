# Training:
```
python train.py [Fold] [Kernel] [Random_state] [DATA]
```
# Testing
```
python test.py [Fold] [Kernel] [Best-Model-Path] [DATA]
```

where
1. Kernel: 

    `rbf, matern12, matern32, matern52, matern_rbf, maternXrbf, local_p_delta`

2. Fold

    `0, 1, 2`

# Results
All the stationary GP baseline results are evaluated in `run_models.ipynb`
