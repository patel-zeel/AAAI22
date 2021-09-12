# Training:
```
python train_ngpu.py [Fold] [Kernel] [Random_state] [DATA]
```
# Testing
```
python test_ngpu.py [Fold] [Kernel] [Best-Model-Path] [DATA]
```

where
1. Kernel (Hamming Kernel enabled):
    
    `rbf, matern12, matern32, matern52, matern_rbf, maternXrbf, local_p_delta`
2. DATA:

    `prev`: Yearly Data
    
    `march`: Monthly data (March | One-hot encoded)
    
    `march_nsgp`: Monthly data (March | for Categorical Kernel)

3. Fold:

    `0, 1, 2`

# Results
All the stationary GP baseline results are evaluated in `run_models.ipynb`