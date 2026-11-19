# STAT527-Knockoff-Project
Simulation study of two-stage fixed-X knockoffs under homoskedastic Gaussian, heavy-tailed, and Gaussian heteroskedastic errors.

##Requirements
Use Python 3.11 and NumPy 1.26.4.
```bash
python -m pip install numpy==1.26.4
```

## Generate a small test batch

Run this command from the project folder:

```bash
python generate_data.py --seed 527 --replicates 5 --output simulation_data_test
```

## Generate the full dataset
```bash
python generate_data.py --seed 527 --replicates 1000 --output simulation_data_main
```

The full run generates 1,000 replicates for each predictor correlation setting, rho = 0 and rho = 0.5.

Use a new output folder when rerunning the generator. The script stops if an output file already exists.

## Simulation design
- 400 observations and 500 predictors.
- 200 observations for screening and 200 for held-out selection.
- Signal counts of 10, 20, and 40.
- Each active set mixes coefficient magnitudes 0.25, 0.5, and 1.0.
- Three magnitude profiles: balanced, weak-heavy, and strong-heavy.
- Six error scenarios: Gaussian, standardized t with 3/5/10 degrees of freedom, and Gaussian heteroskedastic errors with h = 0.5/1.0.

Each replicate contains 9 coefficient configurations and 6 error scenarios, producing 54 response vectors.

## Output

Each replicate is saved as a compressed NumPy file (.npz) containing the predictor matrix, coefficient vectors, true supports, responses, errors, true standard deviations, and screening/held-out row indices.

Within each replicate, error scenarios share the design, coefficients, and sample split for paired comparisons.

## Reproducibility

All group members should use the same script version, parameters, seed, and NumPy version.

## Next steps
The script generates data. Subsequent analysis will fit LASSO, construct fixed-X knockoffs, and evaluate empirical FDR, power,and stage-specific signal losses.


