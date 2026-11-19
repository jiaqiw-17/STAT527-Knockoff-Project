import argparse
import json
from pathlib import Path
import numpy as np


def generate_replicate(seed, replicate, n, p, rho, signal_counts,
                       magnitudes, dfs, hetero_levels):
    """Generate all signal configurations and error scenarios for one replicate.

    With the defaults, one replicate contains:
      * one X matrix sample size = 400, preditors p = 500;
      * 9 beta vectors: 3 signal counts(s={10,20,40}) x 3 magnitude profiles(alpha={0.25,0.5,1.0});
      * 6 error vectors: Gaussian, three Student (v={3,5,10}), two heteroskedastic(h={0.5,1.0});
      * 54 response vectors: 9 beta vectors x 6 error vectors.

    The next replicate generates a new design, supports, signs, split and errors.
    Within this replicate, configurations share these random ingredients to
    make comparisons more directly paired.
    """
    # Combining the master seed and replicate index makes each replicate
    # reproducible. Generating replicate 10 alone gives the same result as
    # generating it as part of a batch with the same seed and parameters.
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
    # STEP 1: Generate AR(1) the correlated predictor matrix X.
    # Each row is an independent Gaussian observation. Within a row,
    # X[i, j] = rho * X[i, j-1] + sqrt(1-rho**2) * independent noise.
    # This gives population variance 1 and correlation rho**abs(j-k).
    # We do not normalize columns to unit length. 
    innovations = rng.standard_normal((n, p))
    X = np.empty_like(innovations)
    X[:, 0] = innovations[:, 0]
    innovation_scale = np.sqrt(1.0 - rho**2)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + innovation_scale * innovations[:, j]

    # STEP 2: Prepare signal locations, coefficient signs and sample splitting.
    # A random ordering of predictor indices defines nested active sets:
    # the first 10 active predictors are included in the first 20, and those 20 are included in the first 40. This is a deliberate paired design.
    # Each active variable keeps the same sign across coefficient magnitudes.
    # A sign is +1 or -1 with equal probability; counts need not be equal.
    permutation = rng.permutation(p)
    signs = rng.choice([-1.0, 1.0], size=p)
    # Shuffle row indices once, then assign half to screening and half to held-out selection. All 54 responses use this same split.
    # This code only records the split. It does not fit LASSO or knockoffs.
    rows = rng.permutation(n)
    n_screen = n // 2
    screen_idx = rows[:n_screen]
    heldout_idx = rows[n_screen:]

    # STEP 3: Generate the homoskedastic Gaussian baseline.
    # There is one independent N(0, 1) draw per observation.
    # Every observation has true conditional standard deviation 1.
    # Baseline error variance is fixed at 1 throughout this generator.
    # The Gaussian baseline and heteroskedastic scenarios reuse these draws.
    # This pairing isolates the change caused by observation-specific scales.
    gaussian = rng.standard_normal(n)
    errors = [gaussian]
    true_sd = [np.ones(n)]
    names = ["gaussian"]

    # STEP 4: Generate three homoskedastic heavy-tailed scenarios.
    # Defaults are df = 3, 5 and 10. Student t has variance df/(df-2), 
    # so multiplying by sqrt((df-2)/df) makes the error variance exactly 1.
    # The distribution remains heavy-tailed after this rescaling.
    # Every row in one scenario has the same error distribution.
    # Each df uses its own random draws. 
    # An empirical sample variance can differ from 1 even though the generating distribution has variance 1.
    for df in dfs:
        errors.append(rng.standard_t(df, size=n) * np.sqrt((df - 2.0) / df))
        true_sd.append(np.ones(n))
        names.append(f"t_df_{df:g}")

    # STEP 5: Generate two Gaussian heteroskedastic scenarios.
    # Defaults h = 0.5 and 1.0 control how strongly X[:, 0] changes variance.
    # Python column 0 is predictor X_{i1} in the mathematical notation.
    # Conditional variance is c * exp(h * X_{i1}), normalized to average 1.
    # This is the generating model, and its true standard deviations are saved for oracle rescaling. 
    # Estimated rescaling must fit its own variance model using screening responses in the later analysis.
    for h in hetero_levels:
        log_variance = h * X[:, 0]
        # Subtracting the maximum prevents overflow for large exponentials.
        # This constant cancels when we divide by the mean:
        # variance[i] = exp(h*X[i,0]) / mean(exp(h*X[:,0])).
        # The mean CONDITIONAL variance is exactly 1, while the realized
        # errors themselves still have random empirical variance.
        shifted = np.exp(log_variance - log_variance.max())
        variance = shifted / shifted.mean()
        sd = np.sqrt(variance)
        errors.append(sd * gaussian)
        true_sd.append(sd)
        names.append(f"hetero_h_{h:g}")

    # STEP 6: Mix ALL THREE magnitudes within each active set.
    # For magnitudes ordered weak to strong, the profiles allocate counts
    # approximately in ratios 1:1:1, 5:3:2, and 2:3:5.
    # Integer counts use the largest-remainder method: round down, then
    # assign remaining positions to the largest fractional remainders.
    # Ties favor the smaller magnitude. Strong-heavy counts are the reverse
    # of weak-heavy counts, giving paired mirror-image profiles.
    #
    # Default order: s=10 (balanced, weak-heavy, strong-heavy), then s=20
    # with the same three profiles, then s=40 with the same three profiles.
    # Active locations and signs are shared across profiles. One random
    # ordering of active positions assigns magnitudes independently of the
    # predictor column order. All profiles use this same assignment order.
    beta = []
    support = []
    configurations = []
    profile_names = []
    magnitude_counts = []
    magnitude_rng = np.random.default_rng(
        np.random.SeedSequence([seed, replicate, 1])
    )
    for count in signal_counts:
        active = permutation[:count]
        assignment_order = magnitude_rng.permutation(active)
        counts_by_profile = []
        for proportions in ([1 / 3, 1 / 3, 1 / 3], [0.5, 0.3, 0.2]):
            expected = count * np.asarray(proportions)
            counts = np.floor(expected).astype(int)
            remainder_order = np.argsort(-(expected - counts), kind="stable")
            counts[remainder_order[:count - counts.sum()]] += 1
            counts_by_profile.append(counts)
        counts_by_profile.append(counts_by_profile[1][::-1].copy())
        for profile, counts in zip(
            ("balanced", "weak_heavy", "strong_heavy"), counts_by_profile
        ):
            coefficients = np.zeros(p)
            assigned_magnitudes = np.repeat(np.asarray(magnitudes), counts)
            coefficients[assignment_order] = (
                assigned_magnitudes * signs[assignment_order]
            )
            beta.append(coefficients)
            support.append(coefficients != 0)
            configurations.append(f"s={count},profile={profile}")
            profile_names.append(profile)
            magnitude_counts.append(counts)

    # STEP 7: Combine every coefficient vector with every error scenario.
    # beta has shape (9, 500); errors has shape (6, 400) by default.
    # Scenario indices:
    #   0: Gaussian           1: t with df=3       2: t with df=5
    #   3: t with df=10       4: hetero h=0.5      5: hetero h=1.0
    beta = np.stack(beta)
    errors = np.stack(errors)
    # X @ beta.T computes all nine signal vectors, with shape (400, 9).
    # Transpose gives (9, 400), then [:, None, :] gives (9, 1, 400).
    # errors[None, :, :] has shape (1, 6, 400).
    # NumPy broadcasts these arrays to (9, 6, 400), implementing
    # y[configuration, scenario, :] = X @ beta[configuration] + errors[scenario].
    # Thus there are 54 response vectors, each containing 400 observations.
    # For example, y[4, 4, :] uses s=20, the weak-heavy profile and hetero h=0.5.
    # Within each scenario, all nine signal configurations share its errors.
    # Generate the reponse below.
    y = (X @ beta.T).T[:, None, :] + errors[None, :, :]

    # STEP 8: Return the design, outcomes, ground truth and reproducibility
    # information. Each saved file contains all configurations for ONE
    # replicate. It can later be reused for screening sizes k=50,70,90.
    # k, fitted sigma_hat, knockoff statistics, FDR and power are computed
    # in the analysis stage; they are not generated here.
    return {
        "X": X,
        "beta": beta,
        "support": np.stack(support),
        "signal_configurations": np.asarray(configurations),
        "profile_names": np.asarray(profile_names),
        "magnitudes": np.asarray(magnitudes),
        "magnitude_counts": np.stack(magnitude_counts),
        "scenario_names": np.asarray(names),
        "errors": errors,
        "true_sd": np.stack(true_sd),
        "y": y,
        "screen_idx": screen_idx,
        "heldout_idx": heldout_idx,
        "rho": np.asarray(rho),
        "seed": np.asarray(seed),
        "replicate": np.asarray(replicate),
    }


def main():
    # Command-line options allow changing settings without editing the code.
    # Example: python generate_data.py --replicates 5 --output simulation_data
    # Default 1000 replicates matches the simulation table. For an initial
    # check, request --replicates 5 and use a separate output directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("simulation_data"))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0,
                        help="First replicate index, useful for separate batches.")
    parser.add_argument("--seed", type=int, default=527)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--signals", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--magnitudes", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0])
    parser.add_argument("--dfs", type=float, nargs="+", default=[3, 5, 10])
    parser.add_argument("--hetero", type=float, nargs="+", default=[0.5, 1.0])
    args = parser.parse_args()
    # Check input ranges before generating data. In particular, df > 2
    # is required for finite Student t variance and our variance normalization.
    if not (2 <= args.n < args.p):
        parser.error("Require 2 <= n < p for this high-dimensional experiment.")
    if any(not (-1 < rho < 1) for rho in args.rho):
        parser.error("rho must lie strictly between -1 and 1.")
    if args.seed < 0 or args.start < 0 or args.replicates < 1:
        parser.error("Use nonnegative seed/start and positive replicates.")
    if any(s < 1 or s > args.p for s in args.signals):
        parser.error("Signal counts must lie between 1 and p.")
    if len(args.magnitudes) != 3 or not all(
        args.magnitudes[j] < args.magnitudes[j + 1] for j in range(2)
    ):
        parser.error("Provide exactly three increasing magnitudes.")
    if any(not np.isfinite(a) or a <= 0 for a in args.magnitudes):
        parser.error("Signal magnitudes must be finite and positive.")
    if any(not np.isfinite(df) or df <= 2 for df in args.dfs):
        parser.error("Student t degrees of freedom must exceed 2.")
    if any(not np.isfinite(h) for h in args.hetero):
        parser.error("Heteroskedasticity parameters must be finite.")

    # Generate replicates sequentially and save one compressed NumPy file each.
    # --start allows separate batches, e.g. --start 5 --replicates 5 creates
    # indices 5 through 9. Existing filenames trigger an error to protect data.
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "n": args.n, "p": args.p, "n_screen": args.n // 2,
        "n_heldout": args.n - args.n // 2, "rho": args.rho,
        "signal_counts": args.signals, "magnitudes": args.magnitudes,
        "profiles": {"balanced": [1/3, 1/3, 1/3],
                     "weak_heavy": [0.5, 0.3, 0.2],
                     "strong_heavy": [0.2, 0.3, 0.5]},
        "baseline_variance": 1, "t_df": args.dfs, "hetero_h": args.hetero,
        "analysis_settings": {"k_main": 70, "k_sensitivity": [50, 90],
                              "q": 0.1, "residual_folds": 5},
        "replicates": args.replicates, "start": args.start, "seed": args.seed,
        "numpy_version": np.__version__,
    }
    with (args.output / f"settings_start_{args.start}.json").open("x") as output:
        json.dump(settings, output, indent=2)

    for rho in args.rho:
        rho_dir = args.output / f"rho_{rho:g}"
        rho_dir.mkdir(parents=True, exist_ok=True)
        for replicate in range(args.start, args.start + args.replicates):
            path = rho_dir / f"replicate_{replicate:05d}.npz"
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}")
            data = generate_replicate(
                args.seed, replicate, args.n, args.p, rho,
                args.signals, args.magnitudes, args.dfs, args.hetero)
            with path.open("xb") as output:
                np.savez_compressed(output, **data)
            if replicate % 100 == 0 or replicate == args.start + args.replicates - 1:
                print(f"rho={rho:g}, completed replicate {replicate}", flush=True)



if __name__ == "__main__":
    main()
