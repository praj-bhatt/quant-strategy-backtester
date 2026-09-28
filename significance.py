import numpy as np

from metrics import calculate_sharpe_ratio


def run_significance_test(data, num_simulations=1000, seed=42):
    rng = np.random.default_rng(seed)

    num_days = len(data)

    long_days = int((data["Position"] == 1).sum())
    short_days = int((data["Position"] == -1).sum())

    simulated_sharpes = []

    for _ in range(num_simulations):
        random_position = np.zeros(num_days)

        long_indices = rng.choice(
            num_days,
            size=long_days,
            replace=False,
        )

        remaining_indices = np.setdiff1d(
            np.arange(num_days),
            long_indices,
        )

        short_indices = rng.choice(
            remaining_indices,
            size=short_days,
            replace=False,
        )

        random_position[long_indices] = 1
        random_position[short_indices] = -1

        random_returns = random_position * data["Asset_Return"]

        random_sharpe = calculate_sharpe_ratio(random_returns)

        simulated_sharpes.append(random_sharpe)

    simulated_sharpes = np.array(simulated_sharpes)

    actual_sharpe = calculate_sharpe_ratio(
        data["Position"] * data["Asset_Return"]
    )

    percentile_rank = (
        np.mean(simulated_sharpes <= actual_sharpe) * 100
    )

    threshold_95 = np.percentile(simulated_sharpes, 95)

    if actual_sharpe > threshold_95:
        verdict = "statistically significant"
    else:
        verdict = "not distinguishable from random"

    return {
        "random_sharpe_mean": simulated_sharpes.mean(),
        "random_sharpe_std": simulated_sharpes.std(),
        "actual_sharpe_percentile": percentile_rank,
        "verdict": verdict,
    }
