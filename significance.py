import numpy as np

from metrics import calculate_sharpe_ratio


def run_significance_test(data, num_simulations=1000, seed=42):
    rng = np.random.default_rng(seed)
    invested_days = int(data["Position"].sum())
    num_days = len(data)

    simulated_sharpes = []

    for _ in range(num_simulations):
        random_indices = rng.choice(
            num_days,
            size=invested_days,
            replace=False,
        )

        random_position = np.zeros(num_days)
        random_position[random_indices] = 1

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
