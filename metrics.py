import numpy as np

DAYS_PER_YEAR = 365.25
TRADING_DAYS_PER_YEAR = 252


def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    return drawdown.min()


def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_rf

    if excess_returns.std() != 0:
        return np.sqrt(TRADING_DAYS_PER_YEAR) * (
            excess_returns.mean() / excess_returns.std()
        )
    else:
        return 0.0


def calculate_cagr(cumulative_return, start_date, end_date):
    """Annualized return based on the actual calendar time elapsed."""
    elapsed_years = (end_date - start_date).days / DAYS_PER_YEAR

    if elapsed_years <= 0:
        raise ValueError(
            "Cannot calculate CAGR: end_date must be after start_date "
            f"(got {start_date} to {end_date})"
        )

    return cumulative_return ** (1 / elapsed_years) - 1


def calculate_metrics(
    data,
    transaction_cost=0.0005,
    short_borrow_rate=0.03,
):
    data = data.copy()

    data["Asset_Return"] = data["Adj Close"].pct_change(fill_method=None)


    data["Strategy_Return"] = data["Position"] * data["Asset_Return"]

    data["Transaction_Cost"] = data["Trade"] * transaction_cost

    daily_borrow_rate = short_borrow_rate / TRADING_DAYS_PER_YEAR

    data["Short_Borrow_Cost"] = (
        (data["Position"] < 0).astype(float) * daily_borrow_rate
    )

    data["Net_Strategy_Return"] = (
        data["Strategy_Return"]
        - data["Transaction_Cost"]
        - data["Short_Borrow_Cost"]
    )

    data["Cum_Asset_Return"] = (1 + data["Asset_Return"]).cumprod()
    data["Cum_Strategy_Return"] = (
        1 + data["Net_Strategy_Return"]
    ).cumprod()

    max_drawdown = calculate_max_drawdown(
        data["Cum_Strategy_Return"]
    )

    buy_hold_max_drawdown = calculate_max_drawdown(
        data["Cum_Asset_Return"]
    )

    sharpe_ratio = calculate_sharpe_ratio(
        data["Net_Strategy_Return"]
    )

    buy_hold_sharpe_ratio = calculate_sharpe_ratio(
        data["Asset_Return"]
    )

    invested_days = (data["Position"] != 0).sum()
    exposure = invested_days / len(data)

    short_days = (data["Position"] < 0).sum()

    start_date = data.index[0]
    end_date = data.index[-1]

    metrics = {
        "buy_hold_return": data["Cum_Asset_Return"].iloc[-1] - 1,
        "strategy_return": data["Cum_Strategy_Return"].iloc[-1] - 1,
        "annualized_return": calculate_cagr(
            data["Cum_Strategy_Return"].iloc[-1],
            start_date,
            end_date,
        ),
        "buy_hold_cagr": calculate_cagr(
            data["Cum_Asset_Return"].iloc[-1],
            start_date,
            end_date,
        ),
        "max_drawdown": max_drawdown,
        "buy_hold_max_drawdown": buy_hold_max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "buy_hold_sharpe_ratio": buy_hold_sharpe_ratio,
        "total_trades": data["Trade"].sum(),
        "invested_days": invested_days,
        "exposure": exposure,
        "short_days": short_days,
        "short_borrow_rate": short_borrow_rate,
        "total_borrow_cost": data["Short_Borrow_Cost"].sum(),
    }

    return metrics, data
