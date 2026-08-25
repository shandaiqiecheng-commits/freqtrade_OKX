"""Backtest a daily OKX spot cross-sectional short-term reversal baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from backtest_okx_cross_sectional_momentum import (
    DEFAULT_CONFIG,
    DEFAULT_COST_RATES,
    DEFAULT_DATA_DIR,
    PERIODS,
    format_results,
    load_price_panels,
    run_period,
)


REVERSAL_HOURS = 24
REBALANCE_HOURS = 24
TOP_N = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def build_targets(opens: pd.DataFrame, closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the three most negative completed 24-hour returns at each daily open."""
    completed_close = closes.shift(1)
    returns = completed_close.div(completed_close.shift(REVERSAL_HOURS)).sub(1.0)
    rebalance_index = opens.index[opens.index.hour == 0]
    targets = pd.DataFrame(0.0, index=rebalance_index, columns=opens.columns)

    for timestamp, row in returns.loc[rebalance_index].iterrows():
        selected = row[row < 0].nsmallest(TOP_N).index
        if len(selected):
            targets.loc[timestamp, selected] = 1.0 / len(selected)

    next_open = opens.shift(-REBALANCE_HOURS).reindex(rebalance_index)
    holding_returns = next_open.div(opens.reindex(rebalance_index)).sub(1.0)
    return targets, holding_returns


def main() -> None:
    """Run the fixed daily short-term reversal baseline in both research periods."""
    arguments = parse_arguments()
    opens, closes = load_price_panels(arguments.config, arguments.data_dir)
    targets, holding_returns = build_targets(opens, closes)
    results = [
        run_period(name, targets, holding_returns, cost_rate, REBALANCE_HOURS)
        for name in PERIODS
        for cost_rate in DEFAULT_COST_RATES
    ]
    print(format_results(results))
    print("Base cost is 0.150% per traded notional: 0.100% taker fee plus 0.050% slippage.")


if __name__ == "__main__":
    main()
