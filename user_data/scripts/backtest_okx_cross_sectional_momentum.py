"""Backtest a 4-hour OKX spot cross-sectional momentum baseline from local OHLCV data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT_DIR / "user_data" / "configs" / "okx_spot_cross_section_4h_research.example.json"
)
DEFAULT_DATA_DIR = ROOT_DIR / "user_data" / "data" / "okx"
MOMENTUM_HOURS = 168
REBALANCE_HOURS = 4
TOP_N = 3
DEFAULT_COST_RATES = (0.0, 0.0005, 0.0010, 0.0015, 0.0020)
PERIODS = {
    "development": ("2025-02-03T00:00:00Z", "2026-03-01T00:00:00Z"),
    "validation": ("2026-03-01T00:00:00Z", "2026-08-25T00:00:00Z"),
}


@dataclass(frozen=True)
class BacktestResult:
    period: str
    cost_rate: float
    return_pct: float
    max_drawdown_pct: float
    sharpe: float
    gross_turnover: float
    active_rebalances: int
    average_positions: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def load_price_panels(config_path: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned one-hour open and close price panels for the configured pair universe."""
    config = json.loads(config_path.read_text())
    pairs = config["exchange"]["pair_whitelist"]
    handler = get_datahandler(data_dir, "feather")
    open_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}

    for pair in pairs:
        candles = handler.ohlcv_load(
            pair,
            "1h",
            CandleType.SPOT,
            timerange=None,
            fill_missing=False,
        )
        if candles.empty:
            raise ValueError(f"No 1h data found for {pair}.")
        candles = candles.set_index("date")
        open_prices[pair] = candles["open"]
        close_prices[pair] = candles["close"]

    opens = pd.DataFrame(open_prices).sort_index()
    closes = pd.DataFrame(close_prices).sort_index()
    common_index = opens.dropna().index.intersection(closes.dropna().index)
    if common_index.empty:
        raise ValueError("OHLCV panels have no common timestamps across the configured universe.")
    return opens.loc[common_index], closes.loc[common_index]


def build_targets(opens: pd.DataFrame, closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank completed seven-day returns at each 4-hour execution point."""
    completed_close = closes.shift(1)
    momentum = completed_close.div(completed_close.shift(MOMENTUM_HOURS)).sub(1.0)
    rebalance_index = opens.index[opens.index.hour % REBALANCE_HOURS == 0]
    targets = pd.DataFrame(0.0, index=rebalance_index, columns=opens.columns)

    for timestamp, row in momentum.loc[rebalance_index].iterrows():
        selected = row[row > 0].nlargest(TOP_N).index
        if len(selected):
            targets.loc[timestamp, selected] = 1.0 / len(selected)

    next_open = opens.shift(-REBALANCE_HOURS).reindex(rebalance_index)
    holding_returns = next_open.div(opens.reindex(rebalance_index)).sub(1.0)
    return targets, holding_returns


def run_period(
    name: str,
    targets: pd.DataFrame,
    holding_returns: pd.DataFrame,
    cost_rate: float,
    rebalance_hours: int = REBALANCE_HOURS,
) -> BacktestResult:
    """Simulate a cash-starting period with proportional one-way transaction costs."""
    start, end = (pd.Timestamp(value) for value in PERIODS[name])
    period_targets = targets[(targets.index >= start) & (targets.index < end)].copy()
    period_returns = holding_returns.reindex(period_targets.index)
    valid_holding_periods = period_returns.notna().all(axis=1)
    period_targets = period_targets[valid_holding_periods]
    period_returns = period_returns[valid_holding_periods]

    previous_weights = pd.Series(0.0, index=period_targets.columns)
    equity = 1.0
    equity_curve: list[float] = []
    net_returns: list[float] = []
    traded_weights: list[float] = []
    active_positions: list[int] = []

    for timestamp, target_weights in period_targets.iterrows():
        gross_traded_weight = (target_weights - previous_weights).abs().sum()
        gross_return = float((target_weights * period_returns.loc[timestamp]).sum())
        net_return = (1.0 - gross_traded_weight * cost_rate) * (1.0 + gross_return) - 1.0
        equity *= 1.0 + net_return
        equity_curve.append(equity)
        net_returns.append(net_return)
        traded_weights.append(gross_traded_weight)
        active_positions.append(int((target_weights > 0).sum()))
        previous_weights = target_weights

    closing_turnover = previous_weights.abs().sum()
    equity *= 1.0 - closing_turnover * cost_rate
    traded_weights.append(closing_turnover)

    returns = pd.Series(net_returns, dtype=float)
    curve = pd.Series(equity_curve, dtype=float)
    drawdown = curve.div(curve.cummax()).sub(1.0).min()
    annualization = np.sqrt(365 * 24 / rebalance_hours)
    sharpe = (
        0.0 if returns.std(ddof=0) == 0 else annualization * returns.mean() / returns.std(ddof=0)
    )
    return BacktestResult(
        period=name,
        cost_rate=cost_rate,
        return_pct=(equity - 1.0) * 100,
        max_drawdown_pct=drawdown * 100,
        sharpe=sharpe,
        gross_turnover=float(sum(traded_weights)),
        active_rebalances=sum(position_count > 0 for position_count in active_positions),
        average_positions=float(np.mean(active_positions)),
    )


def format_results(results: list[BacktestResult]) -> str:
    """Render a compact report for all fixed periods and cost scenarios."""
    rows = [
        {
            "period": result.period,
            "one_way_cost_pct": result.cost_rate * 100,
            "return_pct": result.return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe": result.sharpe,
            "gross_turnover": result.gross_turnover,
            "active_rebalances": result.active_rebalances,
            "average_positions": result.average_positions,
        }
        for result in results
    ]
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3f}")


def main() -> None:
    """Run the frozen momentum baseline across development and validation periods."""
    arguments = parse_arguments()
    opens, closes = load_price_panels(arguments.config, arguments.data_dir)
    targets, holding_returns = build_targets(opens, closes)
    results = [
        run_period(name, targets, holding_returns, cost_rate)
        for name in PERIODS
        for cost_rate in DEFAULT_COST_RATES
    ]
    print(format_results(results))
    print("Base cost is 0.150% per traded notional: 0.100% taker fee plus 0.050% slippage.")


if __name__ == "__main__":
    main()
