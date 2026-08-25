"""Report fixed buy-and-hold benchmarks for the OKX spot research universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT_DIR / "user_data" / "configs" / "okx_spot_cross_section_4h_research.example.json"
)
DEFAULT_DATA_DIR = ROOT_DIR / "user_data" / "data" / "okx"
PERIODS = {
    "development": ("2025-02-03T00:00:00Z", "2026-03-01T00:00:00Z"),
    "validation": ("2026-03-01T00:00:00Z", "2026-08-25T00:00:00Z"),
}
COST_RATES = (0.0, 0.0015)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def load_price_panels(config_path: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load each configured pair's one-hour open and close series into common panels."""
    config = json.loads(config_path.read_text())
    handler = get_datahandler(data_dir, "feather")
    open_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}

    for pair in config["exchange"]["pair_whitelist"]:
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
        raise ValueError("No common one-hour timestamps exist across the configured universe.")
    return opens.loc[common_index], closes.loc[common_index]


def period_price_ratio(
    opens: pd.DataFrame, closes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Buy at the period start open and liquidate at the last completed close before its end."""
    exit_prices = closes[(closes.index >= start) & (closes.index < end)]
    if start not in opens.index or exit_prices.empty:
        raise ValueError(f"No common data is available from {start} to {end}.")
    return exit_prices.iloc[-1].div(opens.loc[start])


def buy_and_hold_return(price_ratio: float, cost_rate: float) -> float:
    """Apply an entry and exit one-way cost to a buy-and-hold gross price ratio."""
    return ((1.0 - cost_rate) * price_ratio * (1.0 - cost_rate) - 1.0) * 100


def build_results(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Calculate fixed BTC, equal-weight universe, and cash benchmarks."""
    rows: list[dict[str, float | str]] = []
    for period_name, (start_value, end_value) in PERIODS.items():
        price_ratios = period_price_ratio(
            opens, closes, pd.Timestamp(start_value), pd.Timestamp(end_value)
        )
        equal_weight_ratio = float(price_ratios.mean())
        btc_ratio = float(price_ratios["BTC/USDT"])
        for cost_rate in COST_RATES:
            rows.extend(
                [
                    {
                        "period": period_name,
                        "benchmark": "BTC buy-and-hold",
                        "one_way_cost_pct": cost_rate * 100,
                        "return_pct": buy_and_hold_return(btc_ratio, cost_rate),
                    },
                    {
                        "period": period_name,
                        "benchmark": "20-pair equal-weight buy-and-hold",
                        "one_way_cost_pct": cost_rate * 100,
                        "return_pct": buy_and_hold_return(equal_weight_ratio, cost_rate),
                    },
                    {
                        "period": period_name,
                        "benchmark": "USDT cash",
                        "one_way_cost_pct": cost_rate * 100,
                        "return_pct": 0.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the frozen benchmark report."""
    arguments = parse_arguments()
    results = build_results(*load_price_panels(arguments.config, arguments.data_dir))
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Base cost is 0.150% per side: 0.100% taker fee plus 0.050% slippage.")


if __name__ == "__main__":
    main()
