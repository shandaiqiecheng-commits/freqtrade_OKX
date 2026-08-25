"""OKX spot FreqAI strategy skeleton with technical-entry filters.

This strategy is intended for historical research only until it has passed the
project's backtest and dry-run acceptance criteria.
"""

from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class OkxSpotFreqaiHybridStrategy(IStrategy):
    """Long-only spot strategy: FreqAI return forecast gated by technical signals."""

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 400

    minimal_roi = {"0": 0.03}
    stoploss = -0.05
    trailing_stop = False
    # Control backtest: isolate the effect of entries by using only ROI and stoploss exits.
    use_exit_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """Create past-and-present-only features for FreqAI expansion."""
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Add non-period-expanded market features."""
        dataframe["%-return-1"] = dataframe["close"].pct_change()
        dataframe["%-volume"] = dataframe["volume"]
        dataframe["%-close"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Add calendar features once after FreqAI feature expansion."""
        dataframe["%-hour"] = dataframe["date"].dt.hour
        dataframe["%-weekday"] = dataframe["date"].dt.dayofweek
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """Predict future return; its sign is the strategy's direction forecast."""
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-future_return"] = dataframe["close"].shift(-horizon) / dataframe["close"] - 1
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Run FreqAI then retain technical indicators for transparent trade filters."""
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()

        dataframe["ai_entry_threshold"] = (
            dataframe["&-future_return_mean"] + dataframe["&-future_return_std"]
        )
        dataframe["ai_exit_threshold"] = (
            dataframe["&-future_return_mean"] - dataframe["&-future_return_std"]
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter only when the return forecast and all technical filters agree."""
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return"] > dataframe["ai_entry_threshold"],
            dataframe["ema_fast"] > dataframe["ema_slow"],
            dataframe["rsi"].between(50, 70),
            dataframe["adx"] > 20,
            dataframe["volume"] > dataframe["volume_mean"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "freqai_return_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Leave exit signals disabled for the entry-quality control backtest."""
        return dataframe
