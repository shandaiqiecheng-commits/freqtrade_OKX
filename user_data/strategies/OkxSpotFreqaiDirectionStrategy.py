"""OKX spot FreqAI direction-classification research strategy."""

from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, timeframe_to_minutes


class OkxSpotFreqaiDirectionStrategy(IStrategy):
    """Long-only spot strategy using a 24-hour FreqAI direction classification target."""

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = 400

    minimal_roi = {"0": 0.03}
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True

    direction_threshold = 0.005

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
        """Label the 24-hour return as up, neutral, or down."""
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        future_return = dataframe["close"].shift(-horizon) / dataframe["close"] - 1

        self.freqai.class_names = ["down", "neutral", "up"]
        dataframe["&-direction"] = "neutral"
        dataframe.loc[future_return > self.direction_threshold, "&-direction"] = "up"
        dataframe.loc[future_return < -self.direction_threshold, "&-direction"] = "down"
        dataframe.loc[future_return.isna(), "&-direction"] = np.nan
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Run FreqAI then retain technical indicators for transparent trade filters."""
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter only when the up class and technical filters agree."""
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-direction"] == "up",
            dataframe["ema_fast"] > dataframe["ema_slow"],
            dataframe["rsi"].between(50, 70),
            dataframe["adx"] > 20,
            dataframe["volume"] > dataframe["volume_mean"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "freqai_direction_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Keep dataframe-based exit signals disabled for the horizon control backtest."""
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        """Exit when the holding period reaches the FreqAI target horizon."""
        label_periods = self.freqai_info["feature_parameters"]["label_period_candles"]
        horizon = timedelta(minutes=label_periods * timeframe_to_minutes(self.timeframe))

        if current_time >= trade.open_date_utc + horizon:
            return "label_horizon_elapsed"
        return None
