"""Direction classifier with normalized volatility and price-position features."""

from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, timeframe_to_minutes


class OkxSpotFreqaiDirectionRegimeStrategy(IStrategy):
    """Long-only 24-hour direction classifier with one market-regime feature family."""

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
        "entry": "limit", "exit": "limit", "stoploss": "market", "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        """Create baseline and normalized volatility/price-position features."""
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)

        atr = ta.ATR(dataframe, timeperiod=period)
        bands = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        band_range = bands["upper"] - bands["lower"]
        dataframe["%-atr_ratio-period"] = atr / dataframe["close"]
        dataframe["%-bb_width-period"] = band_range / bands["mid"]
        dataframe["%-bb_position-period"] = (dataframe["close"] - bands["lower"]) / band_range
        dataframe["%-return_volatility-period"] = dataframe["close"].pct_change().rolling(period).std()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-return-1"] = dataframe["close"].pct_change()
        dataframe["%-volume"] = dataframe["volume"]
        dataframe["%-close"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-hour"] = dataframe["date"].dt.hour
        dataframe["%-weekday"] = dataframe["date"].dt.dayofweek
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        future_return = dataframe["close"].shift(-horizon) / dataframe["close"] - 1
        self.freqai.class_names = ["down", "neutral", "up"]
        dataframe["&-direction"] = "neutral"
        dataframe.loc[future_return > self.direction_threshold, "&-direction"] = "up"
        dataframe.loc[future_return < -self.direction_threshold, "&-direction"] = "down"
        dataframe.loc[future_return.isna(), "&-direction"] = np.nan
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [dataframe["do_predict"] == 1, dataframe["&-direction"] == "up", dataframe["ema_fast"] > dataframe["ema_slow"], dataframe["rsi"].between(50, 70), dataframe["adx"] > 20, dataframe["volume"] > dataframe["volume_mean"], dataframe["volume"] > 0]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (1, "freqai_direction_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> str | None:
        label_periods = self.freqai_info["feature_parameters"]["label_period_candles"]
        horizon = timedelta(minutes=label_periods * timeframe_to_minutes(self.timeframe))
        if current_time >= trade.open_date_utc + horizon:
            return "label_horizon_elapsed"
        return None
