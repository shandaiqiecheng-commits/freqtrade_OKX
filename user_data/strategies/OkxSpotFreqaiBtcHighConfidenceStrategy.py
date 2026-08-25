"""BTC-only high-confidence validation strategy."""

from functools import reduce

from pandas import DataFrame

from OkxSpotFreqaiDirectionRegimeStrategy import OkxSpotFreqaiDirectionRegimeStrategy


class OkxSpotFreqaiBtcHighConfidenceStrategy(OkxSpotFreqaiDirectionRegimeStrategy):
    """Validate the fixed 0.95 up-probability threshold selected on development data."""

    minimum_up_probability = 0.95

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter BTC only when the fixed high-confidence and technical filters agree."""
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-direction"] == "up",
            dataframe["up"] >= self.minimum_up_probability,
            dataframe["ema_fast"] > dataframe["ema_slow"],
            dataframe["rsi"].between(50, 70),
            dataframe["adx"] > 20,
            dataframe["volume"] > dataframe["volume_mean"],
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "freqai_btc_high_confidence")
        return dataframe
