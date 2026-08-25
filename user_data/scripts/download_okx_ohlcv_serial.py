"""Download one OKX spot OHLCV series without Freqtrade's parallel history requests."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import ccxt

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_msecs


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "okx"


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp as a UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "Timestamp must include a timezone, for example 2025-01-01T00:00:00Z."
        )
    return parsed.astimezone(UTC)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, help="OKX spot pair, for example DOGE/USDT.")
    parser.add_argument(
        "--timeframe", required=True, help="Freqtrade timeframe, for example 15m or 1h."
    )
    parser.add_argument(
        "--start", required=True, type=parse_datetime, help="Inclusive UTC ISO-8601 start."
    )
    parser.add_argument(
        "--end", required=True, type=parse_datetime, help="Exclusive UTC ISO-8601 end."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--checkpoint-pages",
        type=int,
        default=20,
        help="Write accumulated candles after this many successful API pages.",
    )
    return parser.parse_args()


def load_resume_cursor(data_handler, pair: str, timeframe: str, start_ms: int) -> int:
    """Resume only when local candles cover the request start without a gap."""
    existing = data_handler.ohlcv_load(
        pair, timeframe, CandleType.SPOT, timerange=None, fill_missing=False
    )
    if existing.empty:
        return start_ms

    timeframe_ms = timeframe_to_msecs(timeframe)
    dates_ms = [
        int(date.timestamp() * 1000)
        for date in existing["date"]
        if date.timestamp() * 1000 >= start_ms
    ]
    if not dates_ms or dates_ms[0] != start_ms:
        return start_ms

    expected_count = ((dates_ms[-1] - start_ms) // timeframe_ms) + 1
    if len(dates_ms) != expected_count:
        return start_ms
    return dates_ms[-1] + timeframe_ms


def store_candles(data_handler, pair: str, timeframe: str, raw_candles: list[list]) -> None:
    """Convert CCXT candles with Freqtrade's converter and write the standard Feather format."""
    dataframe = ohlcv_to_dataframe(
        raw_candles,
        timeframe,
        pair,
        fill_missing=False,
        drop_incomplete=False,
    )
    data_handler.ohlcv_store(pair, timeframe, dataframe, CandleType.SPOT)


def fetch_page(exchange, arguments: argparse.Namespace, cursor_ms: int) -> list[list]:
    """Fetch one page with bounded backoff for rate and transient network errors."""
    for attempt in range(1, arguments.max_retries + 1):
        try:
            return exchange.fetch_ohlcv(
                arguments.pair, arguments.timeframe, since=cursor_ms, limit=100
            )
        except (ccxt.RateLimitExceeded, ccxt.NetworkError) as error:
            if attempt == arguments.max_retries:
                raise
            wait_seconds = arguments.delay_seconds * (2**attempt)
            print(
                f"{type(error).__name__}. Waiting {wait_seconds:.1f}s before retry {attempt + 1}."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Unreachable retry state.")


def load_markets(exchange, arguments: argparse.Namespace) -> None:
    """Load the OKX market catalog with the same bounded network retry policy."""
    for attempt in range(1, arguments.max_retries + 1):
        try:
            exchange.load_markets()
            return
        except (ccxt.RateLimitExceeded, ccxt.NetworkError) as error:
            if attempt == arguments.max_retries:
                raise
            wait_seconds = arguments.delay_seconds * (2**attempt)
            print(
                f"{type(error).__name__}. Waiting {wait_seconds:.1f}s before retry {attempt + 1}."
            )
            time.sleep(wait_seconds)


def serial_download(arguments: argparse.Namespace) -> None:
    """Fetch, checkpoint, and validate one bounded OHLCV series."""
    if arguments.start >= arguments.end:
        raise ValueError("--start must be earlier than --end.")
    if arguments.delay_seconds <= 0:
        raise ValueError("--delay-seconds must be greater than zero.")
    if arguments.max_retries < 1:
        raise ValueError("--max-retries must be at least one.")
    if arguments.checkpoint_pages < 1:
        raise ValueError("--checkpoint-pages must be at least one.")

    timeframe_ms = timeframe_to_msecs(arguments.timeframe)
    start_ms = int(arguments.start.timestamp() * 1000)
    end_ms = int(arguments.end.timestamp() * 1000)
    data_handler = get_datahandler(arguments.data_dir, "feather")
    cursor_ms = load_resume_cursor(data_handler, arguments.pair, arguments.timeframe, start_ms)

    exchange = ccxt.okx(
        {
            "httpsProxy": arguments.proxy,
            "enableRateLimit": True,
            "rateLimit": int(arguments.delay_seconds * 1000),
        }
    )

    if cursor_ms >= end_ms:
        print(f"{arguments.pair} {arguments.timeframe} already reaches the requested end time.")
        return

    load_markets(exchange, arguments)
    if arguments.pair not in exchange.markets:
        raise ValueError(f"OKX spot market is not available: {arguments.pair}")

    stored = data_handler.ohlcv_load(
        arguments.pair, arguments.timeframe, CandleType.SPOT, timerange=None, fill_missing=False
    )
    candles_by_timestamp = {
        candle[0]: candle
        for candle in [
            [int(row.date.timestamp() * 1000), row.open, row.high, row.low, row.close, row.volume]
            for row in stored.itertuples(index=False)
        ]
    }
    pages_since_checkpoint = 0

    while cursor_ms < end_ms:
        page = fetch_page(exchange, arguments, cursor_ms)
        page = [candle for candle in page if start_ms <= candle[0] < end_ms]
        if not page:
            timestamp = datetime.fromtimestamp(cursor_ms / 1000, UTC)
            raise RuntimeError(f"OKX returned no candles at {timestamp!s}.")

        candles_by_timestamp.update({candle[0]: candle for candle in page})
        pages_since_checkpoint += 1
        cursor_ms = page[-1][0] + timeframe_ms
        if pages_since_checkpoint == arguments.checkpoint_pages or cursor_ms >= end_ms:
            raw_candles = sorted(candles_by_timestamp.values(), key=lambda candle: candle[0])
            store_candles(data_handler, arguments.pair, arguments.timeframe, raw_candles)
            print(f"Stored through {datetime.fromtimestamp(page[-1][0] / 1000, UTC).isoformat()}.")
            pages_since_checkpoint = 0
        if cursor_ms < end_ms:
            time.sleep(arguments.delay_seconds)

    final_data = data_handler.ohlcv_load(
        arguments.pair, arguments.timeframe, CandleType.SPOT, timerange=None, fill_missing=False
    )
    matching = final_data[
        (final_data["date"] >= arguments.start) & (final_data["date"] < arguments.end)
    ]
    expected_count = (end_ms - start_ms) // timeframe_ms
    if len(matching) != expected_count:
        raise RuntimeError(
            f"Downloaded {len(matching)} candles, but {expected_count} are required "
            "for a complete range."
        )
    print(
        f"Completed {arguments.pair} {arguments.timeframe}: {len(matching)} candles "
        "in requested range."
    )


if __name__ == "__main__":
    serial_download(parse_arguments())
