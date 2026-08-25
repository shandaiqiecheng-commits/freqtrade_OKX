"""Collect one-second OKX spot order-book and trade snapshots without trading."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import websockets


OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "microstructure" / "okx"
DEFAULT_PAIRS = ("BTC-USDT", "ETH-USDT")
LOGGER = logging.getLogger(__name__)


@dataclass
class SecondSnapshot:
    """Mutable state for one instrument and one exchange-timestamp second."""

    second_ms: int
    bids: list[list[str]] = field(default_factory=list)
    asks: list[list[str]] = field(default_factory=list)
    book_timestamp_ms: int | None = None
    book_sequence_id: int | None = None
    buy_base_volume: float = 0.0
    sell_base_volume: float = 0.0
    buy_quote_volume: float = 0.0
    sell_quote_volume: float = 0.0
    buy_trade_count: int = 0
    sell_trade_count: int = 0

    def set_book(self, book: dict) -> None:
        """Keep the latest five-level book received in this second."""
        self.bids = book["bids"]
        self.asks = book["asks"]
        self.book_timestamp_ms = int(book["ts"])
        self.book_sequence_id = int(book["seqId"])

    def add_trade(self, trade: dict) -> None:
        """Accumulate traded base and quote volume using OKX aggressor side semantics."""
        size = float(trade["sz"])
        notional = float(trade["px"]) * size
        count = int(trade["count"])
        if trade["side"] == "buy":
            self.buy_base_volume += size
            self.buy_quote_volume += notional
            self.buy_trade_count += count
        else:
            self.sell_base_volume += size
            self.sell_quote_volume += notional
            self.sell_trade_count += count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--max-seconds", type=int, help="Optional bounded run for validation.")
    return parser.parse_args()


def create_subscription() -> str:
    """Build the fixed public subscriptions for BTC and ETH spot microstructure."""
    arguments = [
        {"channel": channel, "instId": pair}
        for pair in DEFAULT_PAIRS
        for channel in ("books5", "trades")
    ]
    return json.dumps({"op": "subscribe", "args": arguments})


def snapshot_row(pair: str, snapshot: SecondSnapshot) -> dict:
    """Convert a second aggregate into one durable, self-describing JSON record."""
    best_bid_price = float(snapshot.bids[0][0]) if snapshot.bids else None
    best_ask_price = float(snapshot.asks[0][0]) if snapshot.asks else None
    midpoint = None
    spread_bps = None
    if best_bid_price is not None and best_ask_price is not None:
        midpoint = (best_bid_price + best_ask_price) / 2
        spread_bps = (best_ask_price - best_bid_price) / midpoint * 10_000

    return {
        "exchange": "okx",
        "instrument": pair,
        "bucket_start_utc": datetime.fromtimestamp(snapshot.second_ms / 1000, UTC).isoformat(),
        "book_event_timestamp_ms": snapshot.book_timestamp_ms,
        "book_sequence_id": snapshot.book_sequence_id,
        "bids": snapshot.bids,
        "asks": snapshot.asks,
        "best_bid_price": best_bid_price,
        "best_ask_price": best_ask_price,
        "midpoint": midpoint,
        "spread_bps": spread_bps,
        "buy_base_volume": snapshot.buy_base_volume,
        "sell_base_volume": snapshot.sell_base_volume,
        "buy_quote_volume": snapshot.buy_quote_volume,
        "sell_quote_volume": snapshot.sell_quote_volume,
        "buy_trade_count": snapshot.buy_trade_count,
        "sell_trade_count": snapshot.sell_trade_count,
    }


def append_snapshot(output_dir: Path, pair: str, snapshot: SecondSnapshot) -> None:
    """Append one snapshot to its exchange-date and instrument JSON Lines partition."""
    bucket_date = datetime.fromtimestamp(snapshot.second_ms / 1000, UTC).date().isoformat()
    filename = output_dir / pair.replace("-", "_") / f"{bucket_date}.jsonl"
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot_row(pair, snapshot), separators=(",", ":")) + "\n")


def advance_snapshot(
    pair: str,
    timestamp_ms: int,
    states: dict[str, SecondSnapshot],
    output_dir: Path,
) -> SecondSnapshot | None:
    """Flush a completed second before accepting an event from a newer second."""
    event_second_ms = timestamp_ms // 1000 * 1000
    current = states.get(pair)
    if current is None:
        current = SecondSnapshot(second_ms=event_second_ms)
        states[pair] = current
        return current
    if event_second_ms < current.second_ms:
        LOGGER.warning("Discarded out-of-order %s event at %s.", pair, timestamp_ms)
        return None
    if event_second_ms > current.second_ms:
        append_snapshot(output_dir, pair, current)
        current = SecondSnapshot(second_ms=event_second_ms)
        states[pair] = current
    return current


def process_message(message: dict, states: dict[str, SecondSnapshot], output_dir: Path) -> None:
    """Route an OKX public-channel message into its current one-second aggregate."""
    argument = message.get("arg", {})
    channel = argument.get("channel")
    pair = argument.get("instId")
    if pair not in DEFAULT_PAIRS or channel not in {"books5", "trades"}:
        return
    for item in message.get("data", []):
        snapshot = advance_snapshot(pair, int(item["ts"]), states, output_dir)
        if snapshot is None:
            continue
        if channel == "books5":
            snapshot.set_book(item)
        else:
            snapshot.add_trade(item)


async def collect_connection(
    arguments: argparse.Namespace,
    states: dict[str, SecondSnapshot],
    deadline: float | None,
) -> None:
    """Collect until a disconnect or optional deadline, then let the caller reconnect."""
    async with websockets.connect(
        OKX_PUBLIC_WS,
        proxy=arguments.proxy,
        ping_interval=20,
        ping_timeout=20,
    ) as socket:
        await socket.send(create_subscription())
        while True:
            timeout = (
                None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            )
            if timeout == 0:
                return
            raw_message = await asyncio.wait_for(socket.recv(), timeout=timeout)
            process_message(json.loads(raw_message), states, arguments.output_dir)


async def run_collector(arguments: argparse.Namespace) -> None:
    """Reconnect indefinitely with bounded backoff and flush the last complete state on exit."""
    if arguments.max_seconds is not None and arguments.max_seconds < 1:
        raise ValueError("--max-seconds must be at least one.")

    states: dict[str, SecondSnapshot] = {}
    deadline = None
    if arguments.max_seconds is not None:
        deadline = asyncio.get_running_loop().time() + arguments.max_seconds
    retry_delay = 1.0
    try:
        while deadline is None or asyncio.get_running_loop().time() < deadline:
            try:
                await collect_connection(arguments, states, deadline)
                retry_delay = 1.0
            except TimeoutError as error:
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    break
                LOGGER.warning(
                    "OKX WebSocket disconnected: %s. Retrying in %.1fs.", error, retry_delay
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
            except (OSError, websockets.WebSocketException) as error:
                LOGGER.warning(
                    "OKX WebSocket disconnected: %s. Retrying in %.1fs.", error, retry_delay
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
    finally:
        for pair, snapshot in states.items():
            append_snapshot(arguments.output_dir, pair, snapshot)


def main() -> None:
    """Start the public, read-only microstructure collector."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_collector(parse_arguments()))


if __name__ == "__main__":
    main()
