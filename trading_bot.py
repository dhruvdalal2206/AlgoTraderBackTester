"""
Algorithmic Trading Bot - S&P 500 | 20 SMA Touch Strategy
=========================================================
Strategy:
  - Universe : S&P 500 stocks
  - Entry     : Stock has moved ±2 % intraday AND price is touching the 20-period SMA
  - Stop-loss : 1 % below entry price
  - Target 1  : +1.5 % → sell 50 % of position
  - Target 2  : another +1.5 % from T1 (i.e. +3 % from entry) → sell remaining 50 %
  - Emergency : remaining 50 % also sold if stop-loss is hit after T1 triggered

Scheduling:
  - Runs every minute during NYSE market hours
  - Designed to be deployed on Render (cron or always-on web service)
  - Timezone: IST (Asia/Kolkata) → NYSE opens at 19:30 IST, closes at 02:00 IST next day

Requirements:
  pip install alpaca-py pandas numpy pytz apscheduler requests
"""

import os
import io
import logging
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


import numpy as np
import pandas as pd
import pytz
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ─────────────────────────────────────────────
# CONFIGURATION  (set via environment variables)
# ─────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_API_KEY", "YOUR_PAPER_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET", "YOUR_PAPER_API_SECRET")
BASE_URL   = "https://paper-api.alpaca.markets"   # paper trading endpoint

# Risk parameters
PRICE_CHANGE_THRESHOLD = 0.02   # 2 % intraday move required
SMA_PERIOD             = 20     # 20-bar SMA
SMA_TOUCH_PCT          = 0.005  # price must be within 0.5 % of SMA to count as "touching"
STOP_LOSS_PCT          = 0.01   # 1 % stop-loss from entry
TARGET1_PCT            = 0.015  # +1.5 % → sell 50 %
TARGET2_PCT            = 0.015  # another +1.5 % from T1 → sell remaining 50 %
POSITION_SIZE_USD      = 1000   # USD to allocate per trade (adjust to your account size)
MAX_OPEN_POSITIONS     = 5      # max simultaneous positions

IST = pytz.timezone("Asia/Kolkata")
NYSE_TZ = pytz.timezone("America/New_York")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client  = StockHistoricalDataClient(API_KEY, API_SECRET)

# ─────────────────────────────────────────────
# IN-MEMORY STATE  (survives within a single run session)
# ─────────────────────────────────────────────
# Structure: { symbol: { entry_price, stop_loss, target1, target2,
#                        t1_hit, qty_total, qty_remaining } }
open_trades: dict = {}


# ─────────────────────────────────────────────
# S&P 500 SYMBOL LIST
# ─────────────────────────────────────────────
def get_sp500_symbols() -> list[str]:
    """Load S&P 500 tickers from local sp500.txt file."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), "sp500.txt")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
            log.info(f"Loaded {len(symbols)} S&P 500 symbols from local sp500.txt")
            return symbols
        else:
            raise FileNotFoundError("Local sp500.txt file not found.")
    except Exception as e:
        log.error(f"Failed to load S&P 500 list from local file: {e}. Using fallback list.")
        # Fallback: top 20 liquid S&P 500 names
        return [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B",
            "JPM","UNH","V","XOM","JNJ","PG","MA","HD","CVX","MRK",
            "ABBV","LLY"
        ]



SP500_SYMBOLS = get_sp500_symbols()


# ─────────────────────────────────────────────
# MARKET HOURS CHECK
# ─────────────────────────────────────────────
def is_market_open() -> bool:
    """Return True if NYSE is currently open (via Alpaca clock endpoint)."""
    try:
        clock = trade_client.get_clock()
        return clock.is_open
    except Exception as e:
        log.warning(f"Could not check market clock: {e}")
        return False


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
def get_bars(symbol: str, limit: int = 25, timeframe: TimeFrame = TimeFrame.Minute) -> pd.DataFrame:
    """Fetch recent OHLCV bars for a symbol."""
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        bars = data_client.get_stock_bars(req).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        bars = bars.sort_index()
        return bars
    except Exception as e:
        log.debug(f"Bar fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def compute_sma(bars: pd.DataFrame, period: int = 20) -> float | None:
    """Compute the simple moving average of close prices."""
    closes = bars["close"].values
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def get_daily_change(symbol: str) -> float | None:
    """
    Return today's intraday percentage change:
    (current_price - open_price) / open_price
    """
    try:
        bars = get_bars(symbol, limit=390, timeframe=TimeFrame.Minute)  # full day
        if bars.empty or len(bars) < 2:
            return None
        open_price  = float(bars.iloc[0]["open"])
        last_price  = float(bars.iloc[-1]["close"])
        if open_price == 0:
            return None
        return (last_price - open_price) / open_price
    except Exception as e:
        log.debug(f"Daily change error for {symbol}: {e}")
        return None


def get_current_price(symbol: str) -> float | None:
    """Get the latest close price from 1-minute bars."""
    bars = get_bars(symbol, limit=2, timeframe=TimeFrame.Minute)
    if bars.empty:
        return None
    return float(bars.iloc[-1]["close"])


# ─────────────────────────────────────────────
# ORDER HELPERS
# ─────────────────────────────────────────────
def place_market_order(symbol: str, qty: int, side: OrderSide) -> bool:
    """Submit a market order. Returns True on success."""
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = trade_client.submit_order(req)
        log.info(f"Order placed → {side.value.upper()} {qty} {symbol} | id={order.id}")
        return True
    except Exception as e:
        log.error(f"Order failed for {symbol}: {e}")
        return False


def get_account_buying_power() -> float:
    """Return available buying power in USD."""
    try:
        account = trade_client.get_account()
        return float(account.buying_power)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# ENTRY LOGIC
# ─────────────────────────────────────────────
def scan_for_entries():
    """Scan S&P 500 for entry signals and open new trades."""
    if len(open_trades) >= MAX_OPEN_POSITIONS:
        log.info(f"Max positions ({MAX_OPEN_POSITIONS}) reached. Skipping scan.")
        return

    buying_power = get_account_buying_power()
    if buying_power < POSITION_SIZE_USD:
        log.info(f"Insufficient buying power: ${buying_power:.2f}")
        return

    candidates = [s for s in SP500_SYMBOLS if s not in open_trades]

    for symbol in candidates:
        if len(open_trades) >= MAX_OPEN_POSITIONS:
            break

        # ── 1. Check intraday change ≥ ±2 % ──────────────────────────
        change = get_daily_change(symbol)
        if change is None or abs(change) < PRICE_CHANGE_THRESHOLD:
            continue

        # ── 2. Get 20-bar SMA on 1-min bars ──────────────────────────
        bars = get_bars(symbol, limit=SMA_PERIOD + 5, timeframe=TimeFrame.Minute)
        if bars.empty or len(bars) < SMA_PERIOD:
            continue

        sma   = compute_sma(bars, SMA_PERIOD)
        price = float(bars.iloc[-1]["close"])
        if sma is None or sma == 0:
            continue

        # ── 3. Price is "touching" the SMA (within 0.5 %) ────────────
        distance = abs(price - sma) / sma
        if distance > SMA_TOUCH_PCT:
            continue

        # ── 4. All conditions met → place BUY order ──────────────────
        qty = max(1, int(POSITION_SIZE_USD / price))
        log.info(
            f"SIGNAL → {symbol} | price={price:.2f} | SMA={sma:.2f} "
            f"| change={change*100:.2f}% | qty={qty}"
        )

        if place_market_order(symbol, qty, OrderSide.BUY):
            entry_price = price
            open_trades[symbol] = {
                "entry_price" : entry_price,
                "stop_loss"   : round(entry_price * (1 - STOP_LOSS_PCT), 4),
                "target1"     : round(entry_price * (1 + TARGET1_PCT), 4),
                "target2"     : round(entry_price * (1 + TARGET1_PCT + TARGET2_PCT), 4),
                "t1_hit"      : False,
                "qty_total"   : qty,
                "qty_remaining": qty,
                "direction"   : "up" if change > 0 else "down",
            }
            log.info(
                f"Trade opened → {symbol} | entry={entry_price:.2f} "
                f"| SL={open_trades[symbol]['stop_loss']:.2f} "
                f"| T1={open_trades[symbol]['target1']:.2f} "
                f"| T2={open_trades[symbol]['target2']:.2f}"
            )
            time.sleep(0.3)   # rate-limit courtesy


# ─────────────────────────────────────────────
# EXIT LOGIC
# ─────────────────────────────────────────────
def manage_open_trades():
    """Check all open positions for stop-loss or target hits."""
    to_close = []

    for symbol, trade in list(open_trades.items()):
        price = get_current_price(symbol)
        if price is None:
            continue

        qty_remaining = trade["qty_remaining"]
        if qty_remaining <= 0:
            to_close.append(symbol)
            continue

        # ── Stop-loss hit ─────────────────────────────────────────────
        if price <= trade["stop_loss"]:
            log.info(
                f"STOP-LOSS hit → {symbol} | price={price:.2f} "
                f"| SL={trade['stop_loss']:.2f} | selling {qty_remaining} shares"
            )
            if place_market_order(symbol, qty_remaining, OrderSide.SELL):
                to_close.append(symbol)
            continue

        # ── Target 2 hit (sell remaining 50 %) ────────────────────────
        if trade["t1_hit"] and price >= trade["target2"]:
            log.info(
                f"TARGET 2 hit → {symbol} | price={price:.2f} "
                f"| T2={trade['target2']:.2f} | selling {qty_remaining} shares"
            )
            if place_market_order(symbol, qty_remaining, OrderSide.SELL):
                to_close.append(symbol)
            continue

        # ── Target 1 hit (sell first 50 %) ────────────────────────────
        if not trade["t1_hit"] and price >= trade["target1"]:
            half_qty = max(1, qty_remaining // 2)
            log.info(
                f"TARGET 1 hit → {symbol} | price={price:.2f} "
                f"| T1={trade['target1']:.2f} | selling {half_qty} of {qty_remaining} shares"
            )
            if place_market_order(symbol, half_qty, OrderSide.SELL):
                open_trades[symbol]["t1_hit"]       = True
                open_trades[symbol]["qty_remaining"] = qty_remaining - half_qty
                # raise stop-loss to break-even after T1
                open_trades[symbol]["stop_loss"]     = trade["entry_price"]
                log.info(
                    f"Stop-loss raised to break-even ({trade['entry_price']:.2f}) for {symbol}"
                )

    # Remove closed trades
    for symbol in to_close:
        log.info(f"Position closed and removed from tracker: {symbol}")
        open_trades.pop(symbol, None)


# ─────────────────────────────────────────────
# CLOSE ALL POSITIONS BEFORE MARKET CLOSE (EOD)
# ─────────────────────────────────────────────
def close_all_positions_eod():
    """
    Liquidate all remaining tracked positions ~5 minutes before NYSE close
    to avoid overnight holds on paper account.
    """
    nyse_now = datetime.now(NYSE_TZ)
    # NYSE closes at 16:00 ET; close out at 15:55
    if nyse_now.hour == 15 and nyse_now.minute >= 55:
        log.info("EOD: Closing all open positions before market close.")
        for symbol, trade in list(open_trades.items()):
            qty = trade["qty_remaining"]
            if qty > 0:
                place_market_order(symbol, qty, OrderSide.SELL)
        open_trades.clear()


# ─────────────────────────────────────────────
# MAIN SCHEDULER TICK
# ─────────────────────────────────────────────
def run_tick():
    """Called every minute by the scheduler."""
    ist_now = datetime.now(IST)
    log.info(f"─── Tick at {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')} ───")

    if not is_market_open():
        log.info("Market is closed. Waiting...")
        return

    # 1. Manage exits first
    manage_open_trades()

    # 2. Check EOD liquidation
    close_all_positions_eod()

    # 3. Scan for new entries
    scan_for_entries()

    log.info(f"Open trades: {list(open_trades.keys()) or 'None'}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  S&P 500 | 20 SMA Touch Strategy Bot — Starting Up")
    log.info("=" * 60)
    log.info(f"  API Key   : {API_KEY[:8]}...")
    log.info(f"  Paper URL : {BASE_URL}")
    log.info(f"  Parameters:")
    log.info(f"    Price change threshold : {PRICE_CHANGE_THRESHOLD*100:.1f}%")
    log.info(f"    SMA period             : {SMA_PERIOD}")
    log.info(f"    SMA touch tolerance    : {SMA_TOUCH_PCT*100:.2f}%")
    log.info(f"    Stop-loss              : {STOP_LOSS_PCT*100:.1f}%")
    log.info(f"    Target 1               : +{TARGET1_PCT*100:.1f}% (sell 50%)")
    log.info(f"    Target 2               : +{(TARGET1_PCT+TARGET2_PCT)*100:.1f}% from entry (sell 50%)")
    log.info(f"    Position size          : ${POSITION_SIZE_USD}")
    log.info(f"    Max positions          : {MAX_OPEN_POSITIONS}")
    log.info("=" * 60)

    scheduler = BlockingScheduler(timezone=IST)

    # Run every minute during NYSE hours (IST: Mon–Fri 19:30–02:00+1)
    # APScheduler cron: NYSE open = 19:30 IST, close = 02:00 IST next day
    # We run 19:25–02:05 IST to catch pre/post boundary
    scheduler.add_job(
        run_tick,
        trigger="cron",
        day_of_week="mon-fri",
        hour="19-23",          # 19:25 to 23:59 IST
        minute="*",
        id="market_hours_evening",
    )
    scheduler.add_job(
        run_tick,
        trigger="cron",
        day_of_week="tue-sat",
        hour="0,1",            # 00:00 to 01:59 IST (next calendar day)
        minute="*",
        id="market_hours_night",
    )

    # Also run once immediately for testing
    log.info("Running one immediate tick for connectivity check...")
    run_tick()

    log.info("Scheduler started. Bot is live. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped by user.")
