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
import yfinance as yf
import logging
import time
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

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
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, PositionSide
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ─────────────────────────────────────────────
# CONFIGURATION  (set via environment variables)
# ─────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_API_KEY", "YOUR_PAPER_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET", "YOUR_PAPER_API_SECRET")
BASE_URL   = "https://paper-api.alpaca.markets"   # paper trading endpoint

# Risk parameters (loaded from environment variables on Render, with fallbacks)
PRICE_CHANGE_THRESHOLD = float(os.environ.get("PRICE_CHANGE_THRESHOLD", 0.015))
SMA_PERIOD             = int(os.environ.get("SMA_PERIOD", 20))
SMA_TOUCH_PCT          = float(os.environ.get("SMA_TOUCH_PCT", 0.005))
STOP_LOSS_PCT          = float(os.environ.get("STOP_LOSS_PCT", 0.015))
TARGET1_PCT            = float(os.environ.get("TARGET1_PCT", 0.02))
TARGET2_PCT            = float(os.environ.get("TARGET2_PCT", 0.02))
POSITION_SIZE_USD      = float(os.environ.get("POSITION_SIZE_USD", 1000))
MAX_OPEN_POSITIONS     = int(os.environ.get("MAX_OPEN_POSITIONS", 20))
LEVERAGE_MULTIPLIER    = float(os.environ.get("LEVERAGE_MULTIPLIER", 2.0))

IST = pytz.timezone("Asia/Kolkata")
NYSE_TZ = pytz.timezone("America/New_York")

# Webhook Alert URL (Discord/Telegram)
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

def send_notification(message: str):
    """Send a webhook notification to Discord or Telegram if configured."""
    if not WEBHOOK_URL:
        return
    try:
        if "discord.com" in WEBHOOK_URL:
            requests.post(WEBHOOK_URL, json={"content": message}, timeout=5)
        elif "api.telegram.org" in WEBHOOK_URL:
            requests.post(WEBHOOK_URL, json={"text": message}, timeout=5)
        else:
            requests.post(WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as e:
        log.error(f"Failed to send webhook notification: {e}")

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
eod_summary_sent_date = None


# ─────────────────────────────────────────────
# S&P 500 SYMBOL LIST
# ─────────────────────────────────────────────
def get_sp500_symbols() -> list[str]:
    """Fetch S&P 500 symbols from Wikipedia with robust fallbacks."""
    try:
        log.info("Fetching S&P 500 symbols dynamically from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
        # Use StringIO to avoid Pandas read_html deprecation warning
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        # Wikipedia uses '.' instead of '-' for some tickers (e.g. BRK.B).
        # Alpaca uses '-' (e.g. BRK-B) or normal letters. Let's replace '.' with '-'
        symbols = [sym.replace(".", "-") for sym in df["Symbol"].tolist()]
        log.info(f"Successfully fetched {len(symbols)} S&P 500 symbols from Wikipedia.")
        return symbols
    except Exception as e:
        log.warning(f"Failed to fetch S&P 500 symbols from Wikipedia: {e}. Falling back to local file.")
        # Fallback 1: Local file
        try:
            file_path = os.path.join(os.path.dirname(__file__), "sp500.txt")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    symbols = [line.strip() for line in f if line.strip()]
                log.info(f"Loaded {len(symbols)} S&P 500 symbols from local sp500.txt")
                return symbols
            else:
                raise FileNotFoundError("Local sp500.txt file not found.")
        except Exception as ex:
            log.error(f"Failed to load S&P 500 list from local file: {ex}. Using fallback list.")
            # Fallback 2: Hardcoded top liquid symbols
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
def compute_sma(bars: pd.DataFrame, period: int = 20) -> float | None:
    """Compute the simple moving average of close prices."""
    closes = bars["close"].values
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


# ─────────────────────────────────────────────
# ORDER HELPERS
# ─────────────────────────────────────────────
def log_portfolio_market_value():
    """Fetch all positions from Alpaca and log the sum of their market values (raw and absolute)."""
    try:
        # Give a moment for the order to settle on Alpaca before querying positions
        time.sleep(1.0)
        positions = trade_client.get_all_positions()
        raw_mv_sum = sum(float(pos.market_value) for pos in positions if pos.market_value is not None)
        abs_mv_sum = sum(abs(float(pos.market_value)) for pos in positions if pos.market_value is not None)
        log.info(
            f"[PORTFOLIO SUMMARY] Raw Market Value Sum: ${raw_mv_sum:.2f} | "
            f"Absolute Exposure: ${abs_mv_sum:.2f} | Open Positions: {len(positions)}"
        )
    except Exception as e:
        log.error(f"Failed to log portfolio market value: {e}")


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
        # Log the updated portfolio market value after the order is placed
        threading.Thread(target=log_portfolio_market_value, daemon=True).start()
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


def get_account_equity() -> float:
    """Return total account equity in USD."""
    try:
        account = trade_client.get_account()
        return float(account.equity)
    except Exception as e:
        log.error(f"Failed to fetch account equity: {e}")
        return 0.0


# ─────────────────────────────────────────────
# ENTRY LOGIC
# ─────────────────────────────────────────────
def scan_for_entries(bulk_data):
    """Scan S&P 500 for entry signals and open new trades using bulk data."""
    ist_now = datetime.now(IST)
    # No new trades after 1:00 AM IST
    if (ist_now.hour == 1 and ist_now.minute >= 0) or ist_now.hour == 2:
        log.info("After 1:00 AM IST. Skipping entry scans.")
        return

    if len(open_trades) >= MAX_OPEN_POSITIONS:
        log.info(f"Max positions ({MAX_OPEN_POSITIONS}) reached. Skipping scan.")
        return

    equity = get_account_equity()
    if equity <= 0:
        log.warning("Could not fetch account equity. Using fallback position size of $1000.")
        position_size = 1000.0
    else:
        # Equal allocation with leverage: divide (equity * leverage) by max positions
        position_size = (equity * LEVERAGE_MULTIPLIER) / MAX_OPEN_POSITIONS

    buying_power = get_account_buying_power()
    if buying_power < position_size:
        log.info(f"Insufficient buying power: ${buying_power:.2f} (Required: ${position_size:.2f})")
        return

    candidates = [s for s in SP500_SYMBOLS if s not in open_trades]

    for symbol in candidates:
        if len(open_trades) >= MAX_OPEN_POSITIONS:
            break

        if symbol not in bulk_data:
            continue

        symbol_df = bulk_data[symbol].dropna(subset=['Close'])
        if symbol_df.empty or len(symbol_df) < SMA_PERIOD:
            continue

        # Rename columns to lowercase for compatibility
        symbol_df = symbol_df.copy()
        symbol_df.columns = [c.lower() for c in symbol_df.columns]

        # ── 1. Check intraday change ≥ ±PRICE_CHANGE_THRESHOLD ──────
        nyse_today = datetime.now(NYSE_TZ).date()
        today_bars = symbol_df[symbol_df.index.date == nyse_today]
        if today_bars.empty:
            continue

        open_price = float(today_bars.iloc[0]["open"])
        price = float(today_bars.iloc[-1]["close"])
        if open_price == 0:
            continue

        change = (price - open_price) / open_price
        if abs(change) < PRICE_CHANGE_THRESHOLD:
            continue

        # ── 2. Get 20-bar SMA on 5-min bars ──────────────────────────
        sma = compute_sma(symbol_df, SMA_PERIOD)
        if sma is None or sma == 0:
            continue

        # ── 3. Price is "touching" the SMA (within SMA_TOUCH_PCT) ────
        distance = abs(price - sma) / sma
        if distance > SMA_TOUCH_PCT:
            continue

        # ── 4. All conditions met → place order ──────────────────
        qty = max(1, int(position_size / price))
        is_long = change > 0
        side = OrderSide.BUY if is_long else OrderSide.SELL
        
        log.info(
            f"SIGNAL → {symbol} | {'LONG' if is_long else 'SHORT'} | price={price:.2f} | SMA={sma:.2f} "
            f"| change={change*100:.2f}% | qty={qty}"
        )

        if place_market_order(symbol, qty, side):
            entry_price = price
            if is_long:
                stop_loss = round(entry_price * (1 - STOP_LOSS_PCT), 4)
                target1   = round(entry_price * (1 + TARGET1_PCT), 4)
                target2   = round(entry_price * (1 + TARGET1_PCT + TARGET2_PCT), 4)
            else:
                stop_loss = round(entry_price * (1 + STOP_LOSS_PCT), 4)
                target1   = round(entry_price * (1 - TARGET1_PCT), 4)
                target2   = round(entry_price * (1 - TARGET1_PCT - TARGET2_PCT), 4)

            open_trades[symbol] = {
                "entry_price" : entry_price,
                "stop_loss"   : stop_loss,
                "target1"     : target1,
                "target2"     : target2,
                "t1_hit"      : False,
                "qty_total"   : qty,
                "qty_remaining": qty,
                "direction"   : "up" if is_long else "down",
            }
            log.info(
                f"Trade opened → {symbol} | {'LONG' if is_long else 'SHORT'} | entry={entry_price:.2f} "
                f"| SL={open_trades[symbol]['stop_loss']:.2f} "
                f"| T1={open_trades[symbol]['target1']:.2f} "
                f"| T2={open_trades[symbol]['target2']:.2f}"
            )
            send_notification(
                f"🔔 **TRADE OPENED**: {symbol} | {'LONG' if is_long else 'SHORT'} | entry={entry_price:.2f} "
                f"| SL={open_trades[symbol]['stop_loss']:.2f} "
                f"| T1={open_trades[symbol]['target1']:.2f} "
                f"| T2={open_trades[symbol]['target2']:.2f}"
            )
            time.sleep(0.3)   # rate-limit courtesy


# ─────────────────────────────────────────────
# EXIT LOGIC
# ─────────────────────────────────────────────
def manage_open_trades(bulk_data):
    """Check all open positions for stop-loss or target hits using bulk data."""
    to_close = []

    for symbol, trade in list(open_trades.items()):
        if symbol not in bulk_data:
            continue
        symbol_df = bulk_data[symbol].dropna(subset=['Close'])
        if symbol_df.empty:
            continue
        
        price = float(symbol_df.iloc[-1]['Close'])

        qty_remaining = trade["qty_remaining"]
        if qty_remaining <= 0:
            to_close.append(symbol)
            continue

        is_long = trade["direction"] == "up"
        exit_side = OrderSide.SELL if is_long else OrderSide.BUY

        # ── Stop-loss hit ─────────────────────────────────────────────
        sl_hit = (is_long and price <= trade["stop_loss"]) or (not is_long and price >= trade["stop_loss"])
        if sl_hit:
            log.info(
                f"STOP-LOSS hit → {symbol} | price={price:.2f} "
                f"| SL={trade['stop_loss']:.2f} | closing {qty_remaining} shares"
            )
            send_notification(
                f"🔴 **STOP-LOSS HIT**: {symbol} | price={price:.2f} "
                f"| SL={trade['stop_loss']:.2f} | closing {qty_remaining} shares"
            )
            if place_market_order(symbol, qty_remaining, exit_side):
                to_close.append(symbol)
            continue

        # ── Target 2 hit (sell remaining 50 %) ────────────────────────
        t2_hit = (is_long and price >= trade["target2"]) or (not is_long and price <= trade["target2"])
        if trade["t1_hit"] and t2_hit:
            log.info(
                f"TARGET 2 hit → {symbol} | price={price:.2f} "
                f"| T2={trade['target2']:.2f} | closing {qty_remaining} shares"
            )
            send_notification(
                f"🟢 **TARGET 2 HIT**: {symbol} | price={price:.2f} "
                f"| T2={trade['target2']:.2f} | closing {qty_remaining} shares"
            )
            if place_market_order(symbol, qty_remaining, exit_side):
                to_close.append(symbol)
            continue

        # ── Target 1 hit (sell first 50 %) ────────────────────────────
        t1_hit = (is_long and price >= trade["target1"]) or (not is_long and price <= trade["target1"])
        if not trade["t1_hit"] and t1_hit:
            half_qty = max(1, qty_remaining // 2)
            log.info(
                f"TARGET 1 hit → {symbol} | price={price:.2f} "
                f"| T1={trade['target1']:.2f} | closing {half_qty} of {qty_remaining} shares"
            )
            send_notification(
                f"🟢 **TARGET 1 HIT**: {symbol} | price={price:.2f} "
                f"| T1={trade['target1']:.2f} | closing {half_qty} of {qty_remaining} shares"
            )
            if place_market_order(symbol, half_qty, exit_side):
                open_trades[symbol]["t1_hit"]       = True
                open_trades[symbol]["qty_remaining"] = qty_remaining - half_qty
                # raise/lower stop-loss to break-even after T1
                open_trades[symbol]["stop_loss"]     = trade["entry_price"]
                log.info(
                    f"Stop-loss moved to break-even ({trade['entry_price']:.2f}) for {symbol}"
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
    Liquidate ALL open positions directly from Alpaca API at 1:15 AM IST
    to guarantee zero overnight holds, and send a daily summary.
    """
    global eod_summary_sent_date
    ist_now = datetime.now(IST)
    today_str = ist_now.strftime("%Y-%m-%d")

    # Check if time is 1:15 AM IST or later
    if ist_now.hour == 1 and ist_now.minute >= 15:
        if eod_summary_sent_date == today_str:
            return  # already sent summary for today

        log.info("EOD (1:15 AM IST reached): Liquidating all open positions on Alpaca account.")
        try:
            positions = trade_client.get_all_positions()
            if positions:
                send_notification("🧹 **EOD SQUARE-OFF**: Liquidating open positions...")
                for pos in positions:
                    symbol = pos.symbol
                    qty = abs(int(pos.qty))
                    is_long = pos.side == PositionSide.LONG
                    exit_side = OrderSide.SELL if is_long else OrderSide.BUY
                    log.info(f"EOD Liquidating {symbol} | qty={qty} | side={pos.side}")
                    place_market_order(symbol, qty, exit_side)
                # Wait 3 seconds for orders to settle before fetching final equity
                time.sleep(3.0)
            
            open_trades.clear()

            # Fetch account equity & calculate daily returns
            account = trade_client.get_account()
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_pnl = equity - last_equity
            daily_pnl_pct = (daily_pnl / last_equity) * 100 if last_equity > 0 else 0.0

            status_indicator = "🟢" if daily_pnl >= 0 else "🔴"
            summary_msg = (
                f"{status_indicator} **DAILY PERFORMANCE SUMMARY** {status_indicator}\n"
                f"• Ending Equity: ${equity:,.2f}\n"
                f"• Daily P&L: {daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)\n"
                f"• Positions: All positions successfully squared off."
            )
            log.info(summary_msg)
            send_notification(summary_msg)
            eod_summary_sent_date = today_str

        except Exception as e:
            log.error(f"EOD liquidation failed: {e}")
            send_notification(f"⚠️ **EOD LIQUIDATION ERROR**: {e}")


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

    # Identify tickers we need to fetch data for
    tickers_to_fetch = list(set(SP500_SYMBOLS + list(open_trades.keys())))
    if not tickers_to_fetch:
        log.info("No tickers to fetch.")
        return

    try:
        log.info(f"Bulk downloading data for {len(tickers_to_fetch)} symbols...")
        bulk_data = yf.download(
            tickers=tickers_to_fetch,
            period="5d",
            interval="5m",
            group_by="ticker",
            threads=15,
            progress=False,
            timeout=15
        )
        if bulk_data.empty:
            raise ValueError("Bulk data download returned an empty DataFrame.")
    except Exception as e:
        log.error(f"Bulk download failed: {e}. Skipping this tick.")
        return

    # 1. Manage exits first
    manage_open_trades(bulk_data)

    # 2. Check EOD liquidation
    close_all_positions_eod()

    # 3. Scan for new entries
    scan_for_entries(bulk_data)

    log.info(f"Open trades: {list(open_trades.keys()) or 'None'}")


def initialize_open_trades_from_alpaca():
    """Load existing open positions from Alpaca on startup to populate open_trades."""
    try:
        positions = trade_client.get_all_positions()
        for pos in positions:
            symbol = pos.symbol
            qty = int(pos.qty)
            entry_price = float(pos.avg_entry_price)
            direction = "up" if pos.side == PositionSide.LONG else "down"
            
            is_long = direction == "up"
            if is_long:
                stop_loss = round(entry_price * (1 - STOP_LOSS_PCT), 4)
                target1   = round(entry_price * (1 + TARGET1_PCT), 4)
                target2   = round(entry_price * (1 + TARGET1_PCT + TARGET2_PCT), 4)
            else:
                stop_loss = round(entry_price * (1 + STOP_LOSS_PCT), 4)
                target1   = round(entry_price * (1 - TARGET1_PCT), 4)
                target2   = round(entry_price * (1 - TARGET1_PCT - TARGET2_PCT), 4)
                
            open_trades[symbol] = {
                "entry_price" : entry_price,
                "stop_loss"   : stop_loss,
                "target1"     : target1,
                "target2"     : target2,
                "t1_hit"      : False,
                "qty_total"   : abs(qty),
                "qty_remaining": abs(qty),
                "direction"   : direction,
            }
        log.info(f"Initialized {len(open_trades)} active positions from Alpaca: {list(open_trades.keys())}")
        raw_mv_sum = sum(float(pos.market_value) for pos in positions if pos.market_value is not None)
        abs_mv_sum = sum(abs(float(pos.market_value)) for pos in positions if pos.market_value is not None)
        log.info(
            f"[PORTFOLIO SUMMARY] Raw Market Value Sum: ${raw_mv_sum:.2f} | "
            f"Absolute Exposure: ${abs_mv_sum:.2f} | Open Positions: {len(positions)}"
        )
        send_notification(
            f"🚀 **BOT STARTUP**: Loaded {len(open_trades)} active positions. "
            f"Raw MV: ${raw_mv_sum:.2f} | Absolute Exposure: ${abs_mv_sum:.2f}"
        )
    except Exception as e:
        log.error(f"Failed to initialize open positions from Alpaca: {e}")
        send_notification(f"⚠️ **BOT STARTUP ERROR**: {e}")


# ─────────────────────────────────────────────
# HTTP HEALTH CHECK SERVER
# ─────────────────────────────────────────────
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is healthy and running!")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress logging to keep output clean
        return

def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    log.info(f"Starting health check server on port {port}...")
    server.serve_forever()


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
    log.info(f"    Position size          : Dynamic (Total Equity * {LEVERAGE_MULTIPLIER} / {MAX_OPEN_POSITIONS})")
    log.info(f"    Max positions          : {MAX_OPEN_POSITIONS}")
    log.info("=" * 60)

    # Initialize active positions from Alpaca
    initialize_open_trades_from_alpaca()

    # Start health check server in a background thread for Render
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()

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
