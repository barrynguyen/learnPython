"""
Stock Price Tracker with Moving Average Crossover & Volume Spike Alerts

Usage:
  python stock_tracker.py AAPL
  python stock_tracker.py TSLA --period 6mo
  python stock_tracker.py AAPL MSFT --no-chart
  python stock_tracker.py --demo          # synthetic data (no internet needed)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tabulate import tabulate
import argparse
import sys


# ─── Alert Thresholds ────────────────────────────────────────────────────────

VOLUME_SPIKE_MULTIPLIER = 2.0   # Volume > 2x 20-day average = spike
PRICE_CHANGE_THRESHOLD  = 0.03  # 3% intraday change = unusual move


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for symbol '{symbol}'")
    df.index = df.index.tz_localize(None)
    return df


def generate_demo_data(symbol: str = "DEMO", days: int = 365) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for offline demo."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")

    # Geometric Brownian Motion for price
    mu, sigma = 0.0003, 0.018
    returns = rng.normal(mu, sigma, days)
    # Inject a few large moves to trigger alerts
    spike_days = rng.choice(days, size=8, replace=False)
    returns[spike_days] += rng.choice([-1, 1], size=8) * rng.uniform(0.04, 0.09, 8)

    price = 150.0 * np.exp(np.cumsum(returns))
    high  = price * (1 + rng.uniform(0.002, 0.015, days))
    low   = price * (1 - rng.uniform(0.002, 0.015, days))
    open_ = price * (1 + rng.normal(0, 0.005, days))

    base_vol = 50_000_000
    volume   = rng.integers(int(base_vol * 0.5), int(base_vol * 1.5), days).astype(float)
    # Volume spikes on same days as price spikes
    volume[spike_days] *= rng.uniform(2.5, 5.0, 8)

    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  price,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


# ─── Indicator Calculation ───────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for window in (20, 50, 200):
        df[f"MA{window}"] = df["Close"].rolling(window).mean()
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    return df


def detect_ma_crossovers(df: pd.DataFrame) -> list[dict]:
    """Return list of crossover events: price crossing MA20/50/200."""
    alerts = []
    for ma in ("MA20", "MA50", "MA200"):
        valid = df[["Close", ma]].dropna()
        if len(valid) < 2:
            continue

        above = valid["Close"] > valid[ma]
        cross = above.astype(int).diff().abs()   # 1 where direction flipped

        for date, row in valid[cross == 1].iterrows():
            direction = "vượt lên" if above.loc[date] else "cắt xuống"
            alerts.append({
                "date":      date,
                "type":      f"Giá {direction} {ma}",
                "close":     round(row["Close"], 2),
                "ma_value":  round(row[ma], 2),
                "direction": "bullish" if above.loc[date] else "bearish",
            })

    return sorted(alerts, key=lambda x: x["date"])


def detect_volume_spikes(df: pd.DataFrame) -> list[dict]:
    """Return sessions where volume exceeded 2× its 20-day rolling average."""
    valid = df[["Close", "Open", "Volume", "Vol_MA20"]].dropna()
    spikes = valid[valid["Volume"] > VOLUME_SPIKE_MULTIPLIER * valid["Vol_MA20"]]

    alerts = []
    for date, row in spikes.iterrows():
        pct_change = (row["Close"] - row["Open"]) / row["Open"] * 100
        alerts.append({
            "date":        date,
            "type":        "Khối lượng đột biến",
            "close":       round(row["Close"], 2),
            "volume":      int(row["Volume"]),
            "vol_ma20":    int(row["Vol_MA20"]),
            "vol_ratio":   round(row["Volume"] / row["Vol_MA20"], 2),
            "pct_change":  round(pct_change, 2),
        })

    return sorted(alerts, key=lambda x: x["date"])


def detect_price_spikes(df: pd.DataFrame) -> list[dict]:
    """Return sessions with intraday price move > PRICE_CHANGE_THRESHOLD."""
    valid = df[["Close", "Open"]].dropna()
    pct = ((valid["Close"] - valid["Open"]) / valid["Open"]).abs()
    spikes = valid[pct > PRICE_CHANGE_THRESHOLD]

    alerts = []
    for date, row in spikes.iterrows():
        pct_val = (row["Close"] - row["Open"]) / row["Open"] * 100
        alerts.append({
            "date":       date,
            "type":       "Biến động giá mạnh",
            "close":      round(row["Close"], 2),
            "open":       round(row["Open"], 2),
            "pct_change": round(pct_val, 2),
        })

    return sorted(alerts, key=lambda x: x["date"])


# ─── Reporting ───────────────────────────────────────────────────────────────

def _fmt_date(d) -> str:
    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)


def print_summary(symbol: str, df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    day_chg     = latest["Close"] - prev["Close"]
    day_chg_pct = day_chg / prev["Close"] * 100
    vol_ratio   = (latest["Volume"] / latest["Vol_MA20"]
                   if pd.notna(latest["Vol_MA20"]) and latest["Vol_MA20"] > 0
                   else float("nan"))

    rows = [
        ["Mã cổ phiếu",       symbol.upper()],
        ["Ngày",              _fmt_date(df.index[-1])],
        ["Giá đóng cửa",      f"{latest['Close']:.2f}"],
        ["Thay đổi ngày",     f"{day_chg:+.2f} ({day_chg_pct:+.2f}%)"],
        ["MA20",              f"{latest['MA20']:.2f}" if pd.notna(latest["MA20"])  else "N/A"],
        ["MA50",              f"{latest['MA50']:.2f}" if pd.notna(latest["MA50"])  else "N/A"],
        ["MA200",             f"{latest['MA200']:.2f}" if pd.notna(latest["MA200"]) else "N/A"],
        ["Khối lượng hôm nay",f"{int(latest['Volume']):,}"],
        ["KL / MA20-KL",      f"{vol_ratio:.2f}x" if not pd.isna(vol_ratio) else "N/A"],
    ]

    print("\n" + "=" * 50)
    print(f"  THÔNG TIN CỔ PHIẾU: {symbol.upper()}")
    print("=" * 50)
    print(tabulate(rows, tablefmt="simple"))


def print_alerts(crossovers: list, vol_spikes: list, price_spikes: list,
                 recent_days: int = 30) -> None:
    cutoff = datetime.now() - timedelta(days=recent_days)

    print(f"\n{'=' * 50}")
    print(f"  CẢNH BÁO (30 ngày gần nhất)")
    print("=" * 50)

    # MA crossovers
    recent_cross = [a for a in crossovers if a["date"] >= cutoff]
    if recent_cross:
        print(f"\n[MA CROSSOVER] — {len(recent_cross)} tín hiệu")
        rows = [
            [_fmt_date(a["date"]), a["type"], a["close"], a["ma_value"],
             "▲ Tăng" if a["direction"] == "bullish" else "▼ Giảm"]
            for a in recent_cross
        ]
        print(tabulate(rows,
                       headers=["Ngày", "Loại", "Giá đóng", "Giá MA", "Xu hướng"],
                       tablefmt="rounded_outline"))
    else:
        print("\n[MA CROSSOVER] — Không có tín hiệu nào trong 30 ngày qua")

    # Volume spikes
    recent_vol = [a for a in vol_spikes if a["date"] >= cutoff]
    if recent_vol:
        print(f"\n[KHỐI LƯỢNG ĐỘT BIẾN] — {len(recent_vol)} phiên")
        rows = [
            [_fmt_date(a["date"]), f"{a['volume']:,}", f"{a['vol_ma20']:,}",
             f"{a['vol_ratio']}x", f"{a['pct_change']:+.2f}%"]
            for a in recent_vol
        ]
        print(tabulate(rows,
                       headers=["Ngày", "Khối lượng", "KL-MA20", "Tỷ lệ", "% Thay đổi"],
                       tablefmt="rounded_outline"))
    else:
        print("\n[KHỐI LƯỢNG ĐỘT BIẾN] — Không có phiên nào trong 30 ngày qua")

    # Price spikes
    recent_price = [a for a in price_spikes if a["date"] >= cutoff]
    if recent_price:
        print(f"\n[BIẾN ĐỘNG GIÁ MẠNH ≥{PRICE_CHANGE_THRESHOLD*100:.0f}%] — {len(recent_price)} phiên")
        rows = [
            [_fmt_date(a["date"]), a["open"], a["close"], f"{a['pct_change']:+.2f}%"]
            for a in recent_price
        ]
        print(tabulate(rows,
                       headers=["Ngày", "Mở cửa", "Đóng cửa", "% Thay đổi"],
                       tablefmt="rounded_outline"))
    else:
        print(f"\n[BIẾN ĐỘNG GIÁ MẠNH] — Không có phiên nào trong 30 ngày qua")


# ─── Chart ───────────────────────────────────────────────────────────────────

def plot_chart(symbol: str, df: pd.DataFrame,
               crossovers: list, vol_spikes: list) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"Stock Tracker — {symbol.upper()}", fontsize=14, fontweight="bold")

    # ── Price + MA ──────────────────────────────────────────────────────────
    ax1.plot(df.index, df["Close"], label="Giá đóng cửa", color="#2196F3", linewidth=1.5, zorder=3)

    ma_styles = {
        "MA20":  ("#FF9800", "--", 1.2),
        "MA50":  ("#9C27B0", "-.", 1.2),
        "MA200": ("#F44336", ":",  1.8),
    }
    for col, (color, ls, lw) in ma_styles.items():
        valid = df[col].dropna()
        if not valid.empty:
            ax1.plot(valid.index, valid, label=col, color=color,
                     linestyle=ls, linewidth=lw, zorder=2)

    # Crossover markers
    for a in crossovers:
        if a["date"] in df.index:
            color  = "#00C853" if a["direction"] == "bullish" else "#D50000"
            marker = "^"       if a["direction"] == "bullish" else "v"
            ax1.scatter(a["date"], a["close"], color=color, marker=marker,
                        s=120, zorder=5, label=None)

    ax1.set_ylabel("Giá (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#FAFAFA")

    # ── Volume ──────────────────────────────────────────────────────────────
    colors = ["#EF5350" if c < o else "#26A69A"
              for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1, label="Khối lượng")

    if df["Vol_MA20"].notna().any():
        ax2.plot(df.index, df["Vol_MA20"], color="#FF9800", linewidth=1.2,
                 linestyle="--", label="KL-MA20")

    # Volume spike markers
    for a in vol_spikes:
        if a["date"] in df.index:
            ax2.axvline(a["date"], color="#9C27B0", alpha=0.4, linewidth=1.2)

    ax2.set_ylabel("Khối lượng")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#FAFAFA")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = f"{symbol.lower()}_tracker.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nBiểu đồ đã lưu: {out}")
    plt.show()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Theo dõi giá cổ phiếu và cảnh báo MA/volume đột biến",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python stock_tracker.py AAPL
  python stock_tracker.py TSLA --period 6mo
  python stock_tracker.py MSFT --period 2y --no-chart
  python stock_tracker.py AAPL MSFT TSLA --no-chart
  python stock_tracker.py --demo          # dữ liệu demo không cần internet
""",
    )
    parser.add_argument("symbols", nargs="*", help="Mã cổ phiếu (VD: AAPL MSFT)")
    parser.add_argument("--period",   default="1y",
                        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        help="Khoảng thời gian dữ liệu (mặc định: 1y)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Bỏ qua vẽ biểu đồ")
    parser.add_argument("--recent",   type=int, default=30,
                        help="Số ngày gần nhất để lọc cảnh báo (mặc định: 30)")
    parser.add_argument("--demo",     action="store_true",
                        help="Chạy với dữ liệu tổng hợp (không cần internet)")
    return parser.parse_args()


def run(symbol: str, period: str, show_chart: bool, recent_days: int,
        demo: bool = False) -> None:
    if demo:
        print(f"\n[DEMO] Đang tạo dữ liệu tổng hợp cho {symbol.upper()}...")
        df = generate_demo_data(symbol)
    else:
        print(f"\nĐang tải dữ liệu cho {symbol.upper()} ({period})...")
        df = fetch_data(symbol, period)

    df = add_moving_averages(df)

    crossovers   = detect_ma_crossovers(df)
    vol_spikes   = detect_volume_spikes(df)
    price_spikes = detect_price_spikes(df)

    print_summary(symbol, df)
    print_alerts(crossovers, vol_spikes, price_spikes, recent_days)

    if crossovers:
        print(f"\nTổng MA crossover (toàn bộ kỳ): {len(crossovers)}")
    if vol_spikes:
        print(f"Tổng phiên khối lượng đột biến: {len(vol_spikes)}")
    if price_spikes:
        print(f"Tổng phiên biến động giá mạnh:  {len(price_spikes)}")

    if show_chart:
        plot_chart(symbol, df, crossovers, vol_spikes)


def main() -> None:
    args = parse_args()

    if args.demo:
        symbols = args.symbols if args.symbols else ["DEMO"]
        for symbol in symbols:
            run(symbol, args.period, not args.no_chart, args.recent, demo=True)
        return

    if not args.symbols:
        print("Lỗi: cần ít nhất một mã cổ phiếu, hoặc dùng --demo", file=sys.stderr)
        sys.exit(1)

    for symbol in args.symbols:
        try:
            run(symbol, args.period, not args.no_chart, args.recent)
        except ValueError as e:
            print(f"Lỗi: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Lỗi không xác định cho {symbol}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
