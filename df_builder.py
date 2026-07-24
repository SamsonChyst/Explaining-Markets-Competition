"""
Downloading returns and compute excess against Vanguard Total Index (VTI) through yfinance
and FMP API.
Uses only transcripts with known intraday announcement time.
If the announcement time is exactly 00:00:00, it is treated as unknown and skipped.
Groups by year+quarter and computes cross-sectional percentile ranks.

Result observation-per-group:
0      2024-Q2          189
1      2024-Q3          217
2      2024-Q4          239
3      2025-Q1          238
4      2025-Q2          238
5      2025-Q3          225
6      2025-Q4           29
7      2026-Q1           20
8      2026-Q2           11
"""

import json
import os
from datetime import time
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

PRICE_CACHE_DIR = "PriceCache"
ANNOUNCEMENT_CUTOFF = time(16, 0)


def get_prices(
    ticker: str,
    start: str = "2023-06-01",
    cache_dir: str = PRICE_CACHE_DIR,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{ticker.upper()}.csv")

    if os.path.exists(cache_path):
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        prices.index = pd.to_datetime(prices.index).normalize()
        return prices

    prices = pd.DataFrame()

    try:
        prices = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

    except Exception:
        pass

    if not prices.empty:
        prices.index = pd.to_datetime(prices.index).normalize()
        prices.to_csv(cache_path)
        return prices

    print(f"{ticker}: Yahoo failed, trying FMP...")

    api_key = os.getenv("FMP_API")

    if not api_key:
        return pd.DataFrame()

    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/"
        f"{ticker}?from={start}&apikey={api_key}"
    )

    try:
        r = requests.get(url, timeout=30)

        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()

        history = data.get("historical")

        if not history:
            return pd.DataFrame()

        prices = pd.DataFrame(history)

        prices = prices.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        prices["Date"] = pd.to_datetime(prices["Date"])

        prices = prices.set_index("Date").sort_index()

        prices.index = prices.index.normalize()

        prices = prices[
            [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in prices.columns]
        ]

        prices.to_csv(cache_path)

        print(f"{ticker}: downloaded from FMP")

        return prices

    except Exception as e:
        print(f"{ticker}: FMP failed ({e})")
        return pd.DataFrame()


def parse_event_datetime(date_str: str):
    call_dt = pd.Timestamp(date_str)

    if call_dt.tzinfo is not None:
        call_dt = call_dt.tz_localize(None)

    time_known = not (
        call_dt.hour == 0
        and call_dt.minute == 0
        and call_dt.second == 0
        and call_dt.microsecond == 0
    )

    if not time_known:
        return None

    call_date = call_dt.normalize()
    after_close = call_dt.time() > ANNOUNCEMENT_CUTOFF

    return call_dt, call_date, after_close


def compute_excess_return_for_json(
    json_path: str,
    vti_prices: pd.DataFrame,
    stock_prices: pd.DataFrame,
    price_col: str = "Close",
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("summary"):
        return None

    parsed = parse_event_datetime(data["date"])
    if parsed is None:
        return None

    call_dt, call_date, after_close = parsed

    trading_days = vti_prices.index

    is_trading_day = call_date in trading_days

    if after_close or not is_trading_day:
        later_days = trading_days[trading_days > call_date]
        if later_days.empty:
            return None
        reaction_day = later_days.min()
    else:
        reaction_day = call_date

    earlier_days = trading_days[trading_days < reaction_day]
    if earlier_days.empty:
        return None

    prior_day = earlier_days.max()

    try:
        stock_ret = (
            stock_prices.loc[reaction_day, price_col]
            / stock_prices.loc[prior_day, price_col]
            - 1
        )

        vti_ret = (
            vti_prices.loc[reaction_day, price_col]
            / vti_prices.loc[prior_day, price_col]
            - 1
        )

    except KeyError:
        return None

    return {
        "ticker": data["symbol"],
        "year": data["year"],
        "quarter": data["quarter"],
        "call_date": call_dt,
        "reaction_day": reaction_day,
        "stock_return": stock_ret,
        "vti_return": vti_ret,
        "excess_return": stock_ret - vti_ret,
        "summary": data["summary"],
    }


def build_excess_return_table(
    transcripts_dir: str = "Transcripts",
    companies_csv: str = "companies.csv",
) -> pd.DataFrame:

    vti_prices = get_prices("VTI")

    companies = pd.read_csv(companies_csv)[["ticker", "industry"]]
    companies["ticker"] = (
        companies["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    rows = []

    for ticker in sorted(os.listdir(transcripts_dir)):
        ticker_dir = os.path.join(transcripts_dir, ticker)

        if not os.path.isdir(ticker_dir):
            continue

        stock_prices = get_prices(ticker)

        if stock_prices.empty:
            continue

        for filename in sorted(os.listdir(ticker_dir)):
            if not filename.endswith(".json"):
                continue

            result = compute_excess_return_for_json(
                os.path.join(ticker_dir, filename),
                vti_prices,
                stock_prices,
            )

            if result is not None:
                rows.append(result)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.merge(companies, on="ticker", how="left")

    df["year_quarter"] = (
        df["year"].astype(str)
        + "-Q"
        + df["quarter"].astype(str)
    )

    df["cross_sectional_excess_return"] = (
        df.groupby("year_quarter")["excess_return"]
        .rank(method="average", pct=True)
    )

    df = (
        df.sort_values(["year_quarter", "ticker"])
        .reset_index(drop=True)
    )

    return df


if __name__ == "__main__":
    '''
    table = build_excess_return_table()
    table.to_csv("df.csv", index=False)
    print("Saved df.csv")
    '''