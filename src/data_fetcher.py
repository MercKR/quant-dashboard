"""
Data fetcher
  - yfinance   : S&P500, 원유선물, 천연가스선물, 금선물, 개별종목(US/KR)
  - FinanceDataReader : 코스피200선물ETF(261220), 나스닥100선물ETF(304940), 은선물ETF(144600)
  - Binance REST: 비트코인
  - yfinance quarterly financials : 개별종목 펀더멘털
"""

import json
import time
import requests
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr

from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DATA_DIR        = Path(__file__).parent.parent / "data"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
DATA_DIR.mkdir(exist_ok=True)
FUNDAMENTALS_DIR.mkdir(exist_ok=True)

USER_STOCKS_PATH = DATA_DIR / "user_stocks.json"

# ── 기본 종목 정의 ────────────────────────────────────────────────────────────

YFINANCE_SYMBOLS = {
    "S&P500":      "^GSPC",
    "원유선물":     "CL=F",
    "천연가스선물": "NG=F",
    "금선물":       "GC=F",
}

FDR_SYMBOLS = {
    "코스피200선물": "261220",
    "나스닥100선물": "304940",
    "은선물":        "144600",
}

ALL_SYMBOLS = list(YFINANCE_SYMBOLS) + list(FDR_SYMBOLS) + ["비트코인"]


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _parquet_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.replace('/', '_')}.parquet"

def _fund_path(name: str) -> Path:
    return FUNDAMENTALS_DIR / f"{name.replace('/', '_')}.parquet"

def _load_existing(symbol: str) -> pd.DataFrame:
    p = _parquet_path(symbol)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def _save(symbol: str, df: pd.DataFrame):
    if df.empty:
        return
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_parquet(_parquet_path(symbol))

def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new
    if new.empty:
        return existing
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


# ── 사용자 종목 관리 ──────────────────────────────────────────────────────────

def load_user_stocks() -> list[dict]:
    """사용자가 추가한 개별종목 목록 반환. [{name, ticker}, ...]"""
    if USER_STOCKS_PATH.exists():
        try:
            return json.loads(USER_STOCKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_user_stocks(stocks: list[dict]):
    USER_STOCKS_PATH.write_text(
        json.dumps(stocks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ── yfinance (지수/선물/개별종목) ─────────────────────────────────────────────

def fetch_yfinance(symbol_name: str, ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    existing = _load_existing(symbol_name)
    if not existing.empty:
        start = (existing.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if df.empty:
            return existing
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        merged = _merge(existing, df)
        _save(symbol_name, merged)
        return merged
    except Exception as e:
        print(f"  [오류] {symbol_name}: {e}")
        return existing


def fetch_stock_price(name: str, ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """개별종목 가격 데이터 수집 (yfinance 래퍼)."""
    return fetch_yfinance(name, ticker, start)


# ── FinanceDataReader (KRX ETF) ───────────────────────────────────────────────

def fetch_fdr(symbol_name: str, ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    existing = _load_existing(symbol_name)
    if not existing.empty:
        start = (existing.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = fdr.DataReader(ticker, start)
        if df.empty:
            return existing
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        merged = _merge(existing, df)
        _save(symbol_name, merged)
        return merged
    except Exception as e:
        print(f"  [오류] {symbol_name}: {e}")
        return existing


# ── Binance REST (비트코인) ───────────────────────────────────────────────────

def fetch_bitcoin() -> pd.DataFrame:
    symbol_name = "비트코인"
    existing = _load_existing(symbol_name)
    start_ts = (
        int((existing.index[-1] + timedelta(days=1)).timestamp() * 1000)
        if not existing.empty
        else int(datetime(2015, 1, 1).timestamp() * 1000)
    )

    url = "https://api.binance.com/api/v3/klines"
    rows, limit = [], 1000

    while True:
        try:
            data = requests.get(url, params={
                "symbol": "BTCUSDT", "interval": "1d",
                "startTime": start_ts, "limit": limit,
            }, timeout=15).json()
        except Exception as e:
            print(f"  [오류] 비트코인: {e}")
            break
        if not data or isinstance(data, dict):
            break
        for k in data:
            rows.append({
                "Date": pd.to_datetime(k[0], unit="ms"),
                "Open": float(k[1]), "High": float(k[2]),
                "Low":  float(k[3]), "Close": float(k[4]),
                "Volume": float(k[5]),
            })
        if len(data) < limit:
            break
        start_ts = data[-1][0] + 86_400_000
        time.sleep(0.2)

    if not rows:
        return existing
    new_df = pd.DataFrame(rows).set_index("Date")
    new_df.index = new_df.index.tz_localize(None)
    merged = _merge(existing, new_df)
    _save(symbol_name, merged)
    return merged


# ── 개별종목 펀더멘털 ─────────────────────────────────────────────────────────

def fetch_stock_fundamentals(name: str, ticker: str) -> pd.DataFrame:
    """
    yfinance 분기 재무제표로부터 펀더멘털 지표를 계산합니다.
    분기 데이터에 45일 지연을 적용해 look-ahead bias를 최소화합니다.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        try:    inc = t.quarterly_income_stmt
        except: inc = pd.DataFrame()
        try:    bal = t.quarterly_balance_sheet
        except: bal = pd.DataFrame()
        try:    cf  = t.quarterly_cashflow
        except: cf  = pd.DataFrame()

        if inc.empty and bal.empty:
            return pd.DataFrame()

        def get_row(df: pd.DataFrame, *keys) -> pd.Series:
            for k in keys:
                if k in df.index:
                    return df.loc[k]
            return pd.Series(dtype=float)

        def qval(series: pd.Series, date) -> float:
            if series.empty or date not in series.index:
                return np.nan
            v = series[date]
            return float(v) if not pd.isna(v) else np.nan

        def ttm_sum(series: pd.Series, prior_dates: list) -> float:
            """TTM 합계 (가능한 분기 수로 연율화)."""
            vals = []
            for d in prior_dates:
                if d in series.index:
                    v = series[d]
                    if not pd.isna(v):
                        vals.append(float(v))
            if not vals:
                return np.nan
            return sum(vals) * (4.0 / len(vals))   # 분기수로 연율화

        # 각 재무 항목 시리즈 추출
        rev_s    = get_row(inc, "Total Revenue", "Revenue")
        gp_s     = get_row(inc, "Gross Profit")
        ebit_s   = get_row(inc, "EBIT", "Operating Income")
        ebitda_s = get_row(inc, "EBITDA", "Normalized EBITDA")
        ni_s     = get_row(inc, "Net Income")

        ta_s   = get_row(bal, "Total Assets")
        tl_s   = get_row(bal, "Total Liabilities Net Minority Interest", "Total Liabilities")
        ca_s   = get_row(bal, "Current Assets")
        cl_s   = get_row(bal, "Current Liabilities")
        cash_s = get_row(bal, "Cash And Cash Equivalents",
                         "Cash Cash Equivalents And Short Term Investments")
        re_s   = get_row(bal, "Retained Earnings")
        td_s   = get_row(bal, "Total Debt", "Long Term Debt")
        eq_s   = get_row(bal, "Common Stock Equity", "Stockholders Equity",
                         "Total Stockholders Equity", "Total Equity Gross Minority Interest")

        fcf_s = get_row(cf, "Free Cash Flow")

        shares_out = float(info.get("sharesOutstanding") or np.nan)

        # 모든 분기 날짜 수집 (오름차순)
        all_dates: list = []
        for stmt in [inc, bal, cf]:
            if not stmt.empty:
                all_dates.extend(stmt.columns.tolist())
        all_dates = sorted(set(all_dates))

        if not all_dates:
            return pd.DataFrame()

        def safe_div(a, b):
            if pd.isna(a) or pd.isna(b) or b == 0:
                return np.nan
            return a / b

        rows = []
        for i, qdate in enumerate(all_dates):
            prior = all_dates[max(0, i - 3): i + 1]   # 최대 4분기 (TTM)

            rev  = qval(rev_s,  qdate)
            gp   = qval(gp_s,   qdate)
            ebit = qval(ebit_s, qdate)
            ta   = qval(ta_s,   qdate)
            tl   = qval(tl_s,   qdate)
            ca   = qval(ca_s,   qdate)
            cl   = qval(cl_s,   qdate)
            cash = qval(cash_s, qdate)
            re   = qval(re_s,   qdate)
            td   = qval(td_s,   qdate)
            eq   = qval(eq_s,   qdate)

            t_rev   = ttm_sum(rev_s,    prior)
            t_ebit  = ttm_sum(ebit_s,   prior)
            t_ebitda = ttm_sum(ebitda_s, prior)
            t_ni    = ttm_sum(ni_s,     prior)
            t_fcf   = ttm_sum(fcf_s,    prior)

            eff_ebitda = t_ebitda if not pd.isna(t_ebitda) else t_ebit
            wc       = (ca - cl) if not (pd.isna(ca) or pd.isna(cl)) else np.nan
            net_debt = (td - cash) if not (pd.isna(td) or pd.isna(cash)) else np.nan
            td_safe  = 0.0 if pd.isna(td) else td
            cash_safe = 0.0 if pd.isna(cash) else cash
            ic       = (eq + td_safe - cash_safe) if not pd.isna(eq) else np.nan
            nopat    = t_ebit * 0.80 if not pd.isna(t_ebit) else np.nan

            rows.append({
                "date":             qdate,
                "gross_margin":     safe_div(gp, rev) * 100,
                "op_margin":        safe_div(ebit, rev) * 100,
                "net_debt_ebitda":  safe_div(net_debt, eff_ebitda),
                "roic":             safe_div(nopat, ic) * 100,
                "ttm_rev":          t_rev,
                "ttm_ebit":         t_ebit,
                "ttm_ni":           t_ni,
                "ttm_fcf":          t_fcf,
                "total_assets":     ta,
                "total_liabilities": tl,
                "working_capital":  wc,
                "retained_earnings": re,
                "stockholders_equity": eq,
                "shares_outstanding": shares_out,
                # Altman Z 구성 요소 (시가총액은 가격 데이터와 결합 시 계산)
                "z_wc_ta":   safe_div(wc,    ta),
                "z_re_ta":   safe_div(re,    ta),
                "z_ebit_ta": safe_div(t_ebit, ta),
                "z_tl":      tl,
                "z_rev_ta":  safe_div(t_rev, ta),
            })

        if not rows:
            return pd.DataFrame()

        fund_df = pd.DataFrame(rows).set_index("date")
        fund_df.index = pd.to_datetime(fund_df.index).tz_localize(None)
        # 45일 공시 지연 적용 (look-ahead bias 방지)
        fund_df.index = fund_df.index + pd.Timedelta(days=45)
        fund_df = fund_df.sort_index()

        fund_df.to_parquet(_fund_path(name))
        return fund_df

    except Exception as e:
        print(f"  [펀더멘털 오류] {name} ({ticker}): {e}")
        return pd.DataFrame()


def load_stock_fundamentals(name: str) -> pd.DataFrame:
    """저장된 펀더멘털 parquet 로드."""
    p = _fund_path(name)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fetch_user_stock(name: str, ticker: str, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """개별종목 가격 + 펀더멘털 수집."""
    if verbose:
        print(f"수집 중: {name} ({ticker})")
    price = fetch_stock_price(name, ticker)
    if verbose:
        print(f"  → 가격: {len(price)}행")
    fund = fetch_stock_fundamentals(name, ticker)
    if verbose:
        if not fund.empty:
            print(f"  → 펀더멘털: {len(fund)}분기")
        else:
            print(f"  → 펀더멘털: 없음 (ETF·코인·데이터 부족)")
    return price, fund


# ── 전체 수집 ────────────────────────────────────────────────────────────────

def fetch_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    results = {}

    for name, ticker in YFINANCE_SYMBOLS.items():
        if verbose: print(f"수집 중: {name} ({ticker})")
        results[name] = fetch_yfinance(name, ticker)
        time.sleep(0.3)

    for name, ticker in FDR_SYMBOLS.items():
        if verbose: print(f"수집 중: {name} ({ticker}, KRX ETF)")
        results[name] = fetch_fdr(name, ticker)
        time.sleep(0.3)

    if verbose: print("수집 중: 비트코인 (Binance)")
    results["비트코인"] = fetch_bitcoin()

    # 사용자 추가 종목 가격도 함께 업데이트
    for stock in load_user_stocks():
        results[stock["name"]] = fetch_stock_price(stock["name"], stock["ticker"])
        time.sleep(0.3)

    if verbose:
        print()
        for name, df in results.items():
            if not df.empty:
                print(f"  {name}: {len(df):,}행  {df.index[0].date()} ~ {df.index[-1].date()}")
            else:
                print(f"  {name}: 데이터 없음")

    return results


def load_all() -> dict[str, pd.DataFrame]:
    """저장된 parquet에서 전체 심볼(기본+사용자 종목) 로드."""
    result = {}
    for symbol in ALL_SYMBOLS:
        p = _parquet_path(symbol)
        if p.exists():
            result[symbol] = pd.read_parquet(p)
    for stock in load_user_stocks():
        p = _parquet_path(stock["name"])
        if p.exists():
            result[stock["name"]] = pd.read_parquet(p)
    return result


def load_fund_all() -> dict[str, pd.DataFrame]:
    """사용자 종목의 펀더멘털 parquet 전부 로드."""
    result = {}
    for stock in load_user_stocks():
        df = load_stock_fundamentals(stock["name"])
        if not df.empty:
            result[stock["name"]] = df
    return result


if __name__ == "__main__":
    fetch_all()
