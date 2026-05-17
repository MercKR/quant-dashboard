"""
퀀트 지표 계산 모듈.
입력: OHLCV DataFrame (인덱스=날짜)
출력: 지표 컬럼이 추가된 DataFrame
"""

import numpy as np
import pandas as pd

# ── 기본 수익률 ──────────────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pct_change_1d"]  = df["Close"].pct_change(1) * 100   # 전일대비 증감률(%)
    df["pct_change_5d"]  = df["Close"].pct_change(5) * 100
    df["pct_change_20d"] = df["Close"].pct_change(20) * 100
    df["log_return_1d"]  = np.log(df["Close"] / df["Close"].shift(1)) * 100
    return df


# ── 이동평균 괴리율 ──────────────────────────────────────────────────────────

def add_ma_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in [20, 60, 200]:
        ma = df["Close"].rolling(w).mean()
        df[f"ma_dev_{w}d"] = (df["Close"] - ma) / ma * 100   # %
    return df


# ── 볼린저밴드 폭 ────────────────────────────────────────────────────────────

def add_bb_width(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    ma  = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["bb_width"] = (2 * k * std) / ma * 100   # %
    return df


# ── RSI ─────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


# ── ATR (Average True Range) ─────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]     = tr.rolling(window).mean()
    df["atr_14_pct"] = df["atr_14"] / df["Close"] * 100   # 가격 대비 %
    return df


# ── Realized Volatility ──────────────────────────────────────────────────────

def add_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    for w in [5, 20, 60]:
        df[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100   # 연율화 %
    return df


# ── Z-Score (수익률의 표준화) ────────────────────────────────────────────────

def add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ret = df["log_return_1d"] if "log_return_1d" in df.columns else np.log(df["Close"] / df["Close"].shift(1)) * 100
    for w in [20, 60]:
        mu  = ret.rolling(w).mean()
        std = ret.rolling(w).std()
        df[f"zscore_{w}d"] = (ret - mu) / std.replace(0, np.nan)
    return df


# ── 거래량 비율 ──────────────────────────────────────────────────────────────

def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        avg_vol = df["Volume"].rolling(window).mean()
        df["volume_ratio"] = df["Volume"] / avg_vol.replace(0, np.nan)
    else:
        df["volume_ratio"] = np.nan
    return df


# ── 연속 방향 (연속 상승/하락 일수) ─────────────────────────────────────────

def add_consecutive_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    direction = np.sign(df["Close"].diff())
    streak = []
    count = 0
    for d in direction:
        if d == 0 or np.isnan(d):
            streak.append(count)
            continue
        if count == 0 or np.sign(count) == d:
            count += int(d)
        else:
            count = int(d)
        streak.append(count)
    df["streak"] = streak
    return df


# ── 전체 지표 한 번에 ────────────────────────────────────────────────────────

INDICATOR_META = {
    "pct_change_1d":      {"label": "전일대비(%)",       "unit": "%",   "decimals": 2},
    "pct_change_5d":      {"label": "5일 수익률(%)",     "unit": "%",   "decimals": 2},
    "pct_change_20d":     {"label": "20일 수익률(%)",    "unit": "%",   "decimals": 2},
    "log_return_1d":      {"label": "로그수익률(1일,%)", "unit": "%",   "decimals": 3},
    "ma_dev_20d":         {"label": "20일MA 괴리율(%)",  "unit": "%",   "decimals": 2},
    "ma_dev_60d":         {"label": "60일MA 괴리율(%)",  "unit": "%",   "decimals": 2},
    "ma_dev_200d":        {"label": "200일MA 괴리율(%)", "unit": "%",   "decimals": 2},
    "bb_width":           {"label": "BB폭(%)",           "unit": "%",   "decimals": 2},
    "rsi_14":             {"label": "RSI(14)",           "unit": "",    "decimals": 1},
    "atr_14_pct":         {"label": "ATR(14, %)",        "unit": "%",   "decimals": 2},
    "realized_vol_20d":   {"label": "실현변동성(20일,%)", "unit": "%",  "decimals": 1},
    "realized_vol_60d":   {"label": "실현변동성(60일,%)", "unit": "%",  "decimals": 1},
    "zscore_20d":         {"label": "Z-Score(20일)",     "unit": "",    "decimals": 2},
    "zscore_60d":         {"label": "Z-Score(60일)",     "unit": "",    "decimals": 2},
    "volume_ratio":       {"label": "거래량비율(20일)",  "unit": "x",   "decimals": 2},
    "streak":             {"label": "연속방향(일)",      "unit": "일",  "decimals": 0},
}


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_returns(df)
    df = add_ma_deviation(df)
    df = add_bb_width(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_realized_vol(df)
    df = add_zscore(df)
    df = add_volume_ratio(df)
    df = add_consecutive_days(df)
    return df


def get_indicator_columns() -> list[str]:
    return list(INDICATOR_META.keys())


def get_indicator_label(col: str) -> str:
    return INDICATOR_META.get(col, {}).get("label", col)
