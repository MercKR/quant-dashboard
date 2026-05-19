"""
퀀트 지표 계산 모듈.
입력: OHLCV DataFrame (인덱스=날짜)
출력: 지표 컬럼이 추가된 DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── 기본 수익률 ──────────────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pct_change_1d"]   = df["Close"].pct_change(1) * 100
    df["pct_change_5d"]   = df["Close"].pct_change(5) * 100
    df["pct_change_20d"]  = df["Close"].pct_change(20) * 100
    df["pct_change_252d"] = df["Close"].pct_change(252) * 100   # 1년(약 252 거래일)
    df["log_return_1d"]   = np.log(df["Close"] / df["Close"].shift(1)) * 100
    return df


# ── 이동평균 괴리율 ──────────────────────────────────────────────────────────

def add_ma_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in [20, 60, 200]:
        ma = df["Close"].rolling(w).mean()
        df[f"ma_dev_{w}d"] = (df["Close"] - ma) / ma * 100
    return df


# ── 볼린저밴드 폭 ────────────────────────────────────────────────────────────

def add_bb_width(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    ma  = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["bb_width"] = (2 * k * std) / ma * 100
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
    df["atr_14_pct"] = df["atr_14"] / df["Close"] * 100
    return df


# ── Realized Volatility ──────────────────────────────────────────────────────

def add_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    for w in [5, 20, 60]:
        df[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100
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


# ── 12-1M 가격 모멘텀 ────────────────────────────────────────────────────────

def add_momentum_12_1m(df: pd.DataFrame) -> pd.DataFrame:
    """12개월 수익률 - 1개월 수익률. 학계·실무에서 가장 검증된 모멘텀 팩터."""
    df = df.copy()
    if "pct_change_252d" not in df.columns:
        df["pct_change_252d"] = df["Close"].pct_change(252) * 100
    pct_1m = df["Close"].pct_change(21) * 100   # 21 거래일 ≈ 1개월
    df["momentum_12_1m"] = df["pct_change_252d"] - pct_1m
    return df


# ── 펀더멘털 지표 병합 ────────────────────────────────────────────────────────

def add_fundamentals(df: pd.DataFrame, fund_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    분기별 재무 데이터를 일별 가격 DataFrame에 forward-fill로 합칩니다.
    가격 데이터가 필요한 파생 지표(FCF Yield, P/E, P/B, Altman Z)도 이 단계에서 계산.
    """
    if fund_df is None or fund_df.empty:
        return df

    df = df.copy()
    # 분기 → 일별 forward-fill
    fund_d = fund_df.reindex(df.index, method="ffill")

    shares = fund_d.get("shares_outstanding", pd.Series(np.nan, index=df.index))

    # ── FCF Yield = TTM FCF / 시가총액 × 100
    if "ttm_fcf" in fund_d.columns:
        mkt_cap = df["Close"] * shares.replace(0, np.nan)
        df["fcf_yield"] = fund_d["ttm_fcf"] / mkt_cap.replace(0, np.nan) * 100

    # ── Trailing P/E = Close / (TTM 순이익 / 주식수)
    if "ttm_ni" in fund_d.columns:
        ttm_eps = fund_d["ttm_ni"] / shares.replace(0, np.nan)
        df["trailing_pe"] = df["Close"] / ttm_eps.replace(0, np.nan)

    # ── P/B = Close / (자기자본 / 주식수)
    if "stockholders_equity" in fund_d.columns:
        bvps = fund_d["stockholders_equity"] / shares.replace(0, np.nan)
        df["pb_ratio"] = df["Close"] / bvps.replace(0, np.nan)

    # ── Altman Z-Score = 1.2×WC/TA + 1.4×RE/TA + 3.3×EBIT/TA + 0.6×MktCap/TL + 1.0×Rev/TA
    z_cols = ["z_wc_ta", "z_re_ta", "z_ebit_ta", "z_tl", "z_rev_ta"]
    if all(c in fund_d.columns for c in z_cols):
        mkt_cap = df["Close"] * shares.replace(0, np.nan)
        z_mkttl = mkt_cap / fund_d["z_tl"].replace(0, np.nan)
        df["altman_z"] = (
            1.2 * fund_d["z_wc_ta"] +
            1.4 * fund_d["z_re_ta"] +
            3.3 * fund_d["z_ebit_ta"] +
            0.6 * z_mkttl +
            1.0 * fund_d["z_rev_ta"]
        )

    # ── 나머지 펀더멘털 컬럼 그대로 복사
    for col in ["gross_margin", "op_margin", "net_debt_ebitda", "roic"]:
        if col in fund_d.columns:
            df[col] = fund_d[col]

    return df


# ── 지표 메타데이터 ──────────────────────────────────────────────────────────

INDICATOR_META = {
    # 가격 수익률
    "pct_change_1d":      {"label": "전일대비(%)",        "unit": "%",  "decimals": 2},
    "pct_change_5d":      {"label": "5일 수익률(%)",      "unit": "%",  "decimals": 2},
    "pct_change_20d":     {"label": "20일 수익률(%)",     "unit": "%",  "decimals": 2},
    "pct_change_252d":    {"label": "1년 수익률(%)",      "unit": "%",  "decimals": 1},
    "log_return_1d":      {"label": "로그수익률(1일,%)",  "unit": "%",  "decimals": 3},
    # 모멘텀
    "momentum_12_1m":     {"label": "12-1M 모멘텀(%)",   "unit": "%",  "decimals": 1},
    "streak":             {"label": "연속방향(일)",       "unit": "일", "decimals": 0},
    # 이동평균 괴리율
    "ma_dev_20d":         {"label": "20일MA 괴리율(%)",  "unit": "%",  "decimals": 2},
    "ma_dev_60d":         {"label": "60일MA 괴리율(%)",  "unit": "%",  "decimals": 2},
    "ma_dev_200d":        {"label": "200일MA 괴리율(%)", "unit": "%",  "decimals": 2},
    # 변동성
    "bb_width":           {"label": "BB폭(%)",           "unit": "%",  "decimals": 2},
    "rsi_14":             {"label": "RSI(14)",           "unit": "",   "decimals": 1},
    "atr_14_pct":         {"label": "ATR(14, %)",        "unit": "%",  "decimals": 2},
    "realized_vol_20d":   {"label": "실현변동성(20일,%)", "unit": "%", "decimals": 1},
    "realized_vol_60d":   {"label": "실현변동성(60일,%)", "unit": "%", "decimals": 1},
    # 통계
    "zscore_20d":         {"label": "Z-Score(20일)",     "unit": "",   "decimals": 2},
    "zscore_60d":         {"label": "Z-Score(60일)",     "unit": "",   "decimals": 2},
    # 거래량
    "volume_ratio":       {"label": "거래량비율(20일)",  "unit": "x",  "decimals": 2},
    # ── 개별종목 펀더멘털 (해당 종목에만 존재) ──
    "gross_margin":       {"label": "총이익률(%)",        "unit": "%",  "decimals": 1},
    "op_margin":          {"label": "영업이익률(%)",      "unit": "%",  "decimals": 1},
    "roic":               {"label": "ROIC(%)",           "unit": "%",  "decimals": 1},
    "net_debt_ebitda":    {"label": "순부채/EBITDA",      "unit": "x",  "decimals": 2},
    "trailing_pe":        {"label": "Trailing P/E",      "unit": "x",  "decimals": 1},
    "pb_ratio":           {"label": "P/B Ratio",         "unit": "x",  "decimals": 2},
    "fcf_yield":          {"label": "FCF Yield(%)",      "unit": "%",  "decimals": 2},
    "altman_z":           {"label": "Altman Z-Score",    "unit": "",   "decimals": 2},
}

# 항상 존재하는 가격 기반 지표 (지수/ETF/코인 포함)
BASE_INDICATOR_COLS = [
    "pct_change_1d", "pct_change_5d", "pct_change_20d", "pct_change_252d",
    "log_return_1d", "momentum_12_1m", "streak",
    "ma_dev_20d", "ma_dev_60d", "ma_dev_200d",
    "bb_width", "rsi_14", "atr_14_pct",
    "realized_vol_20d", "realized_vol_60d",
    "zscore_20d", "zscore_60d",
    "volume_ratio",
]

# 개별종목에만 존재하는 펀더멘털 지표
FUND_INDICATOR_COLS = [
    "gross_margin", "op_margin", "roic", "net_debt_ebitda",
    "trailing_pe", "pb_ratio", "fcf_yield", "altman_z",
]


def compute_all(df: pd.DataFrame, fund_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = add_returns(df)
    df = add_ma_deviation(df)
    df = add_bb_width(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_realized_vol(df)
    df = add_zscore(df)
    df = add_volume_ratio(df)
    df = add_consecutive_days(df)
    df = add_momentum_12_1m(df)
    if fund_df is not None and not fund_df.empty:
        df = add_fundamentals(df, fund_df)
    return df


def get_indicator_columns() -> list[str]:
    """가격 기반 지표 컬럼 목록 (항상 존재)."""
    return list(BASE_INDICATOR_COLS)


def get_all_indicator_columns() -> list[str]:
    """가격 기반 + 펀더멘털 지표 전체 목록."""
    return list(BASE_INDICATOR_COLS) + list(FUND_INDICATOR_COLS)


def get_indicator_label(col: str) -> str:
    return INDICATOR_META.get(col, {}).get("label", col)
