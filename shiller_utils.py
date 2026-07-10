import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# last_year, average_return, or hard_code_date
SHILLER_EXTENSION_MODE = "hard_code_date"
SHILLER_AVERAGE_RETURN_LOOKBACK_YEARS = 10
SHILLER_HARD_CODE_DATE = pd.Timestamp("1965-01-01")
#SHILLER_HARD_CODE_DATE = pd.Timestamp("2007-08-01")


def parse_shiller_date(series):
    """
    Convert Shiller-style yyyy.mm strings/floats into proper datetimes.
    Examples:
      1950.01 -> 1950-01-01
      1950.1  -> 1950-10-01
      1950.11 -> 1950-11-01
    """
    s = series.astype(str).str.strip()

    def _norm(val):
        year, month_part = val.split(".", 1)
        # Special case: .1 means October (Excel dropped the zero)
        if month_part == "1":
            month = "10"
        else:
            month = month_part.zfill(2)
        return f"{year}-{month}-01"

    return pd.to_datetime(s.map(_norm), format="%Y-%m-%d")


def _fill_missing_cape_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill early missing CAPE series values with the default fallback."""

    if df is None or df.empty:
        return df

    result = df.copy()
    for column in ("CAPE", "TR CAPE"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(15)
    return result


def _prepare_shiller_extension_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, date-sorted Shiller dataframe with unique columns."""

    if df is None:
        return df
    df_sorted = df.loc[:, ~df.columns.duplicated()].copy()
    return df_sorted.sort_values("Date").reset_index(drop=True)


def extend_shiller_data_with_last_year(df: pd.DataFrame, extension_years: int) -> pd.DataFrame:
    """Extend Shiller data by repeating the last yearly return pattern forward."""

    extension_years = int(extension_years or 0)
    if df is None:
        return df
    if extension_years <= 0 or df.empty:
        return df.copy()

    df_sorted = _prepare_shiller_extension_frame(df)
    if len(df_sorted) < 2:
        return df_sorted

    future_months = extension_years * 12
    cycle_len = min(12, len(df_sorted) - 1)
    if cycle_len <= 0:
        return df_sorted

    cycle_rows = df_sorted.iloc[-(cycle_len + 1):].copy().reset_index(drop=True)
    stock_prices = cycle_rows["Real Total Return Price"].to_numpy(dtype=float)
    bond_prices = cycle_rows["Real Total Bond Returns"].to_numpy(dtype=float)
    cape_values = cycle_rows["CAPE"].to_numpy(dtype=float)
    tr_cape_values = cycle_rows["TR CAPE"].to_numpy(dtype=float)
    stock_returns = stock_prices[1:] / stock_prices[:-1]
    bond_returns = bond_prices[1:] / bond_prices[:-1]
    cape_returns = cape_values[1:] / cape_values[:-1]
    tr_cape_returns = tr_cape_values[1:] / tr_cape_values[:-1]

    last_date = pd.to_datetime(df_sorted["Date"].iloc[-1])
    current_stock = float(df_sorted["Real Total Return Price"].iloc[-1])
    current_bond = float(df_sorted["Real Total Bond Returns"].iloc[-1])
    current_cape = float(df_sorted["CAPE"].iloc[-1])
    current_tr_cape = float(df_sorted["TR CAPE"].iloc[-1])
    template_rows = df_sorted.iloc[-cycle_len:].reset_index(drop=True)

    future_rows = []
    for month_idx in range(future_months):
        cycle_idx = month_idx % cycle_len
        current_stock *= float(stock_returns[cycle_idx])
        current_bond *= float(bond_returns[cycle_idx])

        row = template_rows.iloc[cycle_idx].copy()
        row["Date"] = last_date + pd.DateOffset(months=month_idx + 1)
        row["Real Total Return Price"] = current_stock
        row["Real Total Bond Returns"] = current_bond
        row["CAPE"] = current_cape
        row["TR CAPE"] = current_tr_cape
        future_rows.append(row)

    if not future_rows:
        return df_sorted

    future_df = pd.DataFrame(future_rows)
    future_df = future_df[df_sorted.columns]
    return pd.concat([df_sorted, future_df], ignore_index=True)


def extend_shiller_data_with_average_return(
    df: pd.DataFrame,
    extension_years: int,
    lookback_years: int,
) -> pd.DataFrame:
    """Extend Shiller data using average recent monthly returns."""

    extension_years = int(extension_years or 0)
    lookback_years = int(lookback_years or 0)
    if df is None:
        return df
    if extension_years <= 0 or df.empty:
        return df.copy()

    df_sorted = _prepare_shiller_extension_frame(df)
    if len(df_sorted) < 2:
        return df_sorted

    future_months = extension_years * 12
    lookback_months = max(12, lookback_years * 12)
    lookback_rows = df_sorted.iloc[-(lookback_months + 1):].copy().reset_index(drop=True)
    if len(lookback_rows) < 2:
        return extend_shiller_data_with_last_year(df_sorted, extension_years)

    stock_prices = lookback_rows["Real Total Return Price"].to_numpy(dtype=float)
    bond_prices = lookback_rows["Real Total Bond Returns"].to_numpy(dtype=float)
    stock_returns = stock_prices[1:] / stock_prices[:-1]
    bond_returns = bond_prices[1:] / bond_prices[:-1]
    stock_factor = float(np.nanmean(stock_returns))
    bond_factor = float(np.nanmean(bond_returns))

    last_date = pd.to_datetime(df_sorted["Date"].iloc[-1])
    current_stock = float(df_sorted["Real Total Return Price"].iloc[-1])
    current_bond = float(df_sorted["Real Total Bond Returns"].iloc[-1])
    cape_average = float(np.nanmean(lookback_rows["CAPE"].astype(float)))
    tr_cape_average = float(np.nanmean(lookback_rows["TR CAPE"].astype(float)))

    future_rows = []
    for month_idx in range(future_months):
        current_stock *= stock_factor
        current_bond *= bond_factor

        row = df_sorted.iloc[-1].copy()
        row["Date"] = last_date + pd.DateOffset(months=month_idx + 1)
        row["Real Total Return Price"] = current_stock
        row["Real Total Bond Returns"] = current_bond
        row["CAPE"] = cape_average
        row["TR CAPE"] = tr_cape_average
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df = future_df[df_sorted.columns]
    return pd.concat([df_sorted, future_df], ignore_index=True)


def extend_shiller_data_with_hard_code_date(df: pd.DataFrame, extension_years: int) -> pd.DataFrame:
    """Extend Shiller data from the hard-coded anchor date forward using the post-anchor sequence."""

    extension_years = int(extension_years or 0)
    if df is None:
        return df
    if extension_years <= 0 or df.empty:
        return df.copy()

    df_sorted = _prepare_shiller_extension_frame(df)
    if len(df_sorted) < 2:
        return df_sorted

    anchor_mask = df_sorted["Date"] >= SHILLER_HARD_CODE_DATE
    if not anchor_mask.any():
        return extend_shiller_data_with_last_year(df_sorted, extension_years)

    anchor_idx = int(df_sorted.index[anchor_mask][0])
    anchor_df = df_sorted.iloc[anchor_idx:].copy().reset_index(drop=True)
    if len(anchor_df) < 2:
        return extend_shiller_data_with_last_year(df_sorted, extension_years)

    future_months = extension_years * 12
    cycle_len = len(anchor_df) - 1
    if cycle_len <= 0:
        return df_sorted

    cycle_rows = anchor_df.iloc[: cycle_len + 1].copy().reset_index(drop=True)
    stock_prices = cycle_rows["Real Total Return Price"].to_numpy(dtype=float)
    bond_prices = cycle_rows["Real Total Bond Returns"].to_numpy(dtype=float)
    cape_values = cycle_rows["CAPE"].to_numpy(dtype=float)
    tr_cape_values = cycle_rows["TR CAPE"].to_numpy(dtype=float)
    stock_returns = stock_prices[1:] / stock_prices[:-1]
    bond_returns = bond_prices[1:] / bond_prices[:-1]
    cape_returns = cape_values[1:] / cape_values[:-1]
    tr_cape_returns = tr_cape_values[1:] / tr_cape_values[:-1]

    last_date = pd.to_datetime(df_sorted["Date"].iloc[-1])
    current_stock = float(df_sorted["Real Total Return Price"].iloc[-1])
    current_bond = float(df_sorted["Real Total Bond Returns"].iloc[-1])
    current_cape = float(df_sorted["CAPE"].iloc[-1])
    current_tr_cape = float(df_sorted["TR CAPE"].iloc[-1])
    template_rows = cycle_rows.iloc[:-1].reset_index(drop=True)

    future_rows = []
    for month_idx in range(future_months):
        cycle_idx = month_idx % cycle_len
        current_stock *= float(stock_returns[cycle_idx])
        current_bond *= float(bond_returns[cycle_idx])
        current_cape *= float(cape_returns[cycle_idx])
        current_tr_cape *= float(tr_cape_returns[cycle_idx])

        row = template_rows.iloc[cycle_idx].copy()
        row["Date"] = last_date + pd.DateOffset(months=month_idx + 1)
        row["Real Total Return Price"] = current_stock
        row["Real Total Bond Returns"] = current_bond
        row["CAPE"] = current_cape
        row["TR CAPE"] = current_tr_cape
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    future_df = future_df[df_sorted.columns]
    return pd.concat([df_sorted, future_df], ignore_index=True)


def extend_shiller_data(
    df: pd.DataFrame,
    extension_years: int,
    extension_mode: Optional[str] = None,
) -> pd.DataFrame:
    """Extend Shiller data using the configured synthetic-tail strategy."""

    return extend_shiller_data_with_mode(
        df=df,
        extension_years=extension_years,
        extension_mode=extension_mode or SHILLER_EXTENSION_MODE,
    )


def extend_shiller_data_with_mode(
    df: pd.DataFrame,
    extension_years: int,
    extension_mode: Optional[str],
) -> pd.DataFrame:
    """Extend Shiller data using the requested synthetic-tail strategy."""

    extension_mode = extension_mode or SHILLER_EXTENSION_MODE

    if extension_mode == "average_return":
        return extend_shiller_data_with_average_return(
            df=df,
            extension_years=extension_years,
            lookback_years=SHILLER_AVERAGE_RETURN_LOOKBACK_YEARS,
        )
    if extension_mode == "hard_code_date":
        return extend_shiller_data_with_hard_code_date(
            df=df,
            extension_years=extension_years,
        )
    return extend_shiller_data_with_last_year(df=df, extension_years=extension_years)


def extend_shiller_data_to_cover_date(
    df: pd.DataFrame,
    end_date,
    minimum_extension_years: int = 0,
    extension_mode: Optional[str] = None,
) -> pd.DataFrame:
    """Extend Shiller data enough to cover the requested end date."""

    if df is None:
        return df

    df_sorted = df.loc[:, ~df.columns.duplicated()].copy()
    if df_sorted.empty:
        return df_sorted

    max_date = pd.to_datetime(df_sorted["Date"].max())
    end_ts = pd.to_datetime(end_date)
    months_needed = (int(end_ts.year) - int(max_date.year)) * 12 + (int(end_ts.month) - int(max_date.month))
    required_extension_years = 0 if months_needed <= 0 else (months_needed + 11) // 12
    extension_years = max(int(minimum_extension_years or 0), required_extension_years)

    if extension_years <= 0:
        return df_sorted

    return extend_shiller_data_with_mode(df_sorted, extension_years, extension_mode)


def needs_update(path, days=30):
    """
    Checks if the supplied file exists and is less than 30 days old.
    """
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / (24 * 3600)
    return age_days > days


# Data loading function
def get_cached_shiller_df(session_state):
    """Return Shiller data from session cache, loading if necessary."""
    shiller_df = session_state.get("shiller_df")
    if shiller_df is None:
        shiller_df = load_shiller_data()
        session_state["shiller_df"] = shiller_df
    return shiller_df


def load_shiller_data():
    """
    Loads and preprocesses the Shiller data.
    (Stub - actual implementation will be copied from notebook)
    """
    local_path = Path(os.path.join("tempdir", "shillerdata.xls"))
    # This is the link found in https://shillerdata.com/
    url = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/c9b8cf0f-f01a-49f5-9ea5-d19443390ab2/ie_data.xls?ver=1780495520681")
    # Download if missing or outdated
    if needs_update(local_path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    # Read multi-line headers (rows 4-7) and collapse them
    headers_raw = pd.read_excel(local_path, sheet_name="Data", skiprows=4, nrows=4, header=None)
    headers = (
        headers_raw.fillna("")
        .astype(str)
        .agg(" ".join)  # join rows into one string
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)  # clean spaces
    )

    # Read the actual data (starting row 8)
    df = pd.read_excel(local_path, sheet_name="Data", skiprows=8)
    df.columns = headers
    df = df.loc[:, [str(col).strip() != "" for col in df.columns]].copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Drop the last row which contains footnotes
    df = df.iloc[:-1].reset_index(drop=True)

    # TEMPORARY: Also drop the second to last row which as of 20251004 is a duplicate
    #df = df.iloc[:-1].reset_index(drop=True)

    df = df.rename(columns={
        "Earnings Ratio P/E10 or CAPE": "CAPE",
        "Earnings Ratio TR P/E10 or TR CAPE": "TR CAPE",
    })
    df = df[[
        "Date",
        "Real Total Return Price",
        "Real Total Bond Returns",
        "CAPE",
        "TR CAPE",
    ]].copy()

    df = _fill_missing_cape_values(df)

    df["Date"] = parse_shiller_date(df["Date"])
    return df
