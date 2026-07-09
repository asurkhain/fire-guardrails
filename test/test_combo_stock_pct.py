import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app_settings import Settings
import utils


def _make_shiller_frame(periods=36):
    dates = pd.date_range(start="1980-01-01", periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "Date": dates,
            "Real Total Return Price": np.linspace(100.0, 130.0, periods),
            "Real Total Bond Returns": np.linspace(50.0, 55.0, periods),
            "CAPE": np.full(periods, 20.0),
            "TR CAPE": np.full(periods, 22.0),
        }
    )


def _make_settings(stock_pct=0.9):
    return Settings(
        mode="Historical Mode",
        start_date=pd.Timestamp("1985-01-01").date(),
        retirement_duration_months=12,
        analysis_start_date=pd.Timestamp("1871-01-01").date(),
        initial_value=1_000_000,
        stock_pct=stock_pct,
        target_success_rate=0.8,
        initial_monthly_spending=100000,
        initial_spending_overridden=False,
        upper_guardrail_success=1.0,
        lower_guardrail_success=0.25,
        upper_adjustment_fraction=1.0,
        lower_adjustment_fraction=0.35,
        adjustment_threshold=0.05,
        adjustment_frequency="Monthly",
        spending_cap_option="Unlimited",
        spending_floor_option="Unlimited",
        shiller_extension_mode="hard_code_date",
    )


def test_compute_portfolio_returns_different_stock_pct():
    """Different stock_pct values must produce different portfolio returns."""
    df = _make_shiller_frame(periods=24)
    stock_prices = df["Real Total Return Price"].to_numpy(dtype=float)
    bond_prices = df["Real Total Bond Returns"].to_numpy(dtype=float)

    results = {}
    for pct in [0.0, 0.3, 0.6, 1.0]:
        r = utils.compute_portfolio_returns(stock_prices, bond_prices, pct)
        results[pct] = r

    for pct_a in results:
        for pct_b in results:
            if pct_a < pct_b:
                assert not np.allclose(results[pct_a], results[pct_b]), (
                    f"stock_pct={pct_a} and stock_pct={pct_b} produced identical returns"
                )


def test_compute_portfolio_returns_stock_only_is_stock_returns():
    df = _make_shiller_frame(periods=24)
    stock_prices = df["Real Total Return Price"].to_numpy(dtype=float)
    bond_prices = df["Real Total Bond Returns"].to_numpy(dtype=float)

    stock_returns = np.ones(len(stock_prices))
    stock_returns[1:] = stock_prices[1:] / stock_prices[:-1]
    bond_returns = np.ones(len(bond_prices))
    bond_returns[1:] = bond_prices[1:] / bond_prices[:-1]

    r = utils.compute_portfolio_returns(stock_prices, bond_prices, 1.0)
    assert np.allclose(r, stock_returns)

    r = utils.compute_portfolio_returns(stock_prices, bond_prices, 0.0)
    assert np.allclose(r, bond_returns)


def test_historical_worker_context_uses_stock_pct():
    """_build_historical_worker_context stores portfolio_returns computed with settings.stock_pct."""
    df = _make_shiller_frame()
    settings_50 = _make_settings(stock_pct=0.5)
    settings_70 = _make_settings(stock_pct=0.7)

    context_50 = utils._build_historical_worker_context(df, settings_50)
    context_70 = utils._build_historical_worker_context(df, settings_70)

    try:
        pr_50 = utils.load(context_50.portfolio_returns_path)
        pr_70 = utils.load(context_70.portfolio_returns_path)

        assert pr_50.shape == pr_70.shape
        assert not np.allclose(pr_50, pr_70), (
            "portfolio_returns for stock_pct=0.5 should differ from stock_pct=0.7"
        )
    finally:
        for ctx in (context_50, context_70):
            if ctx.portfolio_returns_path is not None:
                import shutil
                shutil.rmtree(Path(ctx.portfolio_returns_path).parent, ignore_errors=True)


def test_combo_analysis_loop_produces_distinct_keys():
    """Verify run_combo_analysis creates one result per stock_pct with distinct data."""
    df = _make_shiller_frame(periods=24)
    settings = _make_settings(stock_pct=0.8)

    results = utils.run_combo_analysis(
        df=df,
        settings=settings,
        stock_pct_range=(0.3, 0.5),
        upper_guardrail_success_range=(0.90, 0.95),
        lower_guardrail_success_range=(0.20, 0.25),
        target_success_rate_range=(0.80, 0.85),
    )

    stock_pcts_seen = set()
    for key in results:
        assert "sp=" in key
        sp_part = key.split("_")[0]
        stock_pcts_seen.add(sp_part)

    assert len(stock_pcts_seen) > 1, (
        f"Expected multiple stock_pct values, got {stock_pcts_seen}"
    )

    df_by_sp = {}
    for key, result_df in results.items():
        sp_part = key.split("_")[0]
        df_by_sp.setdefault(sp_part, []).append(result_df)

    # Verify at least two stock_pct values produce different End_Portfolio values
    ending_by_sp = {}
    for key, result_df in results.items():
        sp_part = key.split("_")[0]
        ending_by_sp[sp_part] = result_df["Ending_Portfolio"].sum()

    distinct = set(round(v, 2) for v in ending_by_sp.values())
    assert len(distinct) >= 2, (
        f"Expected different Ending_Portfolio across stock_pcts, got: {ending_by_sp}"
    )


def test_run_single_retirement_task_detects_new_context():
    """Simulate loky worker reuse: calling _run_single_retirement_task with
    different context_spec must use the new context, not a stale cached one."""
    df = _make_shiller_frame(periods=24)
    settings_50 = _make_settings(stock_pct=0.5)
    settings_90 = _make_settings(stock_pct=0.9)

    analysis_df = utils.extend_shiller_data_with_mode(
        df=df,
        extension_years=settings_50.historical_future_extension_years or 0,
        extension_mode=settings_50.shiller_extension_mode,
    )
    start_dates = utils.enumerate_historical_retirement_starts(
        df=analysis_df,
        analysis_start_date=settings_50.analysis_start_date,
        duration_months=settings_50.retirement_duration_months,
    )

    context_spec_50 = utils._build_historical_worker_context(df, settings_50, analysis_df=analysis_df)
    context_spec_90 = utils._build_historical_worker_context(df, settings_90, analysis_df=analysis_df)

    temp_dirs = []
    try:
        for spec in (context_spec_50, context_spec_90):
            p = Path(spec.portfolio_returns_path).parent
            temp_dirs.append(p)

        start = start_dates[0]

        # First call — should use context_spec_50
        utils._set_historical_worker_context(None)
        row_50 = utils._run_single_retirement_task(start, context_spec_50)

        # Second call — should detect that context_spec changed and re-materialize
        row_90 = utils._run_single_retirement_task(start, context_spec_90)

        # portfolio_returns arrays must differ
        pr_50 = utils.load(context_spec_50.portfolio_returns_path)
        pr_90 = utils.load(context_spec_90.portfolio_returns_path)
        assert not np.allclose(pr_50, pr_90), (
            "portfolio_returns arrays should differ between stock_pct=0.5 and 0.9"
        )

        # Ending_Portfolio must differ between allocations
        ep_50 = row_50["Ending_Portfolio"]
        ep_90 = row_90["Ending_Portfolio"]
        assert abs(ep_50 - ep_90) > 1.0, (
            f"Ending_Portfolio should differ: 0.5 got {ep_50}, 0.9 got {ep_90}"
        )
    finally:
        for d in temp_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        utils._set_historical_worker_context(None)
