import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app_settings import Settings
import utils


def _make_settings(mode="Historical Mode"):
    return Settings(
        mode=mode,
        start_date=pd.Timestamp("1985-01-01").date(),
        retirement_duration_months=12,
        analysis_start_date=pd.Timestamp("1871-01-01").date(),
        initial_value=1_000_000,
        stock_pct=0.9,
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


def _make_shiller_frame(periods=24):
    dates = pd.date_range(start="1980-01-01", periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "Date": dates,
            "Real Total Return Price": np.linspace(100.0, 130.0, periods),
            "Real Total Bond Returns": np.linspace(50.0, 60.0, periods),
            "CAPE": np.full(periods, 20.0),
            "TR CAPE": np.full(periods, 22.0),
        }
    )


def test_historical_worker_context_does_not_pickle_analysis_df():
    df = _make_shiller_frame()
    settings = _make_settings()

    context = utils._build_historical_worker_context(df, settings, analysis_df=df)
    try:
        assert context.analysis_df is None
        assert context.analysis_df_path is not None
        assert Path(context.analysis_df_path).exists()
    finally:
        if context.analysis_df_path is not None:
            shutil.rmtree(Path(context.analysis_df_path).parent, ignore_errors=True)


def test_historical_worker_task_lazily_initializes_context(monkeypatch):
    df = _make_shiller_frame()
    settings = _make_settings()
    context_spec = utils._build_historical_worker_context(df, settings, analysis_df=df)
    utils._set_historical_worker_context(None)

    fake_results = pd.DataFrame(
        {
            "Withdrawal": [100.0],
            "Portfolio_Value": [900.0],
            "Total_Spending": [120.0],
            "Guardrail_Success": [True],
        }
    )

    monkeypatch.setattr(utils, "get_guardrail_withdrawals", lambda **kwargs: fake_results)
    monkeypatch.setattr(utils, "_spending_stats_from_results", lambda results_df: {"min_spending": 120.0, "max_spending": 120.0, "median_spending": 120.0, "avg_spending": 120.0})
    monkeypatch.setattr(utils, "_guardrail_run_success", lambda results_df, final_value_target: (True, 900.0))
    monkeypatch.setattr(utils, "simulate_fixed_withdrawal_retirement", lambda **kwargs: {"success": True, "ending_value": 800.0})
    monkeypatch.setattr(utils, "simulate_cape_withdrawal_retirement", lambda **kwargs: {
        "success": True,
        "ending_value": 850.0,
        "min_withdrawal": 1.0,
        "max_withdrawal": 2.0,
        "avg_withdrawal": 1.5,
        "median_withdrawal": 1.5,
        "pct_below_fixed": 0.0,
    })

    try:
        row = utils._run_single_retirement_task(pd.Timestamp("1985-01-01").date(), context_spec)

        assert row["Start_Year"] == 1985
        assert row["Guardrail_Success"] is True
        assert utils._HISTORICAL_WORKER_CONTEXT is not None
    finally:
        utils._set_historical_worker_context(None)
        if context_spec.analysis_df_path is not None:
            shutil.rmtree(Path(context_spec.analysis_df_path).parent, ignore_errors=True)


def test_spending_rate_solver_uses_previous_rate_and_precomputed_returns(monkeypatch):
    df = _make_shiller_frame()
    calls = []

    def fake_calculate_success_rate(
        df,
        withdrawal_rate,
        num_months,
        stock_pct=0.75,
        analysis_start_date="1871-01-01",
        initial_value=1_000_000,
        monthly_cashflows=None,
        final_value_target=0.0,
        portfolio_returns=None,
    ):
        calls.append(withdrawal_rate)
        return 1.0 - float(withdrawal_rate)

    def fail_compute_portfolio_returns(*args, **kwargs):
        raise AssertionError("compute_portfolio_returns should not be called when portfolio_returns is supplied")

    monkeypatch.setattr(utils, "calculate_success_rate", fake_calculate_success_rate)
    monkeypatch.setattr(utils, "compute_portfolio_returns", fail_compute_portfolio_returns)

    result = utils.get_spending_rate_for_fixed_success_rate(
        df=df,
        desired_success_rate=0.65,
        num_months=6,
        initial_value=1_000_000,
        stock_pct=0.9,
        cashflows=[],
        cashflow_schedule=np.zeros(6, dtype=np.float64),
        portfolio_returns=np.linspace(1.0, 1.02, len(df)),
        previous_spending_rate=0.4,
        final_value_target=0.0,
    )

    assert calls[0] == pytest.approx(0.4)
    assert result["spending_rate"] == pytest.approx(0.35, abs=0.01)


def test_upper_guardrail_cap_limits_infinite_values():
    capped = utils._cap_upper_guardrail_value(
        upper_guardrail_value=float("inf"),
        current_portfolio_value=200_000.0,
        initial_portfolio_value=1_000_000.0,
    )

    assert capped == pytest.approx(1_000_000.0)

    capped = utils._cap_upper_guardrail_value(
        upper_guardrail_value=5_000_000.0,
        current_portfolio_value=900_000.0,
        initial_portfolio_value=1_000_000.0,
    )

    assert capped == pytest.approx(1_800_000.0)


def test_is_successful_ending_value_uses_positive_portfolio_rule():
    assert utils.is_successful_ending_value(145_685.0) == (True, pytest.approx(145_685.0))
    assert utils.is_successful_ending_value(0.0) == (True, pytest.approx(0.0))
    assert utils.is_successful_ending_value(-1.0) == (False, pytest.approx(0.0))


def test_final_value_target_ramps_over_final_ten_percent():
    full_target = utils.get_effective_final_value_target(
        final_value_target=100_000.0,
        total_retirement_months=360,
        months_remaining=360,
        ramp_down=True,
    )
    halfway_target = utils.get_effective_final_value_target(
        final_value_target=100_000.0,
        total_retirement_months=360,
        months_remaining=18,
        ramp_down=True,
    )
    zero_target = utils.get_effective_final_value_target(
        final_value_target=100_000.0,
        total_retirement_months=360,
        months_remaining=0,
        ramp_down=True,
    )
    disabled_target = utils.get_effective_final_value_target(
        final_value_target=100_000.0,
        total_retirement_months=360,
        months_remaining=18,
        ramp_down=False,
    )

    assert full_target == pytest.approx(100_000.0)
    assert halfway_target == pytest.approx(50_000.0)
    assert zero_target == pytest.approx(0.0)
    assert disabled_target == pytest.approx(100_000.0)


def test_vpw_comparison_columns_are_added_to_guardrail_results():
    df = _make_shiller_frame(periods=24)
    settings = Settings(
        mode="Historical Mode",
        start_date=pd.Timestamp("1980-01-01").date(),
        retirement_duration_months=12,
        analysis_start_date=pd.Timestamp("1871-01-01").date(),
        initial_value=1_000_000,
        stock_pct=0.9,
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
    all_stock_prices = df["Real Total Return Price"].to_numpy(dtype=float)
    all_bond_prices = df["Real Total Bond Returns"].to_numpy(dtype=float)
    portfolio_returns = utils.compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)

    results = utils.get_guardrail_withdrawals(
        df=df,
        settings=settings,
        analysis_df=df,
        cashflows=[],
        conditional_configs=[],
        monthly_cashflows=np.zeros(settings.retirement_duration_months, dtype=np.float64),
        all_stock_prices=all_stock_prices,
        all_bond_prices=all_bond_prices,
        portfolio_returns=portfolio_returns,
    )

    expected_columns = {
        "VPW_Value",
        "VPW_Withdrawal",
        "VPW_Total_Spending",
        "Guardrail_Success",
        "Fixed_Success",
        "CAPE_Success",
        "VPW_Success",
    }
    assert expected_columns.issubset(results.columns)


def test_vpw_single_path_simulation_returns_summary_fields():
    df = _make_shiller_frame(periods=24)
    settings = Settings(
        mode="Simulation Mode",
        start_date=pd.Timestamp("1980-01-01").date(),
        retirement_duration_months=12,
        analysis_start_date=pd.Timestamp("1871-01-01").date(),
        initial_value=1_000_000,
        stock_pct=0.9,
        target_success_rate=0.8,
        initial_monthly_spending=4000,
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
    all_stock_prices = df["Real Total Return Price"].to_numpy(dtype=float)
    all_bond_prices = df["Real Total Bond Returns"].to_numpy(dtype=float)
    portfolio_returns = utils.compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)

    result = utils.simulate_vpw_withdrawal_retirement(
        df=df,
        settings=settings,
        analysis_df=df,
        monthly_cashflows=np.zeros(settings.retirement_duration_months, dtype=np.float64),
        all_stock_prices=all_stock_prices,
        all_bond_prices=all_bond_prices,
        portfolio_returns=portfolio_returns,
    )

    assert set(["success", "ending_value", "min_withdrawal", "max_withdrawal", "avg_withdrawal", "median_withdrawal", "pct_below_fixed"]).issubset(result)


def test_vpw_historical_pct_below_fixed_uses_annual_baseline():
    df = _make_shiller_frame(periods=24)
    settings = Settings(
        mode="Historical Mode",
        start_date=pd.Timestamp("1980-01-01").date(),
        retirement_duration_months=12,
        analysis_start_date=pd.Timestamp("1871-01-01").date(),
        initial_value=1_000_000,
        stock_pct=0.9,
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

    result = utils.simulate_vpw_withdrawal_retirement(
        df=df,
        settings=settings,
        analysis_df=df,
        monthly_cashflows=np.zeros(settings.retirement_duration_months, dtype=np.float64),
        all_stock_prices=df["Real Total Return Price"].to_numpy(dtype=float),
        all_bond_prices=df["Real Total Bond Returns"].to_numpy(dtype=float),
        portfolio_returns=utils.compute_portfolio_returns(
            df["Real Total Return Price"].to_numpy(dtype=float),
            df["Real Total Bond Returns"].to_numpy(dtype=float),
            settings.stock_pct,
        ),
    )

    assert result["pct_below_fixed"] > 0.0
