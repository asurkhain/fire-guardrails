import pytest
import pandas as pd
import numpy as np

from tax_utils import (
    add_tax_metrics_to_withdrawal_results,
    apply_account_return_multipliers,
    allocate_account_withdrawal,
    calculate_fpl_health_expense,
    calculate_federal_income_tax,
    fund_spending_with_tax,
    rebalance_retirement_accounts,
    summarize_tax_metrics_from_results,
)


def test_income_tax_applies_standard_deduction_and_progressive_brackets():
    tax = calculate_federal_income_tax(60_000, 0, "single")

    assert tax["taxable_ordinary_income"] == 43_900
    assert tax["ordinary_income_tax"] == pytest.approx(5_020)
    assert tax["total_tax"] == pytest.approx(5_020)


def test_ltcg_is_stacked_on_taxable_ordinary_income():
    tax = calculate_federal_income_tax(70_000, 20_000, "single")

    # $53,900 of ordinary taxable income already fills the 0% LTCG band.
    assert tax["long_term_capital_gains_tax"] == pytest.approx(3_000)


def test_niit_is_applied_to_the_lesser_of_investment_income_or_magi_excess():
    tax = calculate_federal_income_tax(300_000, 100_000, "single")

    assert tax["niit_taxable_income"] == 100_000
    assert tax["net_investment_income_tax"] == pytest.approx(3_800)


def test_pretax_retirement_distribution_is_not_net_investment_income():
    tax = calculate_federal_income_tax(300_000, 0, "single", net_investment_income=0)

    assert tax["net_investment_income_tax"] == 0


def test_fpl_health_expense_starts_above_threshold_and_interpolates_by_age():
    assert calculate_fpl_health_expense(63_840, "single", 42) == 0
    assert calculate_fpl_health_expense(63_841, "single", 42) == pytest.approx(8_000)
    assert calculate_fpl_health_expense(100_000, "single", 53) == pytest.approx(14_570)
    assert calculate_fpl_health_expense(86_561, "mfj", 64) == pytest.approx(42_280)
    assert calculate_fpl_health_expense(100_000, "single", 65) == 0


def test_tax_calculation_reports_fpl_health_expense_separately_from_tax():
    tax = calculate_federal_income_tax(70_000, 0, "single", age=42)

    assert tax["fpl_health_expense"] == pytest.approx(8_000)
    assert tax["total_tax_and_fpl_health_expense"] == pytest.approx(tax["total_tax"] + 8_000)


def test_withdrawal_order_and_tax_character_follow_account_rules():
    accounts = [
        {"id": "taxable-equity", "account_type": "taxable", "asset_type": "equity", "cost_basis": 60, "total": 100},
        {"id": "roth", "account_type": "retirement post tax", "asset_type": "equity", "cost_basis": 30, "total": 50},
        {"id": "traditional", "account_type": "retirement pretax", "asset_type": "bond", "cost_basis": 0, "total": 100},
    ]

    result = allocate_account_withdrawal(accounts, 170)

    assert [(item["account_id"], item["phase"], item["amount"]) for item in result["transactions"]] == [
        ("taxable-equity", "taxable_equity_to_fpl_limit", 100.0),
        ("roth", "post_tax_basis", 30.0),
        ("traditional", "pretax", 40.0),
    ]
    assert result["long_term_capital_gains"] == 40.0
    assert result["ordinary_income"] == 40.0


def test_bond_gain_is_ordinary_income_and_roth_growth_is_last():
    accounts = [
        {"id": "bond", "account_type": "taxable", "asset_type": "bond", "cost_basis": 80, "total": 100},
        {"id": "roth", "account_type": "retirement post tax", "asset_type": "equity", "cost_basis": 20, "total": 50},
    ]

    result = allocate_account_withdrawal(accounts, 150)

    assert [item["phase"] for item in result["transactions"]] == [
        "taxable_bond_gain",
        "taxable_bond_basis",
        "post_tax_basis",
        "post_tax_growth",
    ]
    # Taxable-bond gains are settled as income before the withdrawal.
    assert result["ordinary_income"] == 20.0
    assert result["long_term_capital_gains"] == 0.0
    roth = result["accounts"][1]
    assert roth.total == 0.0
    assert roth.cost_basis == 0.0


def test_taxable_bond_return_is_immediately_income_and_added_to_basis():
    accounts = [{"id": "bond", "account_type": "taxable", "asset_type": "bond", "cost_basis": 100, "total": 100}]

    result = apply_account_return_multipliers(accounts, 1.0, 1.05)

    assert result["taxable_bond_income"] == pytest.approx(5)
    assert result["accounts"][0].total == pytest.approx(105)
    assert result["accounts"][0].cost_basis == pytest.approx(105)
    withdrawal = allocate_account_withdrawal(result["accounts"], 105)
    assert withdrawal["ordinary_income"] == 0


def test_taxable_withdrawal_uses_bond_gain_then_equity_to_fpl_limit_then_bond_basis():
    accounts = [
        {"id": "bond-gain", "account_type": "taxable", "asset_type": "bond", "cost_basis": 8_000, "total": 10_000},
        {"id": "equity", "account_type": "taxable", "asset_type": "equity", "cost_basis": 0, "total": 100_000},
        {"id": "bond-basis", "account_type": "taxable", "asset_type": "bond", "cost_basis": 100_000, "total": 100_000},
    ]

    result = allocate_account_withdrawal(accounts, 100_000, fpl_income_limit=63_840)

    assert [item["phase"] for item in result["transactions"]] == [
        "taxable_bond_gain",
        "taxable_equity_to_fpl_limit",
        "taxable_bond_basis",
        "taxable_bond_basis",
    ]
    assert result["long_term_capital_gains"] == pytest.approx(61_840)
    assert result["ordinary_income"] == pytest.approx(2_000)


def test_taxable_equity_covers_withdrawal_after_taxable_bond_basis_is_zero():
    accounts = [
        {"id": "bond", "account_type": "taxable", "asset_type": "bond", "cost_basis": 0, "total": 10_000},
        {"id": "equity", "account_type": "taxable", "asset_type": "equity", "cost_basis": 10_000, "total": 50_000},
    ]

    result = allocate_account_withdrawal(accounts, 20_000)

    assert [(item["account_id"], item["phase"], item["amount"]) for item in result["transactions"]] == [
        ("bond", "taxable_bond_gain", 10_000.0),
        ("equity", "taxable_equity_to_fpl_limit", 10_000.0),
    ]


def test_rebalancing_leaves_taxable_accounts_unchanged_and_prioritizes_retirement_accounts():
    accounts = [
        {"id": "taxable", "account_type": "taxable", "asset_type": "equity", "cost_basis": 20, "total": 20},
        {"id": "pretax", "account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0, "total": 80},
        {"id": "posttax", "account_type": "retirement post tax", "asset_type": "bond", "cost_basis": 50, "total": 100},
    ]

    add_bonds = rebalance_retirement_accounts(accounts, 0.40)
    taxable_after = next(account for account in add_bonds["accounts"] if account.account_id == "taxable")
    assert taxable_after.asset_type == "equity" and taxable_after.total == 20
    assert add_bonds["transactions"][0]["account_type"] == "retirement_pretax"
    assert add_bonds["transactions"][0]["to_asset_type"] == "bond"

    add_equity = rebalance_retirement_accounts(accounts, 0.70)
    assert add_equity["transactions"][0]["account_type"] == "retirement_post_tax"
    assert add_equity["transactions"][0]["to_asset_type"] == "equity"


def test_tax_aware_funding_rebalances_after_withdrawal_when_target_is_supplied():
    accounts = [
        {"id": "taxable", "account_type": "taxable", "asset_type": "equity", "cost_basis": 20, "total": 20},
        {"id": "pretax", "account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0, "total": 80},
        {"id": "posttax", "account_type": "retirement post tax", "asset_type": "bond", "cost_basis": 50, "total": 100},
    ]

    result = fund_spending_with_tax(accounts, 0, "single", target_stock_pct=0.40)

    assert result["rebalancing"]["transactions"][0]["account_type"] == "retirement_pretax"
    assert result["rebalancing"]["transactions"][0]["to_asset_type"] == "bond"


def test_tax_metrics_add_tax_paid_tax_inclusive_draw_and_rate_columns():
    results = pd.DataFrame({
        "Withdrawal": [20_000.0],
        "Fixed_SR_Withdrawal": [21_000.0],
        "CAPE_Withdrawal": [22_000.0],
        "VPW_Withdrawal": [23_000.0],
        "Date": [pd.Timestamp("2026-01-01")],
    })
    accounts = [{"account_type": "retirement pretax", "asset_type": "bond", "cost_basis": 0, "total": 100_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    assert enriched.loc[0, "Tax_Paid"] > 0
    assert enriched.loc[0, "Tax_Inclusive_Withdrawal"] > enriched.loc[0, "Withdrawal"]
    assert enriched.loc[0, "Tax_Rate"] == pytest.approx(
        enriched.loc[0, "Tax_Paid"] / enriched.loc[0, "Tax_Inclusive_Withdrawal"]
    )
    for prefix in ("Fixed_", "CAPE_", "VPW_"):
        assert enriched.loc[0, f"{prefix}Tax_Paid"] > 0
        assert enriched.loc[0, f"{prefix}Tax_Rate"] > 0
        assert f"{prefix}ACA_Surcharge" in enriched.columns


def test_tax_metrics_use_year_to_date_income_instead_of_a_fresh_deduction_each_month():
    results = pd.DataFrame({
        "Withdrawal": [20_000.0] * 12,
        "Date": pd.date_range("2026-01-01", periods=12, freq="MS"),
    })
    accounts = [{"account_type": "taxable", "asset_type": "equity", "cost_basis": 0, "total": 500_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    assert enriched["Tax_Paid"].sum() > 0
    assert enriched["Tax_Rate"].iloc[-1] > 0


def test_tax_metrics_continue_to_fund_spending_when_baseline_withdrawal_is_zero():
    results = pd.DataFrame({
        "Withdrawal": [0.0] * 12,
        "Total_Spending": [20_000.0] * 12,
        "Net_Cashflow": [0.0] * 12,
        "Date": pd.date_range("2026-01-01", periods=12, freq="MS"),
    })
    accounts = [{"account_type": "taxable", "asset_type": "equity", "cost_basis": 0, "total": 500_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    assert enriched["Tax_Inclusive_Withdrawal"].sum() > 0
    assert enriched["Tax_Paid"].sum() > 0


def test_tax_metrics_track_aca_surcharge_separately_from_regular_tax():
    results = pd.DataFrame({
        "Withdrawal": [10_000.0] * 12,
        "Date": pd.date_range("2026-01-01", periods=12, freq="MS"),
    })
    accounts = [{"account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0, "total": 500_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    assert enriched["ACA_Surcharge"].sum() == pytest.approx(8_000)


def test_funding_grosses_up_account_draw_for_tax():
    accounts = [{"account_type": "retirement pretax", "asset_type": "bond", "cost_basis": 0, "total": 100_000}]

    result = fund_spending_with_tax(accounts, 20_000, "single")

    assert result["after_tax_cash"] == pytest.approx(20_000)
    assert result["gross_withdrawal"] > 20_000
    assert result["spending_shortfall"] == pytest.approx(0)


def test_tax_metrics_advances_age_and_interpolates_aca_surcharge():
    # 24 months across 2 calendar years, starting at age 42
    results = pd.DataFrame({
        "Withdrawal": [10_000.0] * 24,
        "Date": pd.date_range("2026-01-01", periods=24, freq="MS"),
    })
    accounts = [{"account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0, "total": 500_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    year1_surcharge = enriched.iloc[:12]["ACA_Surcharge"].sum()
    year2_surcharge = enriched.iloc[12:]["ACA_Surcharge"].sum()

    # Age 42: 8,000; Age 43: 8,000 + 1 * (21,140 - 8,000) / 22 = 8,597.27
    assert year1_surcharge == pytest.approx(8_000.0)
    assert year2_surcharge == pytest.approx(8_000.0 + (43 - 42) * (21_140.0 - 8_000.0) / 22.0)


def test_tax_metrics_eliminates_aca_surcharge_at_age_65():
    # Start at age 64 for 2 years (age 64 in year 1, age 65 in year 2)
    results = pd.DataFrame({
        "Withdrawal": [10_000.0] * 24,
        "Date": pd.date_range("2026-01-01", periods=24, freq="MS"),
    })
    accounts = [{"account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0, "total": 500_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 64)

    year1_surcharge = enriched.iloc[:12]["ACA_Surcharge"].sum()
    year2_surcharge = enriched.iloc[12:]["ACA_Surcharge"].sum()

    assert year1_surcharge == pytest.approx(21_140.0)
    assert year2_surcharge == pytest.approx(0.0)


def test_tax_metrics_distribute_taxes_evenly_across_months():
    # 12 months with equal spending
    results = pd.DataFrame({
        "Withdrawal": [17_000.0] * 12,
        "Date": pd.date_range("1968-01-01", periods=12, freq="MS"),
    })
    accounts = [{"account_type": "taxable", "asset_type": "equity", "cost_basis": 1_057_000, "total": 3_014_000}]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    # Monthly gross withdrawals and taxes should be equal across all 12 months (no single-month spike)
    for i in range(1, 12):
        assert enriched["Tax_Inclusive_Withdrawal"].iloc[i] == pytest.approx(enriched["Tax_Inclusive_Withdrawal"].iloc[0])
        assert enriched["Tax_Paid"].iloc[i] == pytest.approx(enriched["Tax_Paid"].iloc[0])
        assert enriched["ACA_Surcharge"].iloc[i] == pytest.approx(enriched["ACA_Surcharge"].iloc[0])


def test_tax_metrics_never_zeros_out_when_spending_is_active():
    # 50 years (600 months) with active withdrawals
    results = pd.DataFrame({
        "Withdrawal": [15_000.0] * 600,
        "Date": pd.date_range("1968-01-01", periods=600, freq="MS"),
    })
    accounts = [
        {"account_type": "taxable", "asset_type": "equity", "cost_basis": 1_057_000, "total": 3_014_000},
        {"account_type": "retirement_pretax", "asset_type": "equity", "cost_basis": 657_000, "total": 657_000},
    ]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    # Tax inclusive withdrawal should never drop below the requested net withdrawal
    assert (enriched["Tax_Inclusive_Withdrawal"] >= enriched["Withdrawal"]).all()
    # In the later years (e.g. year 40-50), taxes and gross draws should still be active
    assert (enriched["Tax_Inclusive_Withdrawal"].iloc[-120:] > 0).all()
    assert (enriched["Tax_Paid"].iloc[-120:] > 0).all()
    assert (enriched["Tax_Rate"].iloc[-120:] > 0).all()


def test_tax_metrics_treat_recurring_cashflows_as_ordinary_income():
    accounts = [{"account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0, "total": 500_000}]
    base_results = pd.DataFrame({
        "Withdrawal": [2_000.0] * 12,
        "Total_Spending": [2_000.0] * 12,
        "Net_Cashflow": [0.0] * 12,
        "Date": pd.date_range("2026-01-01", periods=12, freq="MS"),
    })
    cashflow_results = base_results.copy()
    cashflow_results["Net_Cashflow"] = 8_000.0
    cashflow_results["Total_Spending"] = 10_000.0
    cashflow_results["Withdrawal"] = 2_000.0

    no_cashflow = add_tax_metrics_to_withdrawal_results(base_results, accounts, "single", 42)
    with_cashflow = add_tax_metrics_to_withdrawal_results(cashflow_results, accounts, "single", 42)

    assert with_cashflow["Tax_Paid"].sum() > no_cashflow["Tax_Paid"].sum()
    assert with_cashflow["ACA_Surcharge"].sum() > no_cashflow["ACA_Surcharge"].sum()


def test_tax_metrics_track_taxable_account_depletion():
    results = pd.DataFrame({
        "Withdrawal": [50_000.0] * 6,
        "Total_Spending": [50_000.0] * 6,
        "Net_Cashflow": [0.0] * 6,
        "Date": pd.date_range("2026-01-01", periods=6, freq="MS"),
    })
    accounts = [
        {"account_type": "taxable", "asset_type": "equity", "cost_basis": 0.0, "total": 100_000.0},
        {"account_type": "retirement pretax", "asset_type": "equity", "cost_basis": 0.0, "total": 500_000.0},
    ]

    enriched = add_tax_metrics_to_withdrawal_results(results, accounts, "single", 42)

    assert "Taxable_Accounts_Balance" in enriched.columns
    assert len(enriched["Taxable_Accounts_Balance"]) == len(results)
    assert float(enriched["Taxable_Accounts_Balance"].max()) > 0.0
    assert float(enriched["Taxable_Accounts_Balance"].min()) == pytest.approx(0.0)
    assert (enriched["Taxable_Accounts_Balance"] <= 1e-9).any()


def test_bond_basis_only_drawn_if_staying_under_fpl_cap():
    accounts = [
        {"id": "equity", "account_type": "taxable", "asset_type": "equity", "cost_basis": 0, "total": 100_000},
        {"id": "bond", "account_type": "taxable", "asset_type": "bond", "cost_basis": 20_000, "total": 20_000},
    ]

    # Case 1: Remaining withdrawal ($20,000) <= available bond basis ($20,000)
    # Allows staying under FPL limit ($30,000), so bond basis is drawn.
    res_under = allocate_account_withdrawal(accounts, 50_000, fpl_income_limit=30_000)
    assert [t["phase"] for t in res_under["transactions"]] == [
        "taxable_equity_to_fpl_limit",
        "taxable_bond_basis",
    ]
    assert res_under["long_term_capital_gains"] == pytest.approx(30_000)

    # Case 2: Remaining withdrawal ($50,000) > available bond basis ($20,000)
    # Cannot stay under FPL limit, so bond basis is preserved and taxable equity is drawn first.
    res_over = allocate_account_withdrawal(accounts, 80_000, fpl_income_limit=30_000)
    assert [t["phase"] for t in res_over["transactions"]] == [
        "taxable_equity_to_fpl_limit",
        "taxable_equity_after_fpl_limit",
    ]
    # Bond basis account was preserved
    bond_acc = [a for a in res_over["accounts"] if a.account_id == "bond"][0]
    assert bond_acc.total == pytest.approx(20_000)


def test_summarize_tax_metrics_splits_federal_tax_and_aca_surcharge():
    results = pd.DataFrame({
        "Tax_Paid": [5_000.0, 5_000.0],
        "ACA_Surcharge": [1_000.0, 1_000.0],
        "Tax_Inclusive_Withdrawal": [20_000.0, 20_000.0],
        "Portfolio_Value": [900_000.0, 880_000.0],
        "Withdrawal": [10_000.0, 10_000.0],
        "Date": pd.date_range("2026-01-01", periods=2, freq="YS"),
    })

    summary = summarize_tax_metrics_from_results(results, initial_portfolio=1_000_000, current_age=42)

    assert summary["Total_Tax_Paid"] == pytest.approx(8_000.0)
    assert summary["Total_ACA_Surcharge"] == pytest.approx(2_000.0)
    assert summary["Total_Gross_Withdrawal"] == pytest.approx(40_000.0)
    assert summary["Lifetime_Tax_Rate"] == pytest.approx(0.25)
    assert summary["Pre65_Lifetime_Tax_Rate"] == pytest.approx(0.25)
    assert summary["Avg_Annual_Tax_Pct"] == pytest.approx(0.005)
    assert summary["Pre65_Cliff_Years"] == 2
    assert summary["Pre65_Subsidized_Years"] == 0


def test_summarize_tax_metrics_counts_subsidized_and_medicare_transition_year():
    results = pd.DataFrame({
        "Tax_Paid": [1_000.0] * 36,
        "ACA_Surcharge": [0.0] * 12 + [8_000.0 / 12] * 12 + [0.0] * 12,
        "Tax_Inclusive_Withdrawal": [10_000.0] * 36,
        "Portfolio_Value": [1_000_000.0] * 36,
        "Withdrawal": [5_000.0] * 36,
        "Date": pd.date_range("2024-01-01", periods=36, freq="MS"),
    })

    summary = summarize_tax_metrics_from_results(results, initial_portfolio=1_000_000, current_age=64)

    assert summary["Pre65_Subsidized_Years"] == 1
    assert summary["Pre65_Cliff_Years"] == 0
    assert summary["Medicare_Transition_Year"] == 2026


def test_summarize_tax_metrics_pre65_effective_tax_rate_excludes_post_medicare_years():
    results = pd.DataFrame({
        "Tax_Paid": [2_000.0] * 12 + [500.0] * 12,
        "ACA_Surcharge": [0.0] * 24,
        "Tax_Inclusive_Withdrawal": [10_000.0] * 24,
        "Portfolio_Value": [1_000_000.0] * 24,
        "Withdrawal": [5_000.0] * 24,
        "Date": pd.date_range("2024-01-01", periods=24, freq="MS"),
    })

    summary = summarize_tax_metrics_from_results(results, initial_portfolio=1_000_000, current_age=64)

    assert summary["Lifetime_Tax_Rate"] == pytest.approx(0.125)
    assert summary["Pre65_Lifetime_Tax_Rate"] == pytest.approx(0.20)
    assert summary["Avg_Annual_Tax_Pct"] == pytest.approx(0.015)
    assert summary["Pre65_Avg_Annual_Tax_Pct"] == pytest.approx(0.024)


def test_historical_taxable_worker_records_per_cohort_tax_metrics(monkeypatch):
    from app_settings import Settings
    from tax_utils import _run_single_retirement_task, HistoricalWorkerContext

    start = pd.Timestamp("1968-01-01").date()
    settings = Settings(
        mode="Taxable Historical Mode",
        start_date=start,
        retirement_duration_months=24,
        analysis_start_date=pd.Timestamp("1871-01-01").date(),
        initial_value=1_000_000.0,
        stock_pct=0.90,
        target_success_rate=0.80,
        initial_monthly_spending=3_333.33,
        initial_spending_overridden=False,
        upper_guardrail_success=1.0,
        lower_guardrail_success=0.25,
        upper_adjustment_fraction=0.5,
        lower_adjustment_fraction=0.35,
        adjustment_threshold=0.05,
        adjustment_frequency="Quarterly",
        spending_cap_option="Unlimited",
        spending_floor_option="Unlimited",
        fixed_monthly_withdrawal=3_333.33,
        accounts=[
            {
                "account_type": "retirement pretax",
                "asset_type": "equity",
                "cost_basis": 0.0,
                "total": 1_000_000.0,
            }
        ],
        marital_status="single",
        current_age=42.0,
    )

    minimal_df = pd.DataFrame({
        "Date": pd.date_range("1968-01-01", periods=48, freq="MS"),
        "Real Total Return Price": np.linspace(100.0, 120.0, 48),
        "Real Total Bond Returns": np.linspace(100.0, 105.0, 48),
        "CAPE": np.full(48, 20.0),
    })

    context = HistoricalWorkerContext(
        analysis_df_path=None,
        settings=settings,
        cashflows=[],
        conditional_configs=[],
        monthly_cashflows=np.zeros(24),
        all_stock_prices=minimal_df["Real Total Return Price"].to_numpy(),
        all_bond_prices=minimal_df["Real Total Bond Returns"].to_numpy(),
        portfolio_returns=np.ones(48),
        all_cape_values=np.full(48, 20.0),
        analysis_df=minimal_df,
    )

    monkeypatch.setattr("tax_utils._HISTORICAL_WORKER_CONTEXT", context)
    monkeypatch.setattr(
        "tax_utils.get_guardrail_withdrawals",
        lambda **kwargs: pd.DataFrame({
            "Date": pd.date_range(start, periods=24, freq="MS"),
            "Withdrawal": [10_000.0] * 24,
            "Total_Spending": [10_000.0] * 24,
            "Net_Cashflow": [0.0] * 24,
            "Portfolio_Value": np.linspace(1_000_000.0, 900_000.0, 24),
            "Guardrail_Success": [True] * 24,
            "Fixed_SR_Withdrawal": [10_000.0] * 24,
            "Fixed_SR_Total_Spending": [10_000.0] * 24,
            "CAPE_Withdrawal": [10_000.0] * 24,
            "CAPE_Total_Spending": [10_000.0] * 24,
            "VPW_Withdrawal": [10_000.0] * 24,
            "VPW_Total_Spending": [10_000.0] * 24,
        }),
    )
    monkeypatch.setattr(
        "tax_utils.simulate_fixed_withdrawal_retirement",
        lambda **kwargs: {"success": True, "ending_value": 500_000.0},
    )
    monkeypatch.setattr(
        "tax_utils.simulate_cape_withdrawal_retirement",
        lambda **kwargs: {
            "success": True,
            "ending_value": 500_000.0,
            "min_withdrawal": 8_000.0,
            "max_withdrawal": 12_000.0,
            "avg_withdrawal": 10_000.0,
            "median_withdrawal": 10_000.0,
            "pct_below_fixed": 0.1,
        },
    )
    monkeypatch.setattr(
        "tax_utils.simulate_vpw_withdrawal_retirement",
        lambda **kwargs: {
            "success": True,
            "ending_value": 500_000.0,
            "min_withdrawal": 8_000.0,
            "max_withdrawal": 12_000.0,
            "avg_withdrawal": 10_000.0,
            "median_withdrawal": 10_000.0,
            "pct_below_fixed": 0.1,
        },
    )

    row = _run_single_retirement_task(start)

    assert row["Total_Tax_Paid"] > 0
    assert row["Total_ACA_Surcharge"] >= 0
    assert row["Lifetime_Tax_Rate"] > 0
    assert row["Avg_Annual_Tax_Pct"] > 0
    assert row["Pre65_Cliff_Years"] >= 0
    for prefix in ("Fixed_", "CAPE_", "VPW_"):
        assert row[f"{prefix}Total_Tax_Paid"] > 0
        assert row[f"{prefix}Lifetime_Tax_Rate"] > 0
        assert row[f"{prefix}Avg_Annual_Tax_Pct"] > 0
