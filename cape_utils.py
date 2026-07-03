from __future__ import annotations

import numpy as np
import pandas as pd
from helpers import is_successful_ending_value_with_clamp

from app_settings import Settings


def _cashflow_schedule_for_window(cashflows, window_length, start_offset=0):
    """Build a per-month cashflow schedule for a retirement window."""
    schedule = np.zeros(int(window_length), dtype=np.float64)
    if not cashflows or window_length <= 0:
        return schedule

    for flow in cashflows:
        try:
            start = int(flow.get("start_month", 0))
            end = int(flow.get("end_month", -1))
            amount = float(flow.get("amount", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        if end < start:
            continue

        adj_start = max(start - int(start_offset), 0)
        adj_end = end - int(start_offset)

        if adj_end < 0 or adj_start >= window_length:
            continue

        adj_end = min(adj_end, int(window_length) - 1)
        if adj_end < adj_start:
            continue

        schedule[adj_start:adj_end + 1] += amount

    return schedule


def _compute_portfolio_returns(stock_prices, bond_prices, stock_pct):
    """Compute blended portfolio returns from stock and bond price series."""
    stock_returns = np.ones(len(stock_prices))
    stock_returns[1:] = stock_prices[1:] / stock_prices[:-1]

    bond_returns = np.ones(len(bond_prices))
    bond_returns[1:] = bond_prices[1:] / bond_prices[:-1]

    return float(stock_pct) * stock_returns + (1 - float(stock_pct)) * bond_returns


def calculate_cape_withdrawal_rate(cape: float) -> float:
    return 0.0175 + (0.5 / cape)


def simulate_cape_withdrawal_retirement(
    df: pd.DataFrame,
    settings: Settings,
    analysis_df=None,
    monthly_cashflows=None,
    all_cape_values=None,
    all_stock_prices=None,
    all_bond_prices=None,
    portfolio_returns=None,
) -> dict:
    """Run a cape monthly withdrawal path for one retirement window."""
    start_date = pd.to_datetime(settings.start_date)
    end_date = settings.retirement_end_date()
    if end_date is None:
        raise ValueError("Unable to determine retirement end date from settings.")

    source_df = analysis_df if analysis_df is not None else df
    mask = (source_df["Date"] >= start_date) & (source_df["Date"] <= pd.to_datetime(end_date))
    subset = source_df[mask].copy()
    subset["_Full_Index"] = subset.index
    subset = subset.reset_index(drop=True)
    total_months = len(subset)
    if total_months <= 0:
        return {"success": False, "ending_value": 0.0}

    if monthly_cashflows is None:
        cashflows = settings.cashflows_for_calculation()
        monthly_cashflows = _cashflow_schedule_for_window(cashflows, total_months, 0)
    else:
        monthly_cashflows = np.asarray(monthly_cashflows, dtype=np.float64)
    if all_cape_values is None:
        all_cape_values = source_df["CAPE"].values
    if all_stock_prices is None:
        all_stock_prices = source_df["Real Total Return Price"].values
    if all_bond_prices is None:
        all_bond_prices = source_df["Real Total Bond Returns"].values
    if portfolio_returns is None:
        portfolio_returns = _compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)

    portfolio_value = float(settings.initial_value)
    cap_multiplier = settings.spending_cap_multiplier
    floor_multiplier = settings.spending_floor_multiplier
    baseline_monthly_withdrawal = float(settings.initial_monthly_spending)
    cap_amount = (
        baseline_monthly_withdrawal * float(cap_multiplier)
        if cap_multiplier is not None
        else None
    )
    floor_amount = (
        baseline_monthly_withdrawal * float(floor_multiplier)
        if floor_multiplier is not None
        else None
    )
    depleted = False
    yearly_withdrawals = []

    full_indices = subset["_Full_Index"].to_numpy(dtype=np.int64, copy=False)

    for i in range(total_months):
        if i % 12 == 0:
            yearly_withdrawals.append(0.0)
        if depleted:
            break
        full_idx = int(full_indices[i])
        cape_idx = min(full_idx + 1, len(all_cape_values) - 1)
        swr = calculate_cape_withdrawal_rate(all_cape_values[cape_idx])
        monthly_withdrawal = swr * portfolio_value / 12
        if cap_amount is not None:
            monthly_withdrawal = min(monthly_withdrawal, cap_amount)
        if floor_amount is not None:
            monthly_withdrawal = max(monthly_withdrawal, floor_amount)
        current_cashflow = float(monthly_cashflows[i]) if i < len(monthly_cashflows) else 0.0
        if monthly_withdrawal < baseline_monthly_withdrawal:
            monthly_withdrawal = min(
                monthly_withdrawal + current_cashflow,
                baseline_monthly_withdrawal,
            )
        withdrawal = max(monthly_withdrawal - current_cashflow, 0.0)
        portfolio_value -= withdrawal
        yearly_withdrawals[-1] += monthly_withdrawal
        if i < len(subset) - 1:
            month_return = portfolio_returns[full_idx + 1] if full_idx + 1 < len(portfolio_returns) else 1.0
            portfolio_value *= month_return
        if portfolio_value <= 0:
            depleted = True

    fixed_yearly = float(settings.fixed_monthly_withdrawal) * 12
    success, ending_value = is_successful_ending_value_with_clamp(portfolio_value)
    return {
        "success": success,
        "ending_value": ending_value,
        "min_withdrawal": min(yearly_withdrawals),
        "max_withdrawal": max(yearly_withdrawals),
        "avg_withdrawal": sum(yearly_withdrawals) / len(yearly_withdrawals),
        "median_withdrawal": np.median(yearly_withdrawals),
        "pct_below_fixed": (len([x for x in yearly_withdrawals if x < fixed_yearly]) / len(yearly_withdrawals)),
    }
