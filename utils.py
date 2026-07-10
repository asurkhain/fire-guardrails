import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from joblib import dump, load

import numba as nb
import numpy as np
import pandas as pd

from app_settings import Settings
from cape_utils import calculate_cape_withdrawal_rate, simulate_cape_withdrawal_retirement
from vpw_utils import _apply_vpw_comparison_step, simulate_vpw_withdrawal_retirement
from shiller_utils import *  # noqa: F401,F403



@dataclass
class HistoricalWorkerContext:
    """Shared historical-analysis state prepared once per run."""

    analysis_df_path: str | None
    settings: Settings
    cashflows: list
    conditional_configs: list
    monthly_cashflows: np.ndarray
    all_stock_prices_path: str | None = None
    all_bond_prices_path: str | None = None
    portfolio_returns_path: str | None = None
    all_cape_values_path: str | None = None
    all_stock_prices: np.ndarray | None = None
    all_bond_prices: np.ndarray | None = None
    portfolio_returns: np.ndarray | None = None
    all_cape_values: np.ndarray | None = None
    analysis_df: pd.DataFrame | None = None


_HISTORICAL_WORKER_CONTEXT: HistoricalWorkerContext | None = None


def _set_historical_worker_context(context: HistoricalWorkerContext | None) -> None:
    global _HISTORICAL_WORKER_CONTEXT
    _HISTORICAL_WORKER_CONTEXT = context


def _materialize_historical_worker_context(context: HistoricalWorkerContext) -> HistoricalWorkerContext:
    if context.analysis_df_path is None:
        raise RuntimeError("Historical worker context is missing the analysis dataframe path.")
    analysis_df = pd.read_pickle(context.analysis_df_path)

    if context.all_stock_prices is None:
        if context.all_stock_prices_path is None:
            raise RuntimeError("Historical worker context is missing stock price arrays.")
        all_stock_prices = load(context.all_stock_prices_path, mmap_mode="r")
    else:
        all_stock_prices = context.all_stock_prices

    if context.all_bond_prices is None:
        if context.all_bond_prices_path is None:
            raise RuntimeError("Historical worker context is missing bond price arrays.")
        all_bond_prices = load(context.all_bond_prices_path, mmap_mode="r")
    else:
        all_bond_prices = context.all_bond_prices

    if context.portfolio_returns is None:
        if context.portfolio_returns_path is None:
            raise RuntimeError("Historical worker context is missing portfolio return arrays.")
        portfolio_returns = load(context.portfolio_returns_path, mmap_mode="r")
    else:
        portfolio_returns = context.portfolio_returns

    if context.all_cape_values is None:
        if context.all_cape_values_path is None:
            raise RuntimeError("Historical worker context is missing CAPE arrays.")
        all_cape_values = load(context.all_cape_values_path, mmap_mode="r")
    else:
        all_cape_values = context.all_cape_values

    return HistoricalWorkerContext(
        analysis_df_path=context.analysis_df_path,
        settings=context.settings,
        cashflows=context.cashflows,
        conditional_configs=context.conditional_configs,
        monthly_cashflows=context.monthly_cashflows,
        all_stock_prices_path=context.all_stock_prices_path,
        all_bond_prices_path=context.all_bond_prices_path,
        portfolio_returns_path=context.portfolio_returns_path,
        all_cape_values_path=context.all_cape_values_path,
        all_stock_prices=all_stock_prices,
        all_bond_prices=all_bond_prices,
        portfolio_returns=portfolio_returns,
        all_cape_values=all_cape_values,
        analysis_df=analysis_df,
    )

def _initialize_historical_worker_context(context: HistoricalWorkerContext) -> None:
    _set_historical_worker_context(_materialize_historical_worker_context(context))


def _build_historical_worker_context(
    df: pd.DataFrame,
    settings: Settings,
    analysis_df: pd.DataFrame | None = None,
) -> HistoricalWorkerContext:
    if analysis_df is None:
        analysis_df = extend_shiller_data_with_mode(
            df=df,
            extension_years=settings.historical_future_extension_years or 0,
            extension_mode=settings.shiller_extension_mode,
        )
    cashflows = settings.cashflows_for_calculation()
    conditional_configs = settings.conditional_cashflows_for_calculation()
    monthly_cashflows = cashflow_schedule_for_window(cashflows, settings.retirement_duration_months, 0)

    temp_dir = tempfile.mkdtemp(prefix="fire-guardrails-historical-")
    analysis_df_path = str(Path(temp_dir) / "analysis_df.pkl")
    all_stock_prices_path = str(Path(temp_dir) / "stock_prices.pkl")
    all_bond_prices_path = str(Path(temp_dir) / "bond_prices.pkl")
    portfolio_returns_path = str(Path(temp_dir) / "portfolio_returns.pkl")
    all_cape_values_path = str(Path(temp_dir) / "cape_values.pkl")

    all_stock_prices = analysis_df["Real Total Return Price"].to_numpy(dtype=float)
    all_bond_prices = analysis_df["Real Total Bond Returns"].to_numpy(dtype=float)
    portfolio_returns = compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)
    print(f"[DEBUG] Building context stock_pct={settings.stock_pct} portfolio_returns[:5]={portfolio_returns[:5].tolist()}")
    all_cape_values = analysis_df["CAPE"].to_numpy(dtype=float)
    analysis_df.to_pickle(analysis_df_path)
    dump(np.ascontiguousarray(all_stock_prices), all_stock_prices_path)
    dump(np.ascontiguousarray(all_bond_prices), all_bond_prices_path)
    dump(np.ascontiguousarray(portfolio_returns), portfolio_returns_path)
    dump(np.ascontiguousarray(all_cape_values), all_cape_values_path)
    return HistoricalWorkerContext(
        analysis_df_path=analysis_df_path,
        settings=settings,
        cashflows=cashflows,
        conditional_configs=conditional_configs,
        monthly_cashflows=monthly_cashflows,
        all_stock_prices_path=all_stock_prices_path,
        all_bond_prices_path=all_bond_prices_path,
        portfolio_returns_path=portfolio_returns_path,
        all_cape_values_path=all_cape_values_path,
        analysis_df=None,
    )

def cashflow_schedule_for_window(cashflows, window_length, start_offset=0):
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


def compute_portfolio_returns(stock_prices, bond_prices, stock_pct):
    """Compute blended portfolio returns from stock and bond price series."""
    stock_returns = np.ones(len(stock_prices))
    stock_returns[1:] = stock_prices[1:] / stock_prices[:-1]

    bond_returns = np.ones(len(bond_prices))
    bond_returns[1:] = bond_prices[1:] / bond_prices[:-1]

    return float(stock_pct) * stock_returns + (1 - float(stock_pct)) * bond_returns


def _cap_upper_guardrail_value(
    upper_guardrail_value: float,
    current_portfolio_value: float,
    initial_portfolio_value: float,
) -> float:
    """Limit the upper guardrail so it cannot run away far above the portfolio."""

    cap_value = max(float(initial_portfolio_value), 2.0 * float(current_portfolio_value))
    if not np.isfinite(upper_guardrail_value):
        return cap_value
    return min(float(upper_guardrail_value), cap_value)


def get_effective_final_value_target(
    final_value_target: float,
    total_retirement_months: int,
    months_remaining: int,
    ramp_down: bool = False,
) -> float:
    """Return the active final value target for the current point in retirement.

    When ramp_down is enabled, the target stays flat until the final 10% of the
    retirement horizon, then linearly declines to $1.00 by the end.
    """

    target = max(0.0, float(final_value_target))
    if target <= 0.0 or not ramp_down:
        return target

    total_months = float(total_retirement_months)
    if total_months <= 0.0:
        return target

    ramp_months = max(total_months * 0.10, 1.0)
    remaining = max(float(months_remaining), 0.0)
    if remaining >= ramp_months:
        return target
    return max(0.0, target * (remaining / ramp_months))


@nb.jit(nopython=True)
def test_all_periods(portfolio_returns, num_months, initial_value, monthly_spending, monthly_cashflows, final_value_target):
    """
    Run all paths of a given length that exist in the supplied portfolio_reurns array,
    capturing the success or failure of each path, and returning the proportion of successes.

    We're running this in a tight loop so we'll Numba-compile it for ultra-fast simulation.
    """
    num_periods = len(portfolio_returns) - num_months + 1
    successes = 0

    for start_idx in range(num_periods):
        value = initial_value

        for i in range(num_months):
            withdrawal = monthly_spending - monthly_cashflows[i]
            if withdrawal < 0.0:
                withdrawal = 0.0
            value = value * portfolio_returns[start_idx + i] - withdrawal
            if value <= 0:
                break

        # Success requires: didn't deplete AND final value >= target
        if value > 0 and value >= final_value_target:
            successes += 1

    return successes / num_periods


def calculate_success_rate(df,
                           withdrawal_rate,
                           num_months,
                           stock_pct=0.75,
                           analysis_start_date='1871-01-01',
                           initial_value=1_000_000,
                           monthly_cashflows=None,
                           final_value_target=0.0,
                           portfolio_returns=None):
    """
    Calculate the success rate for a given withdrawal rate.
    Numba-accelerated version. Should be 500x+ faster than brute force.
    First call will be slower due to compilation, subsequent calls will be blazing fast.
    That way we can use it iteratively in a guess-and-check loop.
    """
    if portfolio_returns is None:
        analysis_start = pd.to_datetime(analysis_start_date)
        df_filtered = df[df['Date'] >= analysis_start]

        stock_prices = df_filtered['Real Total Return Price'].values
        bond_prices = df_filtered['Real Total Bond Returns'].values
        portfolio_returns = compute_portfolio_returns(stock_prices, bond_prices, stock_pct)
    monthly_spending = initial_value * withdrawal_rate / 12

    if monthly_cashflows is None:
        monthly_cashflows = np.zeros(num_months, dtype=np.float64)
    else:
        monthly_cashflows = np.asarray(monthly_cashflows, dtype=np.float64)
        if len(monthly_cashflows) != num_months:
            raise ValueError("monthly_cashflows length must match num_months")

    # Call the compiled function
    return test_all_periods(portfolio_returns, num_months, initial_value, monthly_spending, monthly_cashflows, final_value_target)


def get_spending_rate_for_fixed_success_rate(df,
                                             desired_success_rate,
                                             num_months,
                                             analysis_start_date='1871-01-01',
                                             initial_value=1_000_000, stock_pct=0.75,
                                             tolerance=0.001, max_iterations=50,
                                             verbose=False,
                                             cashflows=None,
                                             cashflow_start_offset=0,
                                             cashflow_schedule=None,
                                             portfolio_returns=None,
                                             previous_spending_rate=None,
                                             final_value_target=0.0):
    """
    Compute the annual spending rate such that a historical simulation over periods of the desired length
    yields the desired success rate.

    Parameters
    ----------
    df : pd.DataFrame
        The main dataframe with market data
    desired_success_rate : float
        The target chance of not depleting the portfolio (e.g., 0.90 for 90%),
        the percent of simulation paths that should have ending portfolio values > 0.
    num_months : int
        The size of the time window of the historical simulation paths to run
        (corresponds to the remaining time in retirement).
    analysis_start_date : str, optional
        The start date from which we should begin running simulation paths,
        if we do not want to start at the very beginning.
    initial_value : float, optional
        Initial portfolio value (default 1,000,000)
    stock_pct : float, optional
        Percentage of portfolio in stocks (default 0.75)
    tolerance : float, optional
        Tolerance for success rate matching (default 0.001 = 0.1%)
    max_iterations : int, optional
        Maximum iterations for binary search (default 50)
    verbose : bool, optional
        Print progress (default False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'spending_rate': The annual spending rate that achieves the target success rate
        - 'actual_success_rate': The actual success rate achieved
        - 'num_simulations': Number of simulation paths run
        - 'iterations': Number of binary search iterations performed
    """

    # Edge case: we can't goal seek if the algo is one ended
    if desired_success_rate > 1.0 - tolerance:
        if verbose:
            print(f"Edge case: reducing desired success rate from {desired_success_rate} to effective algo "
                  f"ceiling of {1.0 - tolerance}")
        desired_success_rate = 1.0 - tolerance

    if desired_success_rate < tolerance:
        if verbose:
            print(f"Edge case: increasing desired success rate from {desired_success_rate} to effective algo "
                  f"floor of {tolerance}")
        desired_success_rate = tolerance

    if portfolio_returns is None:
        # Convert analysis_start_date to datetime
        analysis_start = pd.to_datetime(analysis_start_date)

        # Filter dataframe to only include dates from analysis_start_date onwards
        df_filtered = df[df['Date'] >= analysis_start].copy()

        # Get all possible starting dates for simulation periods
        max_end_idx = len(df_filtered) - num_months
        if max_end_idx <= 0:
            raise ValueError(f"Not enough data for {num_months} month simulations starting from {analysis_start_date}")

        num_paths = max_end_idx + 1
        portfolio_returns = compute_portfolio_returns(
            df_filtered['Real Total Return Price'].values,
            df_filtered['Real Total Bond Returns'].values,
            stock_pct,
        )
    else:
        portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
        num_paths = len(portfolio_returns) - num_months + 1
        if num_paths <= 0:
            raise ValueError(f"Not enough data for {num_months} month simulations")

    if cashflow_schedule is None:
        monthly_cashflows = cashflow_schedule_for_window(cashflows, num_months, cashflow_start_offset)
    else:
        monthly_cashflows = np.asarray(cashflow_schedule, dtype=np.float64)
        if len(monthly_cashflows) != num_months:
            raise ValueError("cashflow_schedule length must match num_months")

    def _success_rate_for_rate(rate: float) -> float:
        return calculate_success_rate(
            df,
            rate,
            num_months,
            stock_pct,
            analysis_start_date,
            initial_value,
            monthly_cashflows=monthly_cashflows,
            final_value_target=final_value_target,
            portfolio_returns=portfolio_returns,
        )

    # Binary search for the spending rate.
    # When a previous month's rate is available, start with a narrow bracket around it.
    low_rate = 0.0   # 0% annual spending rate
    high_rate = 1.0  # 100% annual spending rate

    if previous_spending_rate is not None and np.isfinite(previous_spending_rate):
        guess = float(np.clip(previous_spending_rate, 0.0, 1.0))
        guess_success_rate = _success_rate_for_rate(guess)
        if abs(guess_success_rate - desired_success_rate) <= tolerance:
            return {
                'spending_rate': guess,
                'actual_success_rate': guess_success_rate,
                'num_simulations': num_paths,
                'iterations': 1,
            }

        window = 0.05
        low_rate = max(0.0, guess - window)
        high_rate = min(1.0, guess + window)

        low_success_rate = _success_rate_for_rate(low_rate)
        high_success_rate = _success_rate_for_rate(high_rate)

        while not (low_success_rate >= desired_success_rate >= high_success_rate):
            if low_rate <= 0.0 and high_rate >= 1.0:
                low_rate = 0.0
                high_rate = 1.0
                break

            window = min(1.0, window * 2.0)
            low_rate = max(0.0, guess - window)
            high_rate = min(1.0, guess + window)
            low_success_rate = _success_rate_for_rate(low_rate)
            high_success_rate = _success_rate_for_rate(high_rate)

    iteration = 0
    best_rate = None
    best_success_rate = None

    if verbose:
        print(f"Searching for spending rate with {desired_success_rate:.1%} success rate...")

    while iteration < max_iterations:
        mid_rate = (low_rate + high_rate) / 2

        current_success_rate = _success_rate_for_rate(mid_rate)

        # Check if we're within tolerance
        if abs(current_success_rate - desired_success_rate) <= tolerance:
            best_rate = mid_rate
            best_success_rate = current_success_rate
            if verbose:
                print(f"Converged! Found spending rate within tolerance.")
            break

        # Adjust search bounds
        # If success rate is too high, we can spend more
        if current_success_rate > desired_success_rate:
            low_rate = mid_rate
            if verbose:
                print(f"Success rate at {mid_rate} is too high ({current_success_rate:.3f} > {desired_success_rate:.3f}), "
                      f"increasing spending rate range to [{low_rate:.4f}, {high_rate:.4f}]")
        else:
            # If success rate is too low, we need to spend less
            high_rate = mid_rate
            if verbose:
                print(f"Success rate at {mid_rate} is too low ({current_success_rate:.3f} < {desired_success_rate:.3f}), "
                      f"decreasing spending rate range to [{low_rate:.4f}, {high_rate:.4f}]")

        best_rate = mid_rate
        best_success_rate = current_success_rate
        iteration += 1

    if iteration >= max_iterations:
        if verbose:
            print(f"Reached maximum iterations ({max_iterations}). Returning best found rate.")

    # Return results
    return {
        'spending_rate': best_rate,
        'actual_success_rate': best_success_rate,
        'num_simulations': num_paths,
        'iterations': iteration + 1
    }


def is_adjustment_month(ts: pd.Timestamp, adjustment_frequency: str) -> bool:
    """Determine whether the guardrail policy permits an adjustment in the given month."""

    month = int(ts.month)
    if adjustment_frequency == "Monthly":
        return True
    if adjustment_frequency == "Quarterly":
        return ((month - 1) % 3) == 0
    if adjustment_frequency == "Biannually":
        return month in (1, 7)
    if adjustment_frequency == "Annually":
        return month == 1
    # Fallback to monthly behaviour for unexpected values
    return True


def evaluate_conditional_cashflows(
    spending_target: float,
    initial_spending: float,
    month_idx: int,
    conditional_configs: list,
    one_time_triggered: dict,
    recurring_state: dict,
) -> float:
    """
    Evaluate conditional cashflows and return total conditional cashflow amount.

    Parameters
    ----------
    spending_target : float
        The current month's spending level (after cap/floor bounds).
    initial_spending : float
        The initial monthly spending amount used as the baseline for thresholds.
    month_idx : int
        Zero-based index of the current month in the simulation.
    conditional_configs : list
        List of conditional cashflow configuration dicts with keys:
        - cashflow_type: "one_time" or "recurring"
        - trigger_threshold_multiplier: float (0.05 to 0.95)
        - amount: float
    one_time_triggered : dict
        State tracking for one-time cashflows: {idx: triggered_month or None}.
        Updated in place.
    recurring_state : dict
        State tracking for recurring cashflows: {idx: {"active": bool, "started_month": int or None}}.
        Updated in place.

    Returns
    -------
    float
        Total conditional cashflow amount for this month.
    """
    conditional_total = 0.0

    if initial_spending <= 0:
        return conditional_total

    spending_ratio = spending_target / initial_spending

    for idx, cfg in enumerate(conditional_configs):
        threshold = cfg["trigger_threshold_multiplier"]
        amount = cfg["amount"]

        if cfg["cashflow_type"] == "one_time":
            triggered_month = one_time_triggered.get(idx)
            if triggered_month is not None:
                # Already triggered - check if this is the month after trigger
                if month_idx == triggered_month + 1:
                    # One-time contribution in the month following trigger
                    conditional_total += amount
                # After that month, no more contribution
            else:
                # Not yet triggered - check condition
                if spending_ratio < threshold:
                    # Trigger! Mark as triggered, will apply NEXT month
                    one_time_triggered[idx] = month_idx

        else:  # recurring
            state = recurring_state.get(idx, {"active": False, "started_month": None})

            if state["active"]:
                # Currently active - check for recovery (100%+ of initial)
                if spending_ratio >= 1.0:
                    # Recovery - deactivate
                    state["active"] = False
                    state["started_month"] = None
                else:
                    # Still active - contribute
                    conditional_total += amount
            else:
                # Not active - check for activation
                if spending_ratio < threshold:
                    # Activate! Will apply starting NEXT month
                    state["active"] = True
                    state["started_month"] = month_idx
                    # Note: no contribution this month (starts next month)

            recurring_state[idx] = state

    return conditional_total


def get_guardrail_withdrawals(
    df,
    settings: Settings,
    verbose=False,
    on_progress=None,
    on_status=None,
    analysis_df=None,
    cashflows=None,
    conditional_configs=None,
    monthly_cashflows=None,
    all_stock_prices=None,
    all_bond_prices=None,
    portfolio_returns=None,
    all_cape_values=None,
):
    """Creates a dataframe of withdrawals that follow an adaptive guardrail strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical dataset containing the Shiller return series.
    settings : Settings
        Snapshot of the control state selected in the UI.
    verbose : bool, optional
        Emit periodic status messages to stdout, by default False.
    on_progress : callable, optional
        Callback invoked with (current, total) as the simulation progresses.
    on_status : callable, optional
        Callback invoked with textual status updates.

    Returns
    -------
    pandas.DataFrame
        Dataframe with guardrail withdrawal results and diagnostics.
    """
    start_date = pd.to_datetime(settings.start_date)
    end_date = settings.retirement_end_date()
    if end_date is None:
        raise ValueError("Unable to determine retirement end date from settings.")

    if analysis_df is None:
        analysis_df = extend_shiller_data_to_cover_date(
            df=df,
            end_date=end_date,
            minimum_extension_years=(
                int(settings.historical_future_extension_years or 0)
                if settings.mode == "Historical Mode"
                else 0
            ),
            extension_mode=settings.shiller_extension_mode,
        )

    # Filter to analysis period
    mask = (analysis_df["Date"] >= start_date) & (analysis_df["Date"] <= pd.to_datetime(end_date))
    subset = analysis_df[mask].copy()
    subset["_Full_Index"] = subset.index
    subset = subset.reset_index(drop=True)

    total_months = len(subset)
    if cashflows is None:
        cashflows = settings.cashflows_for_calculation()
    if conditional_configs is None:
        conditional_configs = settings.conditional_cashflows_for_calculation()
    if monthly_cashflows is None:
        monthly_cashflows = cashflow_schedule_for_window(cashflows, total_months, 0)
    else:
        monthly_cashflows = np.asarray(monthly_cashflows, dtype=np.float64)
        if len(monthly_cashflows) != total_months:
            raise ValueError("monthly_cashflows length must match total_months")

    # Pre-compute portfolio returns and CAPE values for the entire historical period (for speed)
    if all_stock_prices is None:
        all_stock_prices = analysis_df['Real Total Return Price'].values
    if all_bond_prices is None:
        all_bond_prices = analysis_df['Real Total Bond Returns'].values
    if portfolio_returns is None:
        portfolio_returns = compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)
    if all_cape_values is None:
        all_cape_values = analysis_df["CAPE"].values

    # Initialize results storage
    results = []

    current_portfolio_value = float(settings.initial_value)
    previous_total_spending = float(settings.initial_monthly_spending)
    cap_multiplier = settings.spending_cap_multiplier
    floor_multiplier = settings.spending_floor_multiplier
    cap_amount = (
        previous_total_spending * float(cap_multiplier)
        if cap_multiplier is not None
        else None
    )
    floor_amount = (
        previous_total_spending * float(floor_multiplier)
        if floor_multiplier is not None
        else None
    )

    # Fixed-withdrawal shadow path initialization
    if settings.fixed_monthly_withdrawal is not None:
        fixed_total_spending = float(settings.fixed_monthly_withdrawal)
        fixed_portfolio_value = float(settings.initial_value)
    else:
        fixed_total_spending = previous_total_spending
        fixed_portfolio_value = float(settings.initial_value)

    # CAPE path initialization
    cape_portfolio_value = float(settings.initial_value)
    cape_depleted = False

    # VPW path initialization
    vpw_portfolio_value = float(settings.initial_value)
    vpw_depleted = False

    guardrail_depleted = False
    fixed_depleted = False

    # Conditional cashflow state tracking
    initial_spending_for_conditionals = float(settings.initial_monthly_spending)
    one_time_triggered = {}  # {idx: triggered_month or None}
    recurring_state = {}     # {idx: {"active": bool, "started_month": int or None}}
    for idx, cfg in enumerate(conditional_configs):
        if cfg["cashflow_type"] == "one_time":
            one_time_triggered[idx] = None
        else:
            recurring_state[idx] = {"active": False, "started_month": None}

    current_dates = subset["Date"].tolist()
    full_indices = subset["_Full_Index"].to_numpy(dtype=np.int64, copy=False)
    months = subset["Date"].dt.month.to_numpy(dtype=np.int16, copy=False)
    previous_spending_rate_by_target: dict[float, float] = {}

    def _cached_get_sr(success_rate: float) -> float:
        cache_key = round(float(success_rate), 12)
        previous_rate = previous_spending_rate_by_target.get(cache_key)
        effective_final_value_target = get_effective_final_value_target(
            final_value_target=settings.final_value_target,
            total_retirement_months=settings.retirement_duration_months,
            months_remaining=months_remaining,
            ramp_down=settings.final_value_target_ramp_down,
        )
        result = get_spending_rate_for_fixed_success_rate(
            df=analysis_df,
            desired_success_rate=success_rate,
            num_months=months_remaining,
            initial_value=current_portfolio_value,
            stock_pct=settings.stock_pct,
            cashflows=cashflows,
            cashflow_schedule=schedule_slice,
            portfolio_returns=portfolio_returns,
            previous_spending_rate=previous_rate,
            final_value_target=effective_final_value_target,
        )
        spending_rate = result["spending_rate"]
        if spending_rate is not None:
            previous_spending_rate_by_target[cache_key] = float(spending_rate)
        return spending_rate

    for i in range(total_months):
        last_month = i >= (total_months - 1)
        current_date = current_dates[i]
        current_month = int(months[i])
        months_remaining = len(subset) - i
        is_first_month = i == 0
        if is_first_month:
            adjustment_allowed = False
        elif settings.adjustment_frequency == "Monthly":
            adjustment_allowed = True
        elif settings.adjustment_frequency == "Quarterly":
            adjustment_allowed = ((current_month - 1) % 3) == 0
        elif settings.adjustment_frequency == "Biannually":
            adjustment_allowed = current_month in (1, 7)
        elif settings.adjustment_frequency == "Annually":
            adjustment_allowed = current_month == 1
        else:
            adjustment_allowed = True

        status_line = f"Processing {current_date.strftime('%Y-%m')}, portfolio=${current_portfolio_value:,.0f}, months_remaining={months_remaining}"
        if on_status is not None:
            on_status(status_line)
        if on_progress is not None:
            on_progress(i + 1, total_months)

        current_cashflow = float(monthly_cashflows[i]) if i < len(monthly_cashflows) else 0.0
        schedule_slice = monthly_cashflows[i:i + months_remaining]

        if not guardrail_depleted:

            # Calculate the upper and lower guardrail spending rates, based on the current spending and months remaining
            #
            upper_sr = _cached_get_sr(settings.upper_guardrail_success)
            lower_sr = _cached_get_sr(settings.lower_guardrail_success)

            # The guardrail portfolio values are the values that would result in the current spending amount
            # representing a probability of success equal to each guardrail's probability.
            #
            upper_guardrail_value = previous_total_spending / upper_sr * 12 if upper_sr > 0 else np.inf
            upper_guardrail_value = _cap_upper_guardrail_value(
                upper_guardrail_value=upper_guardrail_value,
                current_portfolio_value=current_portfolio_value,
                initial_portfolio_value=float(settings.initial_value),
            )
            lower_guardrail_value = previous_total_spending / lower_sr * 12 if lower_sr > 0 else np.inf

            if verbose and i % 12 == 0:
                print(f"Processing {current_date.strftime('%Y-%m')}, portfolio=${current_portfolio_value:,.0f}, "
                      f"months_remaining={months_remaining}")

            if adjustment_allowed:
                # Step 3: Check if we hit guardrails
                hit_upper = current_portfolio_value >= upper_guardrail_value
                hit_lower = current_portfolio_value <= lower_guardrail_value

                # Step 4: Calculate proposed spending adjustment
                if hit_upper:
                    desired_success_rate = settings.upper_guardrail_success + settings.upper_adjustment_fraction * (
                        settings.target_success_rate - settings.upper_guardrail_success
                    )
                    new_sr = _cached_get_sr(desired_success_rate)
                    new_proposed_spending = current_portfolio_value * new_sr / 12
                    guardrail_hit = "UPPER"

                elif hit_lower:
                    desired_success_rate = settings.lower_guardrail_success + settings.lower_adjustment_fraction * (
                        settings.target_success_rate - settings.lower_guardrail_success
                    )
                    new_sr = _cached_get_sr(desired_success_rate)
                    new_proposed_spending = current_portfolio_value * new_sr / 12
                    guardrail_hit = "LOWER"

                else:
                    new_proposed_spending = previous_total_spending
                    guardrail_hit = "NONE"

                # Step 6: Only adjust if the new dollar amount differs from the previous dollar amount by more than the configured threshold.
                #
                bounded_spending = new_proposed_spending
                if cap_amount is not None:
                    bounded_spending = min(bounded_spending, cap_amount)
                if floor_amount is not None:
                    bounded_spending = max(bounded_spending, floor_amount)

                percent_change = (
                    abs(bounded_spending - previous_total_spending) / previous_total_spending
                    if previous_total_spending else 0.0
                )

                if percent_change > settings.adjustment_threshold:
                    # Make the adjustment
                    if len(results) == 0:
                        print(f"Setting spending_target to bounded_spending value of {bounded_spending}")
                    spending_target = bounded_spending
                    adjustment_made = True
                else:
                    # Keep previous spending (inflation adjusted)
                    spending_target = previous_total_spending
                    adjustment_made = False
            else:
                guardrail_hit = "SKIPPED"
                percent_change = 0.0
                spending_target = previous_total_spending
                adjustment_made = False
        else:
            upper_guardrail_value = 0.0
            lower_guardrail_value = 0.0
            spending_target = 0.0
            adjustment_made = False
            guardrail_hit = "DEPLETED"
            percent_change = 0.0
            adjustment_allowed = False

        # Evaluate conditional cashflows (modifies state dicts in place)
        conditional_cashflow = evaluate_conditional_cashflows(
            spending_target=spending_target,
            initial_spending=initial_spending_for_conditionals,
            month_idx=i,
            conditional_configs=conditional_configs,
            one_time_triggered=one_time_triggered,
            recurring_state=recurring_state,
        )

        # Combine regular and conditional cashflows
        total_cashflow = current_cashflow + conditional_cashflow
        # recalculate spending target to be at least the total cashflow
        spending_target = max(spending_target, total_cashflow)

        # Withdrawal for this month
        withdrawal_amount = 0.0 if guardrail_depleted else max(spending_target - total_cashflow, 0.0)
        # Fixed path withdrawal for this month
        fixed_actual_withdrawal = 0.0 if fixed_depleted else max(fixed_total_spending - total_cashflow, 0.0)

        # CAPE path
        if not cape_depleted:
            full_idx = int(full_indices[i])
            cape_idx = min(full_idx + 1, len(all_cape_values) - 1)
            swr = calculate_cape_withdrawal_rate(all_cape_values[cape_idx])
            cape_monthly_spending = swr * cape_portfolio_value / 12
            if cap_amount is not None:
                cape_monthly_spending = min(cape_monthly_spending, cap_amount)
            if floor_amount is not None:
                cape_monthly_spending = max(cape_monthly_spending, floor_amount)
            if cape_monthly_spending < float(settings.initial_monthly_spending):
                cape_monthly_spending = min(
                    cape_monthly_spending + current_cashflow,
                    float(settings.initial_monthly_spending),
                )
            cape_actual_withdrawal = max(cape_monthly_spending - total_cashflow, 0.0)
        else:
            cape_monthly_spending = 0.0
            cape_actual_withdrawal = 0.0

        # VPW path
        if not vpw_depleted:
            vpw_monthly_spending, vpw_actual_withdrawal, vpw_payment_rate = _apply_vpw_comparison_step(
                portfolio_value=vpw_portfolio_value,
                months_remaining=months_remaining,
                stock_pct=settings.stock_pct,
                current_cashflow=total_cashflow,
                cap_amount=cap_amount,
                floor_amount=floor_amount,
            )
        else:
            vpw_monthly_spending = 0.0
            vpw_actual_withdrawal = 0.0

        # Update portfolio values - Apply withdrawals taken this month
        current_portfolio_value -= withdrawal_amount
        fixed_portfolio_value -= fixed_actual_withdrawal
        cape_portfolio_value -= cape_actual_withdrawal
        vpw_portfolio_value -= vpw_actual_withdrawal

        # Floor at zero and mark depletion so future months remain at zero
        if current_portfolio_value <= 0:
            current_portfolio_value = 0.0
            guardrail_depleted = True
        if fixed_portfolio_value <= 0:
            fixed_portfolio_value = 0.0
            fixed_depleted = True
        if cape_portfolio_value <= 0:
            cape_portfolio_value = 0.0
            # CAPE can drive the portfolio slightly below 0 in the last month, so ignore that
            if not last_month:
                cape_depleted = True
        if vpw_portfolio_value <= 0:
            vpw_portfolio_value = 0.0
            # VPW can drive the portfolio slightly below 0 in the last month, so ignore that
            if not last_month:
                vpw_depleted = True

        # Store results
        results.append({
            'Date': current_date,
            'Withdrawal': withdrawal_amount,
            'Total_Spending': spending_target,
            'Net_Cashflow': total_cashflow,
            'Conditional_Cashflow': conditional_cashflow,
            'Portfolio_Value': current_portfolio_value,
            'Fixed_SR_Value': fixed_portfolio_value,
            'Fixed_SR_Withdrawal': fixed_actual_withdrawal,
            'Fixed_SR_Total_Spending': fixed_total_spending,
            'CAPE_Value': cape_portfolio_value,
            'CAPE_Withdrawal': cape_actual_withdrawal,
            'CAPE_Total_Spending': cape_monthly_spending,
            'VPW_Value': vpw_portfolio_value,
            'VPW_Withdrawal': vpw_actual_withdrawal,
            'VPW_Total_Spending': vpw_monthly_spending,
            'Guardrail_Success': not guardrail_depleted,
            'Fixed_Success': not fixed_depleted,
            'CAPE_Success': not cape_depleted,
            'VPW_Success': not vpw_depleted,
            'Upper_Guardrail': upper_guardrail_value,
            'Lower_Guardrail': lower_guardrail_value,
            'Guardrail_Hit': guardrail_hit,
            'Percent_Change': percent_change,
            'Adjustment_Made': adjustment_made,
            'Adjustment_Allowed': adjustment_allowed
        })

        # Apply market returns (if not at end)
        if i < len(subset) - 1:
            # Find the index in the full dataset
            full_idx = int(full_indices[i])
            month_return = portfolio_returns[full_idx + 1] if full_idx + 1 < len(portfolio_returns) else 1.0
            current_portfolio_value *= month_return
            fixed_portfolio_value *= month_return
            cape_portfolio_value *= month_return
            vpw_portfolio_value *= month_return



        # Update state
        previous_total_spending = spending_target

    return pd.DataFrame(results)


def enumerate_historical_retirement_starts(
    df: pd.DataFrame,
    analysis_start_date,
    duration_months: int,
) -> list:
    """Return retirement start dates (January 1 of each year) with full data coverage."""
    analysis_start = pd.to_datetime(analysis_start_date)
    max_date = pd.to_datetime(df["Date"].max())
    duration_months = int(duration_months)

    if duration_months <= 0:
        return []

    first_year = analysis_start.year
    starts = []
    for year in range(first_year, max_date.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        if start < analysis_start:
            continue
        end = start + pd.DateOffset(months=duration_months - 1)
        if end > max_date:
            break
        mask = (df["Date"] >= start) & (df["Date"] <= end)
        if mask.sum() >= duration_months:
            starts.append(start.date())
    return starts


def _spending_stats_from_results(results_df: pd.DataFrame) -> dict:
    """Compute monthly spending statistics from a guardrail simulation dataframe."""
    if results_df is None or results_df.empty:
        return {
            "min_spending": np.nan,
            "max_spending": np.nan,
            "median_spending": np.nan,
            "avg_spending": np.nan,
        }

    if "Total_Spending" in results_df.columns:
        spending = results_df["Total_Spending"].astype(float)
    else:
        spending = results_df["Withdrawal"].astype(float)
        if "Net_Cashflow" in results_df.columns:
            spending = spending + results_df["Net_Cashflow"].astype(float)

    return {
        "min_spending": float(spending.min()),
        "max_spending": float(spending.max()),
        "median_spending": float(spending.median()),
        "avg_spending": float(spending.mean()),
    }

def simulate_fixed_withdrawal_retirement(
    df: pd.DataFrame,
    settings: Settings,
    analysis_df=None,
    monthly_cashflows=None,
    all_stock_prices=None,
    all_bond_prices=None,
    portfolio_returns=None,
) -> dict:
    """Run a fixed monthly withdrawal path for one retirement window."""
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

    fixed_monthly = float(settings.fixed_monthly_withdrawal)
    if monthly_cashflows is None:
        cashflows = settings.cashflows_for_calculation()
        monthly_cashflows = cashflow_schedule_for_window(cashflows, total_months, 0)
    else:
        monthly_cashflows = np.asarray(monthly_cashflows, dtype=np.float64)
    if all_stock_prices is None:
        all_stock_prices = source_df["Real Total Return Price"].values
    if all_bond_prices is None:
        all_bond_prices = source_df["Real Total Bond Returns"].values
    if portfolio_returns is None:
        portfolio_returns = compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)

    portfolio_value = float(settings.initial_value)
    depleted = False

    full_indices = subset["_Full_Index"].to_numpy(dtype=np.int64, copy=False)

    for i in range(total_months):
        if depleted:
            break
        current_cashflow = float(monthly_cashflows[i]) if i < len(monthly_cashflows) else 0.0
        withdrawal = max(fixed_monthly - current_cashflow, 0.0)
        portfolio_value -= withdrawal
        if i < len(subset) - 1:
            full_idx = int(full_indices[i])
            month_return = portfolio_returns[full_idx + 1] if full_idx + 1 < len(portfolio_returns) else 1.0
            portfolio_value *= month_return
        if portfolio_value <= 0:
            portfolio_value = 0.0
            depleted = True

    ending_value = max(portfolio_value, 0.0)
    success = ending_value > 0
    return {"success": not depleted, "ending_value": ending_value}


def _run_single_retirement_task(start, context_spec: HistoricalWorkerContext | None = None) -> dict:
    """Run a single historical retirement simulation."""
    context = _HISTORICAL_WORKER_CONTEXT
    if context_spec is not None and (
        context is None or context.portfolio_returns_path != context_spec.portfolio_returns_path
    ):
        context = _materialize_historical_worker_context(context_spec)
        _set_historical_worker_context(context)
    elif context is None:
        raise RuntimeError("Historical worker context has not been initialized.")

    df = context.analysis_df
    settings = context.settings
    period_settings = settings.with_start_date(start)
    fixed_withdrawal = settings.fixed_monthly_withdrawal
    results_df = get_guardrail_withdrawals(
        df=df,
        settings=period_settings,
        verbose=False,
        analysis_df=df,
        cashflows=context.cashflows,
        conditional_configs=context.conditional_configs,
        monthly_cashflows=context.monthly_cashflows,
        all_stock_prices=context.all_stock_prices,
        all_bond_prices=context.all_bond_prices,
        portfolio_returns=context.portfolio_returns,
        all_cape_values=context.all_cape_values,
    )
    spending_stats = _spending_stats_from_results(results_df)

    ending_value = max(results_df["Portfolio_Value"].iloc[-1], 0.0)
    success = results_df["Guardrail_Success"].iloc[-1]
    end_date = period_settings.retirement_end_date()

    initial_monthly = float(period_settings.initial_monthly_spending)
    fixed_monthly = (
        float(period_settings.fixed_monthly_withdrawal)
        if period_settings.fixed_monthly_withdrawal is not None
        else 0.0
    )
    withdraw_compare = fixed_monthly if fixed_monthly > 0.0 else initial_monthly
    pct_below_initial = float((results_df["Total_Spending"] < withdraw_compare).mean()) if not results_df.empty else np.nan

    row = {
        "Start_Year": start.year,
        "Start_Date": start,
        "End_Date": end_date,
        "Guardrail_Success": success,
        "Starting_Portfolio": float(settings.initial_value),
        "Ending_Portfolio": ending_value,
        "pct_below_initial": pct_below_initial,
        **spending_stats,
    }

    if fixed_withdrawal is not None:
        fixed_settings = period_settings
        fixed_result = simulate_fixed_withdrawal_retirement(
            df=df,
            settings=fixed_settings,
            analysis_df=df,
            monthly_cashflows=context.monthly_cashflows,
            all_stock_prices=context.all_stock_prices,
            all_bond_prices=context.all_bond_prices,
            portfolio_returns=context.portfolio_returns,
        )
        row["Fixed_Success"] = fixed_result["success"]
        row["Fixed_Ending_Portfolio"] = fixed_result["ending_value"]

        cape_settings = period_settings
        cape_result = simulate_cape_withdrawal_retirement(
            df=df,
            settings=cape_settings,
            analysis_df=df,
            monthly_cashflows=context.monthly_cashflows,
            all_cape_values=context.all_cape_values,
            all_stock_prices=context.all_stock_prices,
            all_bond_prices=context.all_bond_prices,
            portfolio_returns=context.portfolio_returns,
        )
        row["CAPE_Success"] = cape_result["success"]
        row["CAPE_Ending_Portfolio"] = cape_result["ending_value"]
        row["CAPE_Minimum_Withdrawal"] = cape_result["min_withdrawal"]
        row["CAPE_Maximum_Withdrawal"] = cape_result["max_withdrawal"]
        row["CAPE_Average_Withdrawal"] = cape_result["avg_withdrawal"]
        row["CAPE_Median_Withdrawal"] = cape_result["median_withdrawal"]
        row["CAPE_Pct_Below_Fixed"] = cape_result["pct_below_fixed"]

        vpw_result = simulate_vpw_withdrawal_retirement(
            df=df,
            settings=period_settings,
            analysis_df=df,
            monthly_cashflows=context.monthly_cashflows,
            all_stock_prices=context.all_stock_prices,
            all_bond_prices=context.all_bond_prices,
            portfolio_returns=context.portfolio_returns,
        )
        row["VPW_Success"] = vpw_result["success"]
        row["VPW_Ending_Portfolio"] = vpw_result["ending_value"]
        row["VPW_Minimum_Withdrawal"] = vpw_result["min_withdrawal"]
        row["VPW_Maximum_Withdrawal"] = vpw_result["max_withdrawal"]
        row["VPW_Average_Withdrawal"] = vpw_result["avg_withdrawal"]
        row["VPW_Median_Withdrawal"] = vpw_result["median_withdrawal"]
        row["VPW_Pct_Below_Fixed"] = vpw_result["pct_below_fixed"]
    else:
        row["Fixed_Success"] = None
        row["Fixed_Ending_Portfolio"] = None
        row["CAPE_Success"] = None
        row["CAPE_Ending_Portfolio"] = None
        row["CAPE_Minimum_Withdrawal"] = None
        row["CAPE_Maximum_Withdrawal"] = None
        row["CAPE_Average_Withdrawal"] = None
        row["CAPE_Median_Withdrawal"] = None
        row["CAPE_Pct_Below_Fixed"] = None
        row["VPW_Success"] = None
        row["VPW_Ending_Portfolio"] = None
        row["VPW_Minimum_Withdrawal"] = None
        row["VPW_Maximum_Withdrawal"] = None
        row["VPW_Average_Withdrawal"] = None
        row["VPW_Median_Withdrawal"] = None
        row["VPW_Pct_Below_Fixed"] = None

    return row


NUM_CORES = 8  # Code configurable number of CPU cores. Set to 1 to run sequentially.

def run_combo_analysis(
        df: pd.DataFrame,
        settings: Settings,
        stock_pct_range: tuple[float, float],
        upper_guardrail_success_range: tuple[float, float],
        lower_guardrail_success_range: tuple[float, float],
        target_success_rate_range: tuple[float, float],
        on_progress=None,
        on_status=None,
) -> dict[str, pd.DataFrame]:

    step = 0.05
    stock_pct_min, stock_pct_max = stock_pct_range
    ugr_min, ugr_max = upper_guardrail_success_range
    lgr_min, lgr_max = lower_guardrail_success_range
    tsr_min, tsr_max = target_success_rate_range

    # Generate independent value lists for each parameter
    stock_pct_values = np.round(np.arange(stock_pct_min, stock_pct_max + step, step), 2).tolist()
    ugr_values = np.round(np.arange(ugr_min, ugr_max + step, step), 2).tolist()
    lgr_values = np.round(np.arange(lgr_min, lgr_max + step, step), 2).tolist()
    tsr_values = np.round(np.arange(tsr_min, tsr_max + step, step), 2).tolist()

    # Filter to range bounds (arange can overshoot due to float rounding)
    stock_pct_values = [v for v in stock_pct_values if stock_pct_min <= v <= stock_pct_max]
    ugr_values = [v for v in ugr_values if ugr_min <= v <= ugr_max]
    lgr_values = [v for v in lgr_values if lgr_min <= v <= lgr_max]
    tsr_values = [v for v in tsr_values if tsr_min <= v <= tsr_max]

    # Build all valid combinations: lgr < tsr < ugr
    combos = []
    for sp in stock_pct_values:
        for ugr in ugr_values:
            for lgr in lgr_values:
                for tsr in tsr_values:
                    if not (lgr < tsr < ugr):
                        continue
                    combos.append((sp, ugr, lgr, tsr))

    total = len(combos)
    results: dict[str, pd.DataFrame] = {}
    idx = 0

    for sp, ugr, lgr, tsr in combos:
        combo_settings = Settings(
            mode=settings.mode,
            start_date=settings.start_date,
            retirement_duration_months=settings.retirement_duration_months,
            analysis_start_date=settings.analysis_start_date,
            initial_value=settings.initial_value,
            stock_pct=sp,
            target_success_rate=tsr,
            initial_monthly_spending=settings.initial_monthly_spending,
            initial_spending_overridden=settings.initial_spending_overridden,
            upper_guardrail_success=ugr,
            lower_guardrail_success=lgr,
            upper_adjustment_fraction=settings.upper_adjustment_fraction,
            lower_adjustment_fraction=settings.lower_adjustment_fraction,
            adjustment_threshold=settings.adjustment_threshold,
            adjustment_frequency=settings.adjustment_frequency,
            spending_cap_option=settings.spending_cap_option,
            spending_floor_option=settings.spending_floor_option,
            shiller_extension_mode=settings.shiller_extension_mode,
            final_value_target=settings.final_value_target,
            final_value_target_ramp_down=settings.final_value_target_ramp_down,
            fixed_monthly_withdrawal=settings.fixed_monthly_withdrawal,
            historical_future_extension_years=settings.historical_future_extension_years,
            cashflows=list(settings.cashflows),
            conditional_cashflows=list(settings.conditional_cashflows),
        )

        key = f"sp={sp:.0%}_ugr={ugr:.0%}_lgr={lgr:.0%}_tsr={tsr:.0%}"
        print(f"[DEBUG] Combo iteration stock_pct={sp}")

        def _combo_progress(current, total_inner):
            if on_progress is not None and total > 0:
                combo_fraction = current / total_inner if total_inner > 0 else 0
                on_progress(idx + combo_fraction, total)

        def _combo_status(msg):
            if on_status is not None:
                on_status(f"[{key}] {msg}")

        result_df = run_historical_retirements_analysis(
            df=df,
            settings=combo_settings,
            on_progress=_combo_progress,
            on_status=_combo_status,
        )
        results[key] = result_df
        idx += 1

    return results

def run_historical_retirements_analysis(
    df: pd.DataFrame,
    settings: Settings,
    on_progress=None,
    on_status=None,
) -> pd.DataFrame:
    """Run guardrail simulations for every historical retirement start year."""
    analysis_df = extend_shiller_data_with_mode(
        df=df,
        extension_years=settings.historical_future_extension_years or 0,
        extension_mode=settings.shiller_extension_mode,
    )
    start_dates = enumerate_historical_retirement_starts(
        df=analysis_df,
        analysis_start_date=settings.analysis_start_date,
        duration_months=settings.retirement_duration_months,
    )
    if not start_dates:
        raise ValueError(
            "No historical retirement periods fit the selected duration and analysis start date."
        )

    total = len(start_dates)
    rows = []
    context_spec = _build_historical_worker_context(df, settings, analysis_df=analysis_df)
    temp_dir = Path(context_spec.all_stock_prices_path).parent if context_spec.all_stock_prices_path else None

    try:
        if NUM_CORES > 1:
            from joblib import Parallel, delayed

            parallel = Parallel(
                n_jobs=NUM_CORES,
                backend="loky",
                return_as="generator_unordered",
            )
            try:
                results = parallel(
                    delayed(_run_single_retirement_task)(start, context_spec)
                    for start in start_dates
                )

                for idx, row in enumerate(results):
                    rows.append(row)
                    start_year = row["Start_Year"]

                    if on_status is not None:
                        on_status(f"Completed retirement starting in {start_year} ({idx + 1}/{total})")
                    if on_progress is not None:
                        on_progress(idx + 1, total)
            finally:
                _set_historical_worker_context(None)
            del parallel

            # Sort rows by Start_Year to maintain chronological order in charts and tables
            rows.sort(key=lambda r: r["Start_Year"])
        else:
            # Sequential fallback
            context = _materialize_historical_worker_context(context_spec)
            _set_historical_worker_context(context)
            try:
                for idx, start in enumerate(start_dates):
                    if on_status is not None:
                        on_status(f"Retirement {start.year} ({idx + 1}/{total})")
                    if on_progress is not None:
                        on_progress(idx + 1, total)
                    row = _run_single_retirement_task(start)
                    rows.append(row)
            finally:
                _set_historical_worker_context(None)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return pd.DataFrame(rows)


def get_combo_tempdir() -> str:
    """Return the temp directory used for combo result files, creating it if needed."""
    d = Path(tempfile.gettempdir()) / "fire-guardrails-combo"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def save_combo_results(results: dict[str, pd.DataFrame], path: str) -> None:
    """Save combo results dict {key -> DataFrame} to a file via joblib."""
    dump(results, path)


def load_combo_results(path: str) -> dict[str, pd.DataFrame]:
    """Load combo results dict {key -> DataFrame} previously saved with save_combo_results."""
    return load(path)


def compute_guardrail_guidance_snapshot(
    df,
    asof_date,
    settings: Settings,
) -> dict:
    """Compute a single-point guidance snapshot (no historical loop).

    Parameters
    ----------
    df : pandas.DataFrame
        Historical dataset containing the Shiller return series.
    asof_date : datetime-like
        Date on which guidance is generated.
    settings : Settings
        Snapshot of the control state selected in the UI.

    Notes
    -----
    This mirrors the first-iteration logic in :func:`get_guardrail_withdrawals`:
      - Determine months_remaining based on [asof_date .. end_date]
      - Compute spending rates at target, upper, lower success rates
      - Guardrail portfolio values are the PVs at which CURRENT spending would equal those spending rates
      - Hypothetical adjustments on guardrail hit are computed by moving partway back toward the target
        using the upper/lower adjustment fractions. Threshold gating is intentionally ignored.

    Adjustment frequency, thresholds, and spending bounds are not considered here.
    """
    asof = pd.to_datetime(asof_date)

    # Determine months remaining directly from configured duration
    num_months = int(settings.retirement_duration_months)
    if num_months <= 0:
        # Fallback to a standard 30-year horizon
        num_months = 360

    cashflows = settings.cashflows_for_calculation()
    monthly_cashflows = cashflow_schedule_for_window(cashflows, num_months, 0)
    current_cashflow_amount = float(monthly_cashflows[0]) if len(monthly_cashflows) > 0 else 0.0
    stock_prices = df["Real Total Return Price"].values
    bond_prices = df["Real Total Bond Returns"].values
    portfolio_returns = compute_portfolio_returns(stock_prices, bond_prices, settings.stock_pct)
    effective_final_value_target = get_effective_final_value_target(
        final_value_target=settings.final_value_target,
        total_retirement_months=settings.retirement_duration_months,
        months_remaining=num_months,
        ramp_down=settings.final_value_target_ramp_down,
    )

    # Totals over the coming year for cashflows and withdrawals net of cashflows
    first_year_months = min(12, len(monthly_cashflows))
    annual_cashflow_total = float(np.sum(monthly_cashflows[:first_year_months])) if first_year_months else 0.0

    # Helper to compute spending rate for a given success rate at 'asof'
    def _sr(desired_success_rate: float):
        res = get_spending_rate_for_fixed_success_rate(
            df=df,
            desired_success_rate=float(desired_success_rate),
            num_months=num_months,
            #analysis_start_date=settings.analysis_start_date,
            initial_value=float(settings.initial_value),
            stock_pct=float(settings.stock_pct),
            cashflows=cashflows,
            cashflow_schedule=monthly_cashflows,
            portfolio_returns=portfolio_returns,
            final_value_target=effective_final_value_target,
        )
        return float(res["spending_rate"]) if res["spending_rate"] is not None else None

    target_sr = _sr(settings.target_success_rate)
    upper_sr = _sr(settings.upper_guardrail_success)
    lower_sr = _sr(settings.lower_guardrail_success)

    # Guardrail PVs where CURRENT spending equals the spending rate for upper/lower success rates (matches simulation logic)
    use_spending = (
        float(settings.initial_monthly_spending)
        if settings.initial_monthly_spending is not None
        else None
    )

    if upper_sr is not None and upper_sr > 0 and use_spending is not None:
        upper_guardrail_value = use_spending / upper_sr * 12.0
    else:
        upper_guardrail_value = np.inf
    upper_guardrail_value = _cap_upper_guardrail_value(
        upper_guardrail_value=upper_guardrail_value,
        current_portfolio_value=float(settings.initial_value),
        initial_portfolio_value=float(settings.initial_value),
    )

    if lower_sr is not None and lower_sr > 0 and use_spending is not None:
        lower_guardrail_value = use_spending / lower_sr * 12.0
    else:
        lower_guardrail_value = np.inf

    # Hypothetical adjustment targets if guardrails were hit (move back toward target)
    desired_upper_success_rate = settings.upper_guardrail_success + settings.upper_adjustment_fraction * (
        settings.target_success_rate - settings.upper_guardrail_success
    )
    desired_lower_success_rate = settings.lower_guardrail_success + settings.lower_adjustment_fraction * (
        settings.target_success_rate - settings.lower_guardrail_success
    )

    adj_upper_sr = _sr(desired_upper_success_rate)
    adj_lower_sr = _sr(desired_lower_success_rate)

    # At a guardrail hit, the simulation computes the new spending using the portfolio value at the hit.
    # Mirror that here by using the guardrail PVs rather than today's PV for the hypothetical adjustments
    # and apply the same cap/floor bounds that the simulation enforces.
    adj_upper_monthly = (
        float(upper_guardrail_value) * adj_upper_sr / 12.0
        if (adj_upper_sr is not None and np.isfinite(upper_guardrail_value))
        else None
    )
    adj_lower_monthly = (
        float(lower_guardrail_value) * adj_lower_sr / 12.0
        if (adj_lower_sr is not None and np.isfinite(lower_guardrail_value))
        else None
    )

    # Percent change relative to CURRENT spending
    def _pct_change(new_amt):
        if new_amt is None or float(settings.initial_monthly_spending) == 0.0:
            return None
        return (
            float(new_amt) - float(settings.initial_monthly_spending)
        ) / float(settings.initial_monthly_spending)

    # Implied spending rate from current spending and portfolio value
    implied_sr = (
        (float(settings.initial_monthly_spending) * 12.0 / float(settings.initial_value))
        if float(settings.initial_value) > 0
        else None
    )
    target_monthly_spending = (
        float(settings.initial_value) * target_sr / 12.0
        if target_sr is not None
        else None
    )
    target_monthly_withdrawal = None
    target_annual_withdrawal = None
    if target_monthly_spending is not None:
        target_monthly_withdrawal = max(target_monthly_spending - current_cashflow_amount, 0.0)
        if first_year_months:
            annual_withdrawals = [
                max(float(target_monthly_spending) - float(cf), 0.0)
                for cf in monthly_cashflows[:first_year_months]
            ]
            target_annual_withdrawal = float(np.sum(annual_withdrawals))

    # Calculate actual withdrawal rates (spending rate minus effect of cashflows)
    # Withdrawal rate = (monthly withdrawal * 12) / portfolio value
    implied_withdrawal_rate = None
    target_withdrawal_rate = None
    if float(settings.initial_value) > 0:
        current_withdrawal = max(float(settings.initial_monthly_spending) - current_cashflow_amount, 0.0)
        implied_withdrawal_rate = (current_withdrawal * 12.0) / float(settings.initial_value)
        if target_monthly_withdrawal is not None:
            target_withdrawal_rate = (target_monthly_withdrawal * 12.0) / float(settings.initial_value)

    # CAPE guidance
    cape_row = df[df['Date'] <= asof].iloc[-1]
    cape_value = float(cape_row["CAPE"]) if pd.notna(cape_row["CAPE"]) else None
    if cape_value is not None and cape_value > 0:
        cape_sr = calculate_cape_withdrawal_rate(cape_value)
        cape_monthly = cape_sr * float(settings.initial_value) / 12.0
        cap_mult = settings.spending_cap_multiplier
        floor_mult = settings.spending_floor_multiplier
        if cap_mult is not None:
            cape_monthly = min(cape_monthly, float(settings.initial_monthly_spending) * cap_mult)
        if floor_mult is not None:
            cape_monthly = max(cape_monthly, float(settings.initial_monthly_spending) * floor_mult)
        baseline = float(settings.initial_monthly_spending)
        if cape_monthly < baseline:
            cape_monthly = min(cape_monthly + current_cashflow_amount, baseline)
        cape_monthly_withdrawal = max(cape_monthly - current_cashflow_amount, 0.0)
        if first_year_months:
            cape_annual_withdrawals = [
                max(float(cape_monthly) - float(cf), 0.0)
                for cf in monthly_cashflows[:first_year_months]
            ]
            cape_annual_withdrawal = float(np.sum(cape_annual_withdrawals))
        else:
            cape_annual_withdrawal = None
        cape_withdrawal_rate = (cape_monthly_withdrawal * 12.0) / float(settings.initial_value) if float(settings.initial_value) > 0 else None
    else:
        cape_sr = None
        cape_monthly = None
        cape_monthly_withdrawal = None
        cape_annual_withdrawal = None
        cape_withdrawal_rate = None

    return {
        "asof_date": asof,
        "months_remaining": num_months,
        "implied_spending_rate": implied_sr,
        "target_spending_rate": target_sr,
        "implied_withdrawal_rate": implied_withdrawal_rate,
        "target_withdrawal_rate": target_withdrawal_rate,
        "target_monthly_spending": target_monthly_spending,
        "target_monthly_withdrawal": target_monthly_withdrawal,
        "target_annual_withdrawal": target_annual_withdrawal,
        "cape_spending_rate": cape_sr,
        "cape_monthly_spending": cape_monthly,
        "cape_monthly_withdrawal": cape_monthly_withdrawal,
        "cape_annual_withdrawal": cape_annual_withdrawal,
        "cape_withdrawal_rate": cape_withdrawal_rate,
        "upper_guardrail_value": upper_guardrail_value,
        "lower_guardrail_value": lower_guardrail_value,
        "upper_adjusted_monthly": adj_upper_monthly,
        "upper_adjustment_pct": _pct_change(adj_upper_monthly),
        "lower_adjusted_monthly": adj_lower_monthly,
        "lower_adjustment_pct": _pct_change(adj_lower_monthly),
        "current_cashflow": current_cashflow_amount,
        "annual_cashflow_total": annual_cashflow_total,
    }
