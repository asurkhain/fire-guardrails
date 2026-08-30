import datetime
import os
import traceback
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

import controls
import display
import shiller_utils
import tax_utils
import utils
from app_settings import CashflowSetting, ConditionalCashflowSetting, Settings


if st.runtime.exists():
    st.set_page_config(layout="wide", page_title="Guardrail Withdrawal Simulator")

    # Apply configuration from hyperlink when available (only once per session)
    if "_settings_initialized" not in st.session_state:
        controls.hydrate_settings()

    # Mode toggle (top, persistent)
    mode = st.radio(
        "Mode",
        ["Simulation Mode", "Taxable Simulation Mode", "Historical Mode", "Taxable Historical Mode", "Combo Mode", "Guidance Mode"],
        index=0,
        horizontal=True,
        key="app_mode",
        label_visibility="collapsed"
    )
    is_guidance = (mode == "Guidance Mode")
    is_historical = (mode == "Historical Mode")
    is_taxable = (mode in ("Taxable Mode", "Taxable Historical Mode"))
    is_taxable_simulation = (mode == "Taxable Simulation Mode")
    is_tax_aware = is_taxable or is_taxable_simulation
    is_historical_like = is_historical or is_taxable
    is_simulation = (mode == "Simulation Mode")
    is_combo = (mode == "Combo Mode")
    controls.initialize_display()
    controls.init_initial_portfolio_value(mode)

    def _render_sidebar_label(text: str, color: Optional[str] = None) -> None:
        """Render styled sidebar helper text with optional highlighting."""

        style = (
            "margin: 0 0 0.25rem 0;"
            " font-size: var(--font-size-sm, 0.875rem);"
            " font-weight: var(--font-weight-normal, 400);"
            " line-height: var(--line-height-sm, 1.4);"
        )
        if color:
            style += f" color: {color};"
        st.sidebar.markdown(f"<div style=\"{style}\">{text}</div>", unsafe_allow_html=True)

    # Sidebar for inputs
    st.sidebar.header("Simulation Parameters")

    settings_error = st.session_state.get("_settings_error")
    if settings_error:
        st.sidebar.error(f"Unable to load shared configuration: {settings_error}")

    # Date Inputs
    today_date = datetime.date.today()
    start_date_default = datetime.date(1968, 1, 1)
    start_date_key = "retirement_start_date"
    sim_start_key = "_simulation_retirement_start_date"
    controls.init_start_date_field(today_date, sim_start_key, start_date_key, start_date_default, mode, is_guidance)

    if not is_historical_like and not is_combo:
        start_date = st.sidebar.date_input(
            "Retirement Start Date",
            min_value=datetime.date(1871, 1, 1),
            max_value=today_date,
            help="First retirement date used for withdrawals.\n\nCurrently, only the year and month are used, due to the monthly nature of the Shiller "
                 "dataset. The day of the month is ignored.\n\nIn Simulation Mode, the historical simulations begin from this date.\n\n"
                 "In Guidance Mode, this defaults to today. Even if retirement is already underway, the guidance is forward looking from today.\n\n"
                 "(Hint: You can type the date in YYYY/MM/DD format instead of choosing it from the selector, which may be faster.)",
            disabled=is_guidance,
            key=start_date_key,
        )
    else:
        start_date = datetime.date(1871, 1, 1)

    if is_guidance:
        start_date = today_date
    elif is_simulation or is_taxable_simulation:
        start_date = controls.get_date_state(start_date_key, start_date_default)
        st.session_state[sim_start_key] = start_date

    if is_historical_like or is_combo:
        retirement_duration_years = st.sidebar.number_input(
            "Retirement Duration (years)",
            value=controls.get_int_state("retirement_duration_years", 50),
            min_value=1,
            max_value=100,
            step=1,
            key="retirement_duration_years_hist",
            help="Length of each historical retirement in whole years. Every January 1 from the Historical Analysis Start Date "
                 "through the latest year that fits in the Shiller dataset is used as a retirement start.",
        )
        st.session_state["retirement_duration_years"] = int(retirement_duration_years)
        retirement_duration_months = int(retirement_duration_years) * 12
        st.session_state["retirement_duration_months"] = retirement_duration_months

        historical_future_extension_years = st.sidebar.number_input(
            "Historical Future Extension (years)",
            value=controls.get_int_state(
                "historical_future_extension_years",
                0,
            ),
            min_value=0,
            max_value=100,
            step=1,
            key="historical_future_extension_years",
            help="Repeats the last 12 months of Shiller data forward for this many years before stopping. "
                 "Set to 0 to keep the current end-of-data cutoff behavior.",
        )
    else:
        retirement_duration_years = st.sidebar.number_input(
            "Retirement Duration (years)",
            value=controls.get_int_state("retirement_duration_years", 30),
            min_value=1,
            max_value=100,
            step=1,
            key="retirement_duration_years_sim",
            help="Length of retirement in years.\n\nIn Guidance Mode, this should be the remaining number of years, if retirement is already underway."
        )
        retirement_duration_months = int(retirement_duration_years) * 12
        st.session_state["retirement_duration_months"] = retirement_duration_months

        if is_simulation or is_taxable_simulation:
            notice_ph = st.sidebar.empty()
            shiller_df = st.session_state.get("shiller_df")
            if shiller_df is not None and not shiller_df.empty:
                latest_shiller_date = pd.to_datetime(shiller_df["Date"].max()).date()
                projected_end = (pd.to_datetime(start_date) + pd.DateOffset(months=retirement_duration_months - 1)).date()
                if projected_end > latest_shiller_date:
                    sim_notice = (
                        f"Selected start date runs past Shiller data ending {latest_shiller_date:%Y-%m-%d}; "
                        "synthetic future data will be used."
                    )
                    notice_ph.markdown(
                        "<div style=\"color: #b00020; margin: -0.25rem 0 0.5rem 0; "
                        "font-size: var(--font-size-sm, 0.875rem); line-height: 1.35;\">"
                        f"{sim_notice}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    notice_ph.empty()
            else:
                notice_ph.empty()

    if is_historical_like or is_combo:
        analysis_start_date = st.sidebar.date_input(
            "Historical Analysis Start Date",
            value=controls.get_date_state("analysis_start_date", datetime.date(1871, 1, 1)),
            min_value=datetime.date(1871, 1, 1),
            max_value=datetime.date.today(),
            help="Earliest date for historical data included to estimate success rates (Shiller data begins in 1871).\n\nCurrently, only the year and month "
                 "are used, due to the monthly nature of the Shiller dataset. The day of the month is ignored.\n\nNote that when running historical "
                 "simulations, each month's guardrails will be recalculated based on the historical data available between this start date and that month "
                 "in history. A financial advisor running this strategy in the past would not have had a crystal ball to look into the future!"
                 + "\n\nIn Historical Mode, every retirement starting on January 1 from this date onward (for which a full retirement fits in the data) "
                   "is simulated."
                 + "\n\n(Hint: You can type the date in YYYY/MM/DD format instead of choosing it from the selector, which may be faster.)",
            key="analysis_start_date",
        )
    else:
        if "analysis_start_date" not in st.session_state:
            st.session_state["analysis_start_date"] = datetime.date(1871, 1, 1)
        analysis_start_date = st.session_state["analysis_start_date"]

    # Portfolio inputs
    if is_tax_aware:
        controls.init_taxable_accounts_widget_state()
        st.sidebar.markdown("**Starting Accounts**")
        st.sidebar.caption(
            "Each row is a holding. Edit the table, then select Apply account changes. Taxable bond returns are taxed as they occur; equity gains are taxed when sold."
        )
        edited_accounts_df = st.sidebar.data_editor(
            st.session_state[controls.TAXABLE_ACCOUNTS_DRAFT_KEY],
            column_config={
                "account_type": st.column_config.SelectboxColumn(
                    "Account Type",
                    options=["taxable", "retirement pretax", "retirement post tax"],
                    required=True,
                ),
                "asset_type": st.column_config.SelectboxColumn(
                    "Asset Type", options=["equity", "equity ex-US", "bond"], required=True,
                ),
                "cost_basis": st.column_config.NumberColumn("Cost Basis", min_value=0.0, format="$%.0f"),
                "total": st.column_config.NumberColumn("Total", min_value=0.0, format="$%.0f"),
            },
            num_rows="dynamic",
            hide_index=True,
            key="taxable_mode_accounts_editor",
        )
        apply_account_changes = st.sidebar.button(
            "Apply account changes",
            key="apply_taxable_accounts_btn",
        )
        if apply_account_changes:
            editor_state = st.session_state.get("taxable_mode_accounts_editor", edited_accounts_df)
            saved, message = controls.commit_taxable_accounts_from_editor(editor_state)
            if saved:
                st.sidebar.success(message)
                st.rerun()
            else:
                st.sidebar.error(message)
        taxable_accounts = st.session_state["taxable_mode_accounts"]
        try:
            normalized_accounts = tax_utils.normalize_accounts(taxable_accounts)
        except ValueError as exc:
            st.sidebar.error(str(exc))
            normalized_accounts = []
        initial_value = sum(account.total for account in normalized_accounts)
        equity_value = sum(account.total for account in normalized_accounts if account.asset_type == "equity")
        current_stock_pct = equity_value / initial_value if initial_value > 0 else 0.0
        st.session_state["initial_portfolio_value"] = initial_value
        st.sidebar.caption(
            f"Portfolio total: ${initial_value:,.0f}  |  Current equity allocation: {current_stock_pct:.0%}"
        )
        marital_status = st.sidebar.selectbox(
            "Marital Status", ["single", "mfj"], key="taxable_mode_marital_status",
            format_func=lambda value: "Single" if value == "single" else "Married filing jointly",
        )
        current_age = st.sidebar.number_input(
            "Current Age", min_value=0, max_value=120, value=42, step=1, key="taxable_mode_current_age",
        )
    else:
        taxable_accounts = []
        marital_status = None
        current_age = None
        initial_value = st.sidebar.number_input(
            "Initial Portfolio Value",
            min_value=100_000.0,
            step=100_000.0,
            key="initial_portfolio_value",
            help="Starting portfolio balance in dollars at retirement."
        )

    if is_combo:
        stock_pct_range = st.sidebar.slider(
            "Stock Percentage Range",
            value=st.session_state.get("stock_pct_range", (0.50, 0.90)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="stock_pct_range",
            help="Range of stock allocation to test across simulations."
        )
        stock_pct_min, stock_pct_max = stock_pct_range
        stock_pct = (stock_pct_min + stock_pct_max) / 2.0
    elif is_tax_aware:
        stock_pct = st.sidebar.slider(
            "Stock Percentage",
            value=float(st.session_state.get("stock_pct", controls.DEFAULT_STOCK_PCT)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="stock_pct",
            help="Target equity allocation used for the retirement-only rebalance after each withdrawal. "
                 "Taxable holdings are not traded during rebalancing.",
        )
    else:
        stock_pct = st.sidebar.slider(
            "Stock Percentage",
            value=float(st.session_state.get("stock_pct", controls.DEFAULT_STOCK_PCT)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="stock_pct",
            help="Fraction of the portfolio allocated to US stocks; remainder to 10Y treasuries."
        )

    cap_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 201, 5)]
    floor_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 24, -5)]

    controls.sync_cashflows_from_widgets()
    controls.sync_conditional_cashflows_from_widgets()

    sanitized_cashflows = controls.sanitize_cashflows(st.session_state.get("cashflows"))
    sanitized_conditional_cashflows = controls.sanitize_conditional_cashflows(st.session_state.get("conditional_cashflows"))
    if is_combo:
        _render_sidebar_label("Target Success Rate Range")
        target_success_rate_range = st.sidebar.slider(
            "Target Success Rate Range",
            value=st.session_state.get("target_success_rate_range", (0.70, 0.90)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="target_success_rate_range",
            help="Desired probability of success that will be used to select an initial spending rate.\n\nThe initial "
                 "spending rate will be the rate at which fixed spending over all periods of time with length = the "
                 "configured retirement period length, between the Historical Analysis Start Date and the Retirement Start "
                 "Date, end with >0 values this percent of the time.\n\nSetting this higher, e.g. 0.80-0.99, is more conservative: "
                 "lower initial spending, lower chance of adjustment; setting this lower is more aggressive, 0.75-0.60 provides higher "
                 "initial spending, higher chance of adjustment.",
            label_visibility="collapsed",
        )
        target_success_rate_min, target_success_rate_max = target_success_rate_range
        target_success_rate = (target_success_rate_min + target_success_rate_max) / 2.0
    else:
        isr_params = {
            'start_date': pd.to_datetime(start_date),
            'duration_months': int(retirement_duration_months),
            'analysis_start_date': pd.to_datetime(analysis_start_date),
            'initial_value': float(initial_value),
            'stock_pct': float(stock_pct),
            'desired_success_rate': float(st.session_state.get("target_success_rate", 0.70)),
            'final_value_target': float(st.session_state.get('final_value_target', 0.0)),
            'final_value_target_ramp_down': bool(st.session_state.get('final_value_target_ramp_down', False)),
            'cashflows': controls.cashflows_to_tuple(sanitized_cashflows),
        }
        isr_label_suffix = ""

        try:
            if 'isr_params' not in st.session_state or st.session_state['isr_params'] != isr_params:
                display.update_isr_dynamic_label(isr_params=isr_params, cashflows=sanitized_cashflows)
            isr = st.session_state.get('isr_value')
            if isr is not None:
                isr_label_suffix = f" (Initial SR: {isr*100:.2f}%)"
                auto_initial_yearly_spending = round(float(isr) * 100, 2)
            else:
                isr_label_suffix = " (Initial SR: N/A)"
                auto_initial_yearly_spending = None

        except Exception as e:
            print(e)
            isr_label_suffix = " (Initial SR: N/A)"
            auto_initial_yearly_spending = None

        target_success_label = f"Target Success Rate{isr_label_suffix}"
        target_success_rate = st.sidebar.slider(
            target_success_label,
            value=st.session_state.get("target_success_rate", 0.80),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="Desired probability of success that will be used to select an initial spending rate.\n\nThe initial "
                 "spending rate will be the rate at which fixed spending over all periods of time with length = the "
                 "configured retirement period length, between the Historical Analysis Start Date and the Retirement Start "
                 "Date, end with >0 values this percent of the time.\n\nSetting this higher, e.g. 0.80-0.99, is more conservative: "
                 "lower initial spending, lower chance of adjustment; setting this lower is more aggressive, 0.75-0.60 provides higher "
                 "initial spending, higher chance of adjustment.",
            key="target_success_rate",
        )

    _render_sidebar_label("Initial Yearly Spending %")

    st.sidebar.number_input(
        "Initial Yearly Spending %",
        min_value=0.0,
        step=0.25,
        format="%.2f",
        help="The initial yearly spending as a percentage of the initial portfolio value.\n\n"
             "e.g., 4.25 means 4.25% of the initial portfolio per year.\n\n"
             "Automatically updates as you change the Target Spending Rate, but can be set to a custom value if desired.\n\n",
        key="initial_yearly_spending",
        label_visibility="collapsed",
    )

    initial_yearly_spending_pct = float(st.session_state.get("initial_yearly_spending", 0.0))
    initial_monthly_spending = initial_value * initial_yearly_spending_pct / 100.0 / 12.0
    st.session_state["initial_monthly_spending"] = initial_monthly_spending

    if is_combo:
        _render_sidebar_label("Upper Guardrail Success Rate Range")
        upper_guardrail_range = st.sidebar.slider(
            "Upper Guardrail Success Rate Range",
            value=st.session_state.get("upper_guardrail_success_range", (0.80, 1.00)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="upper_guardrail_success_range",
            help="Range of success rates used to calculate the upper guardrail portfolio value.",
            label_visibility="collapsed",
        )
        upper_guardrail_success_min, upper_guardrail_success_max = upper_guardrail_range

        _render_sidebar_label("Lower Guardrail Success Rate Range")
        lower_guardrail_range = st.sidebar.slider(
            "Lower Guardrail Success Rate Range",
            value=st.session_state.get("lower_guardrail_success_range", (0.25, 0.45)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="lower_guardrail_success_range",
            help="Range of success rates used to calculate the lower guardrail portfolio value.",
            label_visibility="collapsed",
        )
        lower_guardrail_success_min, lower_guardrail_success_max = lower_guardrail_range

        upper_guardrail_success = upper_guardrail_success_max
        lower_guardrail_success = lower_guardrail_success_min
    else:
        # Compute dynamic labels for Guardrail Success Rates showing initial (first period) PVs
        upper_label_suffix = ""
        lower_label_suffix = ""
        try:
            gr_params = {
                'start_date': pd.to_datetime(start_date),
                'duration_months': int(retirement_duration_months),
                'analysis_start_date': pd.to_datetime(analysis_start_date),
                'initial_value': float(initial_value),
                'stock_pct': float(stock_pct),
                'upper_sr': float(st.session_state.get("upper_guardrail_success", 1.00)),
                'lower_sr': float(st.session_state.get("lower_guardrail_success", 0.75)),
                'isr': float(st.session_state.get('isr_value')) if st.session_state.get('isr_value') is not None else None,
                'initial_spending': float(initial_monthly_spending),
                'final_value_target': float(st.session_state.get('final_value_target', 0.0)),
                'final_value_target_ramp_down': bool(st.session_state.get('final_value_target_ramp_down', False)),
                'cashflows': controls.cashflows_to_tuple(sanitized_cashflows),
            }

            if ('guardrail_params' not in st.session_state) or (st.session_state['guardrail_params'] != gr_params):
                display.update_guardrail_dynamic_labels(gr_params=gr_params, cashflows=sanitized_cashflows)

            upper_label_suffix = st.session_state.get('upper_label_suffix', " (Initial PV: N/A)")
            lower_label_suffix = st.session_state.get('lower_label_suffix', " (Initial PV: N/A)")

        except Exception as e:
            print(e)
            upper_label_suffix = " (Initial PV: N/A)"
            lower_label_suffix = " (Initial PV: N/A)"
            st.session_state['upper_label_color'] = None
            st.session_state['lower_label_color'] = None

        upper_guardrail_label = f"Upper Guardrail Success Rate{upper_label_suffix}"
        lower_guardrail_label = f"Lower Guardrail Success Rate{lower_label_suffix}"

        upper_label_color = st.session_state.get('upper_label_color')
        lower_label_color = st.session_state.get('lower_label_color')

        _render_sidebar_label(upper_guardrail_label, upper_label_color)

        upper_guardrail_success = st.sidebar.slider(
            upper_guardrail_label,
            value=st.session_state.get("upper_guardrail_success", 1.00),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="The spending rate used to calculate the upper guardrail portfolio value.\n\nThis is the value where the "
                 "current spending amount, if held constant, will succeed this frequently or more, for all periods with "
                 "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
                 "simulation date.\n\nSetting this higher is more conservative, and will cause you to wait longer to increase "
                 "your spending when markets are up.",
            key="upper_guardrail_success",
            label_visibility="collapsed"
        )

        _render_sidebar_label(lower_guardrail_label, lower_label_color)

        lower_guardrail_success = st.sidebar.slider(
            lower_guardrail_label,
            value=st.session_state.get("lower_guardrail_success", 0.25),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="The spending rate used to calculate the lower guardrail portfolio value.\n\nThis is the value where the "
                 "current spending amount, if held constant, will succeed this frequently or less, for all periods with "
                 "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
                 "simulation date.\n\nSetting this higher is more conservative, and will cause you to decrease your spending "
                 "sooner when markets are down.",
            key="lower_guardrail_success",
            label_visibility="collapsed"
        )

    upper_adjustment_fraction = st.sidebar.slider(
        "Upper Adjustment Fraction",
        value=controls.get_float_state("upper_adjustment_fraction", 0.5),
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="upper_adjustment_fraction",
        help="How much to increase spending when we hit the upper guardrail.\n\nExpressed as a % of the distance between "
             "the Upper Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the upper guardrail "
             "represents 100% success and the target is 90%, setting this value to 50% means we go half the distance back "
             "to the target, and our new spending rate will be based on a 95% chance of success.\n\nSetting this higher "
             "is more aggressive, and will cause you to make larger spending increases when you hit the upper guardrail."
    )

    lower_adjustment_fraction = st.sidebar.slider(
        "Lower Adjustment Fraction",
        value=controls.get_float_state("lower_adjustment_fraction", 0.35),
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="lower_adjustment_fraction",
        help="How much to decrease spending when we hit the lower guardrail.\n\nExpressed as a % of the distance between "
             "the Lower Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the lower guardrail "
             "represents 70% success and the target is 90%, setting this value to 50% means we go half the distance back "
             "to the target, and our new spending rate will be based on an 80% chance of success.\n\nSetting this higher "
             "is more conservative, and will cause you to make larger spending decreases when you hit the lower guardrail."
    )

    default_threshold = 0.0 if is_guidance else 0.05
    adjustment_threshold = st.sidebar.slider(
        "Adjustment Threshold (e.g., 0.05 for 5%)",
        value=controls.get_float_state("adjustment_threshold", default_threshold),
        min_value=0.0,
        max_value=0.2,
        step=0.01,
        key="adjustment_threshold",
        help="The minimum percent difference between our new spending and our prior spending, before we make a change.\n\n"
             "Even if we hit a guardrail, we may elect to set this to 5% to avoid making lots of small adjustments. Set it "
             "to 0% to disable it and allow all guardrail hits to trigger spending adjustments.\n\nSetting this higher is "
             "neither aggressive nor conservative, since it impacts spending increases as well as decreases. It is purely "
             "a question of whether frequent adjustments are acceptable and administratively feasible for the client.\n\n"
             "Not used in Guidance Mode, which will always show spending adjustment suggestions, no matter how small.",
        disabled=is_guidance  # In Guidance Mode, single-run snapshot ignores threshold
    )

    frequency_options = ["Monthly", "Quarterly", "Biannually", "Annually"]
    current_frequency = st.session_state.get("adjustment_frequency", "Quarterly")
    try:
        frequency_index = frequency_options.index(str(current_frequency))
    except ValueError:
        frequency_index = 0
    adjustment_frequency = st.sidebar.selectbox(
        "Adjustment Frequency",
        options=frequency_options,
        index=frequency_index,
        key="adjustment_frequency",
        help="How often spending adjustments are permitted. Choosing Quarterly, Biannually, or Annually restricts guardrail checks "
             "and any resulting spending changes to the beginning of those periods (Jan/Apr/Jul/Oct, Jan/Jul, or January).",
        disabled=is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
    )

    with st.sidebar.expander("Advanced Controls"):
        shiller_extension_options = ["last_year", "average_return", "hard_code_date"]
        shiller_extension_labels = {
            "last_year": "Last Year",
            "average_return": "Average Return",
            "hard_code_date": "Hard Code Date",
        }
        st.selectbox(
            "Shiller Extension Mode",
            options=shiller_extension_options,
            format_func=lambda value: shiller_extension_labels.get(value, value),
            key="shiller_extension_mode",
            help="Controls how synthetic Shiller data is extended beyond the end of the real dataset.",
        )
        st.selectbox(
            "Spending Cap",
            options=cap_options,
            key="spending_cap_option",
            help="Maximum spending level as a percent of the initial yearly spending.",
            disabled=is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
        )
        st.selectbox(
            "Spending Floor",
            options=floor_options,
            key="spending_floor_option",
            help="Minimum spending level as a percent of the initial yearly spending.",
            disabled = is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
        )
        st.number_input(
            "Final Value Target (Buffer)",
            min_value=0.0,
            max_value=float(initial_value),
            step=10000.0,
            format="%.0f",
            key="final_value_target",
            help="Minimum ending portfolio value. "
                 "This affects both the success rate calculation and the recommended safe spending rate.\n\n"
                 "Note that setting this too high may result in huge upper guardrail values - particularly "
                 "if the portfolio value during retirement crosses below this value, as the required spending "
                 "% rate (the denominator of the calculation) for the portfolio to end higher than its current "
                 "value approaches zero.",
        )
        st.checkbox(
            "Ramp Down Final Value Target",
            key="final_value_target_ramp_down",
            help="Keep the buffer flat until the final 10% of retirement, then linearly reduce it to zero by the end.",
        )

        if st.button("Add Recurring Cashflow", key="add_cashflow_btn"):
            st.session_state["cashflows"].append({
                "start_month": 0,
                "end_month": 0,
                "amount": 0.0,
                "label": f"Cashflow {len(st.session_state['cashflows']) + 1}",
            })
            controls.save_recurring_cashflows_to_csv(st.session_state["cashflows"])
            st.rerun()

        controls.draw_cashflow_widget_rows()

        if st.button("Add Conditional Cashflow", key="add_conditional_cashflow_btn"):
            st.session_state["conditional_cashflows"].append({
                "cashflow_type": "one_time",
                "trigger_threshold": "50%",
                "amount": 0.0,
                "label": f"Conditional {len(st.session_state['conditional_cashflows']) + 1}",
            })
            st.rerun()

        controls.draw_conditional_cashflow_widget_rows()

    if is_historical_like or is_simulation or is_taxable_simulation or is_combo:
        _render_sidebar_label("Fixed Yearly Withdrawal %")

        st.sidebar.number_input(
            "Fixed Yearly Withdrawal %",
            min_value=0.0,
            step=0.25,
            format="%.2f",
            key="fixed_yearly_withdrawal_input",
            help="Optional fixed yearly withdrawal as a percentage of the initial portfolio value, used only for comparison.\n\n"
                 "e.g., 3.75 means 3.75% of the initial portfolio per year.\n\n"
                 "Leave at 0 to skip fixed-withdrawal comparison.",
            label_visibility="collapsed",
        )
        _fixed_yearly_pct = float(st.session_state.get("fixed_yearly_withdrawal_input", 0.0) or 0.0)
        fixed_monthly_withdrawal = (
            initial_value * _fixed_yearly_pct / 100.0 / 12.0 if _fixed_yearly_pct > 0 else None
        )
        st.session_state["fixed_monthly_withdrawal"] = fixed_monthly_withdrawal
    else:
        fixed_monthly_withdrawal = None
        st.session_state["fixed_monthly_withdrawal"] = None

    fixed_yearly_withdrawal_including_taxes = None
    if is_taxable and fixed_monthly_withdrawal is not None and normalized_accounts:
        fixed_funding = tax_utils.fund_spending_with_tax(
            normalized_accounts,
            float(fixed_monthly_withdrawal) * 12.0,
            marital_status,
            current_age=current_age,
            target_stock_pct=stock_pct,
        )
        fixed_yearly_withdrawal_including_taxes = float(fixed_funding["gross_withdrawal"])


    # Build Settings object representing the full control state
    cashflow_settings = [
        cf for cf in (CashflowSetting.from_dict(flow) for flow in sanitized_cashflows)
        if cf is not None
    ]
    conditional_cashflow_settings = [
        cf for cf in (ConditionalCashflowSetting.from_dict(flow) for flow in sanitized_conditional_cashflows)
        if cf is not None
    ]

    settings = Settings(
        mode=mode,
        start_date=start_date,
        retirement_duration_months=int(retirement_duration_months),
        analysis_start_date=analysis_start_date,
        initial_value=float(initial_value),
        stock_pct=float(stock_pct),
        target_success_rate=float(target_success_rate),
        initial_monthly_spending=float(initial_monthly_spending),
        initial_spending_overridden=bool(st.session_state.get("_initial_spending_overridden", False)),
        upper_guardrail_success=float(upper_guardrail_success),
        lower_guardrail_success=float(lower_guardrail_success),
        upper_adjustment_fraction=float(upper_adjustment_fraction),
        lower_adjustment_fraction=float(lower_adjustment_fraction),
        adjustment_threshold=float(adjustment_threshold),
        adjustment_frequency=adjustment_frequency,
        spending_cap_option=st.session_state.get(
            "spending_cap_option", controls.DEFAULT_SPENDING_CAP_OPTION,
        ),
        spending_floor_option=st.session_state.get(
            "spending_floor_option", controls.DEFAULT_SPENDING_FLOOR_OPTION,
        ),
        shiller_extension_mode=st.session_state.get("shiller_extension_mode", shiller_utils.SHILLER_EXTENSION_MODE),
        final_value_target=float(st.session_state.get("final_value_target", 0.0)),
        final_value_target_ramp_down=bool(st.session_state.get("final_value_target_ramp_down", False)),
        fixed_monthly_withdrawal=fixed_monthly_withdrawal,
        historical_future_extension_years=(
            int(historical_future_extension_years) if (is_historical_like or is_combo) else None
        ),
        cashflows=cashflow_settings,
        conditional_cashflows=conditional_cashflow_settings,
        accounts=list(taxable_accounts) if is_tax_aware else [],
        marital_status=marital_status if is_tax_aware else "single",
        current_age=float(current_age) if is_tax_aware and current_age is not None else None,
    )

    st.session_state["settings"] = settings

    # Warn if guardrail success rates are in unexpected order
    if not is_combo and not (lower_guardrail_success <= target_success_rate <= upper_guardrail_success):
        st.sidebar.warning(
            "Guardrail success rates are in an unusual order. Typically: "
            "Lower Guardrail \u2264 Target \u2264 Upper Guardrail. "
            "Current values may produce unexpected behavior."
        )

    encoded_config = settings.to_base64()
    st.session_state["_encoded_settings"] = encoded_config
    share_link_url = f"?config={encoded_config}"

    sim_signature = settings.simulation_signature()
    if is_tax_aware:
        sim_signature["taxable_accounts"] = tuple(
            (account.account_type, account.asset_type, account.cost_basis, account.total)
            for account in normalized_accounts
        )
        sim_signature["marital_status"] = marital_status
        sim_signature["current_age"] = current_age
        sim_signature["taxable_target_stock_pct"] = stock_pct
    last_run_signature = st.session_state.get('last_run_signature')
    dirty = last_run_signature is not None and last_run_signature != sim_signature
    st.session_state['dirty'] = dirty

    # When inputs change, visually dim and surround the main area with a red border
    if dirty and not is_guidance and not is_combo:
        controls.draw_dirty_border()

    # ------ Main Program Logic -------
    #
    if is_guidance:
        # Guidance Mode: run a single-iteration snapshot and display text output

        shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)

        # Use the latest available Shiller data date (<= today) as the as-of date.
        today = datetime.date.today()
        latest_shiller_date = pd.to_datetime(shiller_df["Date"].max()).date()
        asof_date = latest_shiller_date if latest_shiller_date <= today else today

        try:
            snap = utils.compute_guardrail_guidance_snapshot(
                df=shiller_df,
                asof_date=asof_date,
                settings=settings,
            )

            display.render_guidance_results(snap=snap)

        except Exception as e:
            st.error(f"Unable to compute guidance snapshot: {e}")

    elif is_historical_like:
        if st.sidebar.button(
            "Run Taxable Analysis" if is_taxable else "Run Historical Analysis",
            help="Run the guardrail strategy for every historical retirement start year that fits in the Shiller dataset.",
        ):
            status_ph = st.empty()
            status_ph.text("Loading Shiller data...")
            shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)
            status_ph.text("Shiller data loaded.")

            progress = status_ph.progress(0, text=("Running taxable retirements... 0%" if is_taxable else "Running historical retirements... 0%"))
            state = {"pct": 0, "status": None}

            def render_historical_progress():
                label = f"Running {'taxable' if is_taxable else 'historical'} retirements... {state['pct']}%"
                if state["status"]:
                    label = f"{label} — {state['status']}"
                progress.progress(state["pct"], text=label)

            def on_historical_progress(current, total):
                pct = int(current * 100 / total) if total else 0
                state["pct"] = pct
                render_historical_progress()

            def on_historical_status(msg):
                state["status"] = msg
                render_historical_progress()

            try:
                run_started_at = perf_counter()
                historical_results_df = (tax_utils if is_taxable else utils).run_historical_retirements_analysis(
                    df=shiller_df,
                    settings=settings,
                    on_progress=on_historical_progress,
                    on_status=on_historical_status,
                )
                historical_elapsed_seconds = perf_counter() - run_started_at
                st.session_state["historical_results_df"] = historical_results_df
                st.session_state["historical_elapsed_seconds"] = historical_elapsed_seconds
                st.session_state["last_run_signature"] = sim_signature
                st.session_state["dirty"] = False
                status_ph.empty()
                display.render_historical_results(
                    historical_results_df,
                    calculation_time_seconds=historical_elapsed_seconds,
                    fixed_monthly_withdrawal=fixed_monthly_withdrawal,
                    fixed_yearly_withdrawal_including_taxes=fixed_yearly_withdrawal_including_taxes,
                )
            except Exception as e:
                status_ph.empty()
                traceback.print_exc()
                st.error(f"Unable to run historical analysis: {e}")

        elif "historical_results_df" in st.session_state:
            if st.session_state.get("dirty"):
                controls.render_dirty_banner()
            display.render_historical_results(
                st.session_state["historical_results_df"],
                calculation_time_seconds=st.session_state.get("historical_elapsed_seconds"),
                fixed_monthly_withdrawal=fixed_monthly_withdrawal,
                fixed_yearly_withdrawal_including_taxes=fixed_yearly_withdrawal_including_taxes,
            )
        else:
            st.subheader("Taxable Historical Mode" if is_taxable else "Historical Mode")
            st.markdown(
                ("Use this mode to run the tax-aware guardrail withdrawal strategy across every historical retirement start year "
                 if is_taxable else "Use this mode to run the guardrail withdrawal strategy across every historical retirement start year ")
                + "for a fixed duration.\n\n"
                "All dollar amounts shown are in real (constant) dollars, net of inflation. For more details, see the "
                "[documentation](https://github.com/asurkhain/fire-guardrails/blob/main/README.md)."
            )
            st.info(f"Adjust parameters in the sidebar and click 'Run {'Taxable' if is_taxable else 'Historical'} Analysis' to start.")

    elif is_combo:
        if st.sidebar.button(
            "Run Combo Analysis",
            help="Run the guardrail withdrawal strategy across the configured parameter ranges.",
        ):
            status_ph = st.empty()
            status_ph.text("Loading Shiller data...")
            shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)
            status_ph.text("Shiller data loaded.")

            progress = status_ph.progress(0, text="Running combo analysis... 0%")
            state = {"pct": 0, "status": None}

            def render_combo_progress():
                label = f"Running combo analysis... {state['pct']}%"
                if state["status"]:
                    label = f"{label} — {state['status']}"
                progress.progress(state["pct"], text=label)

            def on_combo_progress(current, total):
                pct = int(current * 100 / total) if total else 0
                state["pct"] = pct
                render_combo_progress()

            def on_combo_status(msg):
                state["status"] = msg
                render_combo_progress()

            try:
                run_started_at = perf_counter()
                stock_pct_range = st.session_state.get("stock_pct_range", (0.50, 0.90))
                ugr_range = st.session_state.get("upper_guardrail_success_range", (0.80, 1.00))
                lgr_range = st.session_state.get("lower_guardrail_success_range", (0.25, 0.45))
                tsr_range = st.session_state.get("target_success_rate_range", (0.70, 0.90))
                combo_results = utils.run_combo_analysis(
                    df=shiller_df,
                    settings=settings,
                    stock_pct_range=stock_pct_range,
                    upper_guardrail_success_range=ugr_range,
                    lower_guardrail_success_range=lgr_range,
                    target_success_rate_range=tsr_range,
                    on_progress=on_combo_progress,
                    on_status=on_combo_status,
                )
                combo_elapsed_seconds = perf_counter() - run_started_at
                st.session_state["combo_results"] = combo_results
                st.session_state["combo_elapsed_seconds"] = combo_elapsed_seconds
                st.session_state["last_run_signature"] = sim_signature
                st.session_state["dirty"] = False
                status_ph.empty()
                display.render_combo_results(
                    combo_results,
                    calculation_time_seconds=combo_elapsed_seconds,
                    fixed_monthly_withdrawal=fixed_monthly_withdrawal,
                )
                # Auto-save with settings-encoded filename
                _sp = stock_pct_range
                _ugr = ugr_range
                _lgr = lgr_range
                _tsr = tsr_range
                _uaf = settings.upper_adjustment_fraction
                _laf = settings.lower_adjustment_fraction
                _fname = (
                    f"combo_sp={_sp[0]:.0%}-{_sp[1]:.0%}"
                    f"_ugr={_ugr[0]:.0%}-{_ugr[1]:.0%}"
                    f"_lgr={_lgr[0]:.0%}-{_lgr[1]:.0%}"
                    f"_tsr={_tsr[0]:.0%}-{_tsr[1]:.0%}"
                    f"_uaf={_uaf:.0%}_laf={_laf:.0%}.joblib"
                )
                try:
                    utils.save_combo_results(combo_results, str(Path(utils.get_combo_tempdir()) / _fname))
                except Exception as e2:
                    pass
                _combo_dir = Path(utils.get_combo_tempdir())
                _combo_files = sorted([f.name for f in _combo_dir.iterdir() if f.suffix == ".joblib"])
                if _combo_files:
                    _selected = st.selectbox("Switch to saved combo results", _combo_files, key="combo_switch_select", index=0)
                    if st.button("Load", key="combo_switch_btn"):
                        try:
                            loaded = utils.load_combo_results(str(_combo_dir / _selected))
                            st.session_state["combo_results"] = loaded
                            st.session_state["combo_elapsed_seconds"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load combo results: {e}")
            except Exception as e:
                status_ph.empty()
                traceback.print_exc()
                st.error(f"Unable to run combo analysis: {e}")

        elif "combo_results" in st.session_state:
            if st.session_state.get("dirty"):
                controls.render_dirty_banner()
            display.render_combo_results(
                st.session_state["combo_results"],
                calculation_time_seconds=st.session_state.get("combo_elapsed_seconds"),
                fixed_monthly_withdrawal=fixed_monthly_withdrawal,
            )
            _combo_dir = Path(utils.get_combo_tempdir())
            _combo_files = sorted([f.name for f in _combo_dir.iterdir() if f.suffix == ".joblib"])
            if _combo_files:
                _selected = st.selectbox("Switch to saved combo results", _combo_files, key="combo_switch_select", index=0)
                if st.button("Load", key="combo_switch_btn"):
                    try:
                        loaded = utils.load_combo_results(str(_combo_dir / _selected))
                        st.session_state["combo_results"] = loaded
                        st.session_state["combo_elapsed_seconds"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load combo results: {e}")
        else:
            st.subheader("Combo Mode")
            st.markdown(
                "Use this mode to run the guardrail withdrawal strategy across a grid of parameters, "
                "varying stock allocation and guardrail success rates within the configured ranges."
                "Note, the three Success Rate Ranges are dependent e.g. the min of all of them is 1 run, then"
                " the next run is 5% added to all three, etc.\n\n"
                "All dollar amounts shown are in real (constant) dollars, net of inflation. For more details, see the "
                "[documentation](https://github.com/asurkhain/fire-guardrails/blob/main/README.md)."
            )
            _combo_dir = Path(utils.get_combo_tempdir())
            _combo_files = sorted([f.name for f in _combo_dir.iterdir() if f.suffix == ".joblib"])
            if _combo_files:
                _selected = st.selectbox("Load previous combo results", _combo_files, key="combo_load_select")
                if st.button("Load from File", key="combo_load_btn"):
                    try:
                        loaded = utils.load_combo_results(str(_combo_dir / _selected))
                        st.session_state["combo_results"] = loaded
                        st.session_state["combo_elapsed_seconds"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load combo results: {e}")
            else:
                st.caption("No saved combo results found.")
            st.info("Adjust parameters in the sidebar and click 'Run Combo Analysis' to start.")

    elif is_taxable_simulation and st.sidebar.button(
        "Run Taxable Simulation",
        help="Run the guardrail simulation with account-level tax diagnostics.",
    ):
        status_ph = st.empty()
        status_ph.text("Loading Shiller data...")
        shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)
        status_ph.text("Calculating taxable simulation...")
        base_results = tax_utils.get_guardrail_withdrawals(
            df=shiller_df,
            settings=settings,
            verbose=True,
        )
        taxable_results_df = tax_utils.add_tax_metrics_to_withdrawal_results(
            base_results,
            normalized_accounts,
            marital_status,
            current_age,
            target_stock_pct=stock_pct,
            market_data=shiller_df,
        )
        st.session_state["taxable_simulation_results_df"] = taxable_results_df
        st.session_state["last_run_signature"] = sim_signature
        st.session_state["dirty"] = False
        status_ph.empty()
        display.render_simulation_results(
            taxable_results_df,
            fixed_monthly_withdrawal=fixed_monthly_withdrawal,
            current_age=current_age,
        )

    elif is_taxable_simulation:
        if "taxable_simulation_results_df" in st.session_state:
            if st.session_state.get("dirty"):
                controls.render_dirty_banner()
            display.render_simulation_results(
                st.session_state["taxable_simulation_results_df"],
                fixed_monthly_withdrawal=fixed_monthly_withdrawal,
                current_age=current_age,
            )
        else:
            st.subheader("Taxable Simulation Mode")
            st.markdown(
                "Use this mode to run a guardrail simulation with account-based tax withdrawals. "
                "The Tax Rate Over Time chart reports taxes paid divided by the tax-inclusive withdrawal."
            )
            st.info("Adjust parameters in the sidebar and click 'Run Taxable Simulation' to start.")

    elif is_simulation and st.sidebar.button(
        "Run Simulation",
        help="Fetch data and run the guardrail withdrawal simulation with the selected parameters.",
    ):
        status_ph = st.empty()
        status_ph.text("Loading Shiller data...")
        shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)
        status_ph.text("Shiller data loaded.")

        # Progress bar shown in the same status line area (0% -> 100%)
        progress = status_ph.progress(0, text="Calculating guardrail withdrawals... 0%")
        state = {"pct": 0, "status": None}

        def render_progress():
            label = f"Calculating guardrail withdrawals... {state['pct']}%"
            if state["status"]:
                label = f"{label} — {state['status']}"
            progress.progress(state["pct"], text=label)

        def on_progress(current, total):
            pct = int(current * 100 / total) if total else 0
            state["pct"] = pct
            render_progress()

        def on_status(msg):
            state["status"] = msg
            render_progress()

        results_df = utils.get_guardrail_withdrawals(
            df=shiller_df,
            settings=settings,
            verbose=True,
            on_progress=on_progress,
            on_status=on_status
        )

        # Cache results in session state for re-render without recomputation
        st.session_state['results_df'] = results_df
        st.session_state['last_run_signature'] = sim_signature
        st.session_state['dirty'] = False

        # Remove status line entirely to place plots at the very top
        status_ph.empty()

        display.render_simulation_results(results_df, fixed_monthly_withdrawal=fixed_monthly_withdrawal)

    elif is_simulation:
        if "results_df" in st.session_state:
            results_df = st.session_state["results_df"]

            if st.session_state.get("dirty"):
                controls.render_dirty_banner()

            display.render_simulation_results(results_df, fixed_monthly_withdrawal=fixed_monthly_withdrawal)
        else:
            st.subheader("Simulation Mode")
            st.markdown(
                "Use this mode to simulate running a guardrail-based retirement withdrawal strategy during a historical period.\n\n"
                "All dollar amounts shown are in real (constant) dollars, net of inflation. For more details, see the "
                "[documentation](https://github.com/asurkhain/fire-guardrails/blob/main/README.md"
            )
            st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")

    st.divider()
    st.markdown(f"[Shareable link to this run]({share_link_url})")
    st.caption("Copy the link to load these settings on any device.")
