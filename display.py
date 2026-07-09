import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from typing import Optional, Tuple
import shiller_utils
import utils

MONTHS_PER_YEAR = 12


def monthly_to_yearly(amount: Optional[float]) -> Optional[float]:
    """Convert a monthly dollar amount to its yearly equivalent."""
    if amount is None:
        return None
    return float(amount) * MONTHS_PER_YEAR


def yearly_to_monthly(amount: Optional[float]) -> Optional[float]:
    """Convert a yearly dollar amount to its monthly equivalent for calculations."""
    if amount is None:
        return None
    return float(amount) / MONTHS_PER_YEAR


def _fmt_currency(value, escape_for_markdown: bool = False) -> str:
    """Format a currency value with optional Markdown escaping."""
    if value is None or (isinstance(value, float) and (pd.isna(value) or not np.isfinite(value))):
        return "N/A"
    prefix = "\\$" if escape_for_markdown else "$"
    if value < 0:
        return f"-{prefix}{abs(value):,.0f}"
    return f"{prefix}{value:,.0f}"


def update_isr_dynamic_label(isr_params: dict, cashflows: list):
    """
    Updates the dynamic label on the target success rate input that shows the corresponding initial spending rate.
    """
    shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)
    stock_prices = shiller_df["Real Total Return Price"].values
    bond_prices = shiller_df["Real Total Bond Returns"].values
    portfolio_returns = utils.compute_portfolio_returns(stock_prices, bond_prices, isr_params["stock_pct"])
    effective_final_value_target = utils.get_effective_final_value_target(
        final_value_target=isr_params.get('final_value_target', 0.0),
        total_retirement_months=isr_params['duration_months'],
        months_remaining=isr_params['duration_months'],
        ramp_down=bool(isr_params.get('final_value_target_ramp_down', False)),
    )

    res = utils.get_spending_rate_for_fixed_success_rate(
        df=shiller_df,
        desired_success_rate=isr_params['desired_success_rate'],
        num_months=isr_params['duration_months'],
        #analysis_start_date=isr_params['analysis_start_date'],
        initial_value=isr_params['initial_value'],
        stock_pct=isr_params['stock_pct'],
        tolerance=0.001,
        max_iterations=50,
        verbose=False,
        cashflows=cashflows,
        portfolio_returns=portfolio_returns,
        final_value_target=effective_final_value_target,
    )
    st.session_state['isr_value'] = float(res['spending_rate']) if res['spending_rate'] is not None else None
    st.session_state['isr_params'] = isr_params


def update_guardrail_dynamic_labels(gr_params: dict, cashflows: list):
    """
    Updates the dynamic labels on the upper and lower guardrail inputs that show the corresponding portfolio values.
    """
    shiller_df = shiller_utils.get_cached_shiller_df(st.session_state)
    stock_prices = shiller_df["Real Total Return Price"].values
    bond_prices = shiller_df["Real Total Bond Returns"].values
    portfolio_returns = utils.compute_portfolio_returns(stock_prices, bond_prices, gr_params["stock_pct"])
    effective_final_value_target = utils.get_effective_final_value_target(
        final_value_target=gr_params.get('final_value_target', 0.0),
        total_retirement_months=gr_params['duration_months'],
        months_remaining=gr_params['duration_months'],
        ramp_down=bool(gr_params.get('final_value_target_ramp_down', False)),
    )

    # First-period spending used to determine portfolio values that align with the guardrails
    first_month_spending = float(gr_params.get('initial_spending', 0.0))
    final_value_target = effective_final_value_target

    # Compute spending rates at start of retirement using retirement start date as analysis end date
    upper_res = utils.get_spending_rate_for_fixed_success_rate(
        df=shiller_df,
        desired_success_rate=gr_params['upper_sr'],
        num_months=gr_params['duration_months'],
        #analysis_start_date=gr_params['analysis_start_date'],
        initial_value=gr_params['initial_value'],
        stock_pct=gr_params['stock_pct'],
        tolerance=0.001,
        max_iterations=50,
        verbose=False,
        cashflows=cashflows,
        portfolio_returns=portfolio_returns,
        final_value_target=final_value_target,
    )

    lower_res = utils.get_spending_rate_for_fixed_success_rate(
        df=shiller_df,
        desired_success_rate=gr_params['lower_sr'],
        num_months=gr_params['duration_months'],
        #analysis_start_date=gr_params['analysis_start_date'],
        initial_value=gr_params['initial_value'],
        stock_pct=gr_params['stock_pct'],
        tolerance=0.001,
        max_iterations=50,
        verbose=False,
        cashflows=cashflows,
        portfolio_returns=portfolio_returns,
        final_value_target=final_value_target,
    )

    upper_sr = float(upper_res['spending_rate']) if upper_res['spending_rate'] is not None else None
    lower_sr = float(lower_res['spending_rate']) if lower_res['spending_rate'] is not None else None

    upper_pv = first_month_spending / upper_sr * 12 if (upper_sr is not None and upper_sr > 0) else None
    lower_pv = first_month_spending / lower_sr * 12 if (lower_sr is not None and lower_sr > 0) else None

    st.session_state[
        'upper_label_suffix'] = f" (Initial PV: ${upper_pv:,.0f})" if upper_pv is not None else " (Initial PV: N/A)"
    st.session_state[
        'lower_label_suffix'] = f" (Initial PV: ${lower_pv:,.0f})" if lower_pv is not None else " (Initial PV: N/A)"

    upper_color = None
    lower_color = None

    try:
        initial_value = float(gr_params['initial_value'])
    except (KeyError, TypeError, ValueError):
        initial_value = None

    if initial_value is not None:
        if upper_pv is not None and upper_pv < initial_value:
            upper_color = "#d62728"
        if lower_pv is not None and lower_pv > initial_value:
            lower_color = "#d62728"

    st.session_state['upper_label_color'] = upper_color
    st.session_state['lower_label_color'] = lower_color
    st.session_state['guardrail_params'] = gr_params


def update_initial_spending_label(initial_spending: float,
                                  initial_value: float) -> None:
    """Compute the dynamic label text and color for the Initial Yearly Spending control."""

    spending_rate = None
    if initial_value and initial_value > 0:
        spending_rate = float(initial_spending) / float(initial_value)

    if spending_rate is not None and np.isfinite(spending_rate):
        label_text = f"Initial Yearly Spending ({spending_rate * 100:.2f}% SR)"
    else:
        label_text = "Initial Yearly Spending (SR: N/A)"

    st.session_state['initial_spending_label_text'] = label_text
    #st.session_state['initial_spending_label_color'] = label_color


def render_simulation_results(
    results_df: pd.DataFrame,
    fixed_monthly_withdrawal: Optional[float] = None,
) -> None:
    """
    Renders charts and summaries for simulation results.
    """
    if results_df is None or results_df.empty:
        st.info("No simulation results to display.")
        return

    show_guardrail_hits = st.checkbox(
        "Show guardrail hit markers",
        value=True,
        help="Toggle vertical dotted lines at guardrail hits.",
        key="show_guardrail_hits"
    )

    init_withdrawal = float(results_df['Withdrawal'].iloc[0]) if not results_df.empty else 0.0
    yearly_scale = float(MONTHS_PER_YEAR)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.55, 0.45],
        subplot_titles=("Portfolio Value vs Guardrails", "Withdrawals Over Time")
    )

    initial_total_spending = None
    total_spending_customdata = None
    if 'Total_Spending' in results_df.columns and not results_df.empty:
        initial_total_spending = float(results_df['Total_Spending'].iloc[0])
        total_spending_diff = (results_df['Total_Spending'].astype(float) - initial_total_spending) * yearly_scale

        # Apply the formatting function to create a pre-formatted string for customdata
        formatted_total_spending_diff = total_spending_diff.apply(_fmt_currency)

        percent_denominator = initial_total_spending if initial_total_spending != 0 else np.nan
        total_spending_pct_diff = total_spending_diff / percent_denominator
        total_spending_customdata = np.column_stack([
            formatted_total_spending_diff,  # Use the pre-formatted string here as customdata[0]
            total_spending_pct_diff
        ])

    if fixed_monthly_withdrawal is not None and 'Fixed_SR_Value' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Fixed_SR_Value'],
                mode='lines',
                name='Value w/Fixed SR',
                line=dict(color='#7f7f7f'),
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
            ),
            row=1,
            col=1
        )

    if 'CAPE_Value' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['CAPE_Value'],
                mode='lines',
                name='Value w/CAPE',
                line=dict(color='#ff7f0e'),
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
            ),
            row=1,
            col=1
        )

    if 'VPW_Value' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['VPW_Value'],
                mode='lines',
                name='Value w/VPW',
                line=dict(color='#2ca02c'),
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
            ),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Upper_Guardrail'],
            mode='lines',
            name='Upper Guardrail',
            line=dict(color='#2ca02c'),
            opacity=0.45,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Lower_Guardrail'],
            mode='lines',
            name='Lower Guardrail',
            line=dict(color='#d62728'),
            opacity=0.45,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Withdrawal'] * yearly_scale,
            mode='lines',
            name='Withdrawal',
            line=dict(color='#9467bd'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}/yr<extra></extra>'
        ),
        row=2,
        col=1
    )
    if 'Net_Cashflow' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Net_Cashflow'] * yearly_scale,
                mode='lines',
                name='Net Cashflow',
                line=dict(color='#17becf', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}/yr<extra></extra>'
            ),
            row=2,
            col=1
        )
    if 'Total_Spending' in results_df.columns:
        total_spending_hovertemplate = '<b>%{fullData.name}</b>: $%{y:,.0f}/yr'
        if total_spending_customdata is not None:
            total_spending_hovertemplate += '<br>Difference: %{customdata[0]}'
            if initial_total_spending not in (None, 0):
                total_spending_hovertemplate += '<br>% Difference: %{customdata[1]:+.1%}'
        total_spending_hovertemplate += '<extra></extra>'
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Total_Spending'] * yearly_scale,
                mode='lines',
                name='Total Spending',
                line=dict(color='#bcbd22'),
                customdata=total_spending_customdata,
                hovertemplate=total_spending_hovertemplate
            ),
            row=2,
            col=1
        )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=(
                results_df['Fixed_SR_Withdrawal'] * yearly_scale
                if (fixed_monthly_withdrawal is not None and 'Fixed_SR_Withdrawal' in results_df.columns)
                else [init_withdrawal * yearly_scale] * len(results_df)
            ),
            mode='lines',
            name='Initial Withdrawal' if fixed_monthly_withdrawal is None else 'Fixed Withdrawal',
            line=dict(color='#7f7f7f', dash='dash'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}/yr<extra></extra>'
        ),
        row=2,
        col=1
    )

    if 'CAPE_Withdrawal' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['CAPE_Withdrawal'] * yearly_scale,
                mode='lines',
                name='CAPE Withdrawal',
                line=dict(color='#ff7f0e', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}/yr<extra></extra>'
            ),
            row=2,
            col=1
        )

    if 'VPW_Withdrawal' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['VPW_Withdrawal'] * yearly_scale,
                mode='lines',
                name='VPW Withdrawal',
                line=dict(color='#2ca02c', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}/yr<extra></extra>'
            ),
            row=2,
            col=1
        )

    shapes = []
    if show_guardrail_hits:
        def _guardrail_marker(date_value, color):
            return [
                dict(
                    type='line', xref='x', yref='y domain',
                    x0=date_value, x1=date_value, y0=0, y1=1,
                    line=dict(color=color, width=1, dash='dot'),
                    layer='below'
                ),
                dict(
                    type='line', xref='x', yref='y2 domain',
                    x0=date_value, x1=date_value, y0=0, y1=1,
                    line=dict(color=color, width=1, dash='dot'),
                    layer='below'
                )
            ]

        prev_hit = None
        for _, row in results_df.iterrows():
            hit = row.get('Guardrail_Hit')
            if hit == 'UPPER' and prev_hit != 'UPPER':
                shapes.extend(_guardrail_marker(row['Date'], '#2ca02c'))
            elif hit == 'LOWER' and prev_hit != 'LOWER':
                shapes.extend(_guardrail_marker(row['Date'], '#d62728'))
            prev_hit = hit

    fig.update_layout(
        shapes=shapes,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.16,
            xanchor='center',
            x=0.5,
            traceorder='reversed'
        ),
        showlegend=True,
        margin=dict(l=10, r=10, t=80, b=60),
        dragmode='zoom',
        height=900,
        xaxis=dict(rangeslider=dict(visible=False))
    )
    fig.update_annotations(
        font=dict(size=24),
        yshift=12,
        yanchor='bottom'
    )
    fig.update_xaxes(
        type='date',
        hoverformat='%b %d, %Y',
        showticklabels=True,
        row=1,
        col=1
    )
    fig.update_xaxes(
        title_text='Date',
        type='date',
        hoverformat='%b %d, %Y',
        rangeslider=dict(visible=False),
        row=2,
        col=1
    )
    fig.update_yaxes(
        title_text='Value ($)',
        tickprefix='$',
        tickformat=',.0f',
        automargin=True,
        rangemode='tozero',
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text='Withdrawals & Cashflows ($/year)',
        tickprefix='$',
        tickformat=',.0f',
        automargin=True,
        rangemode='tozero',
        row=2,
        col=1
    )

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False})

    final_summary_rows = [
        {
            "Strategy": "Guardrails",
            "Final Portfolio Value": float(results_df["Portfolio_Value"].iloc[-1]) if "Portfolio_Value" in results_df.columns and not results_df.empty else np.nan,
            "Success": bool(results_df["Guardrail_Success"].iloc[-1]) if "Guardrail_Success" in results_df.columns and not results_df.empty else False,
        },
    ]
    if "Fixed_SR_Value" in results_df.columns:
        final_summary_rows.append(
            {
                "Strategy": "Fixed Withdrawal",
                "Final Portfolio Value": float(results_df["Fixed_SR_Value"].iloc[-1]) if not results_df.empty else np.nan,
                "Success": bool(results_df["Fixed_Success"].iloc[-1]) if "Fixed_Success" in results_df.columns and not results_df.empty else False,
            }
        )
    if "CAPE_Value" in results_df.columns:
        final_summary_rows.append(
            {
                "Strategy": "CAPE Withdrawal",
                "Final Portfolio Value": float(results_df["CAPE_Value"].iloc[-1]) if not results_df.empty else np.nan,
                "Success": bool(results_df["CAPE_Success"].iloc[-1]) if "CAPE_Success" in results_df.columns and not results_df.empty else False,
            }
        )
    if "VPW_Value" in results_df.columns:
        final_summary_rows.append(
            {
                "Strategy": "VPW Withdrawal",
                "Final Portfolio Value": float(results_df["VPW_Value"].iloc[-1]) if not results_df.empty else np.nan,
                "Success": bool(results_df["VPW_Success"].iloc[-1]) if "VPW_Success" in results_df.columns and not results_df.empty else False,
            }
        )

    st.markdown("**Final portfolio values and success**")
    final_summary_df = pd.DataFrame(final_summary_rows).set_index("Strategy")
    final_summary_df["Final Portfolio Value"] = final_summary_df["Final Portfolio Value"].map(_fmt_currency)
    final_summary_df["Success"] = final_summary_df["Success"].map(lambda x: "Yes" if x else "No")
    st.table(final_summary_df)

    num_months = len(results_df)
    total_fixed = float(results_df['Fixed_SR_Withdrawal'].sum()) if 'Fixed_SR_Withdrawal' in results_df.columns else float(init_withdrawal) * num_months
    total_guardrails = float(results_df['Withdrawal'].sum())
    total_cashflow = float(results_df['Net_Cashflow'].sum()) if 'Net_Cashflow' in results_df.columns else 0.0
    total_spending_guardrails = float(results_df['Total_Spending'].sum()) if 'Total_Spending' in results_df.columns else total_guardrails + total_cashflow
    total_spending_fixed = float(results_df['Fixed_SR_Total_Spending'].sum()) if 'Fixed_SR_Total_Spending' in results_df.columns else total_fixed + total_cashflow
    avg_yearly_fixed_withdrawal = (total_fixed / num_months * yearly_scale) if num_months else None
    avg_yearly_guardrail_withdrawal = (total_guardrails / num_months * yearly_scale) if num_months else None
    avg_yearly_fixed_spending = (total_spending_fixed / num_months * yearly_scale) if num_months else None
    avg_yearly_guardrail_spending = (total_spending_guardrails / num_months * yearly_scale) if num_months else None

    has_cape = 'CAPE_Withdrawal' in results_df.columns
    has_vpw = 'VPW_Withdrawal' in results_df.columns
    if has_cape:
        total_cape_withdrawal = float(results_df['CAPE_Withdrawal'].sum())
        total_cape_spending = float(results_df['CAPE_Total_Spending'].sum())
        avg_yearly_cape_withdrawal = (total_cape_withdrawal / num_months * yearly_scale) if num_months else None
        avg_yearly_cape_spending = (total_cape_spending / num_months * yearly_scale) if num_months else None
    else:
        avg_yearly_cape_withdrawal = None
        avg_yearly_cape_spending = None

    if has_vpw:
        total_vpw_withdrawal = float(results_df['VPW_Withdrawal'].sum())
        total_vpw_spending = float(results_df['VPW_Total_Spending'].sum())
        avg_yearly_vpw_withdrawal = (total_vpw_withdrawal / num_months * yearly_scale) if num_months else None
        avg_yearly_vpw_spending = (total_vpw_spending / num_months * yearly_scale) if num_months else None
    else:
        avg_yearly_vpw_withdrawal = None
        avg_yearly_vpw_spending = None

    withdrawal_diff_ratio = (
        (avg_yearly_guardrail_withdrawal - avg_yearly_fixed_withdrawal) / avg_yearly_fixed_withdrawal
        if avg_yearly_fixed_withdrawal
        else None
    )
    spending_diff_ratio = (
        (avg_yearly_guardrail_spending - avg_yearly_fixed_spending) / avg_yearly_fixed_spending
        if avg_yearly_fixed_spending
        else None
    )

    if 'Total_Spending' in results_df.columns:
        spending_series = results_df['Total_Spending'].astype(float)
    else:
        spending_series = results_df['Withdrawal'].astype(float) if 'Withdrawal' in results_df.columns else pd.Series(dtype=float)
        if 'Net_Cashflow' in results_df.columns and not spending_series.empty:
            spending_series = spending_series + results_df['Net_Cashflow'].astype(float)

    start_spending = float(spending_series.iloc[0] * yearly_scale) if not spending_series.empty else None
    min_spending = float(spending_series.min() * yearly_scale) if not spending_series.empty else None
    max_spending = float(spending_series.max() * yearly_scale) if not spending_series.empty else None

    if has_cape:
        cape_spending_series = results_df['CAPE_Total_Spending'].astype(float)
        cape_start_spending = float(cape_spending_series.iloc[0] * yearly_scale)
        cape_min_spending = float(cape_spending_series.min() * yearly_scale)
        cape_max_spending = float(cape_spending_series.max() * yearly_scale)
    else:
        cape_start_spending = cape_min_spending = cape_max_spending = None

    if has_vpw:
        vpw_spending_series = results_df['VPW_Total_Spending'].astype(float)
        vpw_start_spending = float(vpw_spending_series.iloc[0] * yearly_scale)
        vpw_min_spending = float(vpw_spending_series.min() * yearly_scale)
        vpw_max_spending = float(vpw_spending_series.max() * yearly_scale)
    else:
        vpw_spending_series = pd.Series(dtype=float)
        vpw_start_spending = vpw_min_spending = vpw_max_spending = None

    def _longest_run(values, target):
        longest = 0
        current = 0
        for val in values:
            if np.isclose(val, target, rtol=1e-9, atol=0.5):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    min_streak = _longest_run(spending_series, min_spending) if not spending_series.empty else 0
    max_streak = _longest_run(spending_series, max_spending) if not spending_series.empty else 0
    cape_min_streak = _longest_run(cape_spending_series, cape_min_spending) if has_cape else 0
    cape_max_streak = _longest_run(cape_spending_series, cape_max_spending) if has_cape else 0
    vpw_min_streak = _longest_run(vpw_spending_series, vpw_min_spending) if has_vpw else 0
    vpw_max_streak = _longest_run(vpw_spending_series, vpw_max_spending) if has_vpw else 0

    def _fmt_pct_diff(new_value, baseline):
        if new_value is None or baseline in (None, 0):
            return "N/A"
        return f"{(new_value / baseline - 1.0):+.0%}"

    # Baseline for "Duration Below": compare against Fixed Yearly
    # Withdrawal if provided, otherwise fall back to the initial spending amount.
    baseline_comparison = (
        float(fixed_monthly_withdrawal)
        if fixed_monthly_withdrawal is not None
        else init_withdrawal
    )
    pct_below_guardrails = (results_df['Withdrawal'] < baseline_comparison).mean()
    pct_below_fixed = (results_df['Fixed_SR_Withdrawal'] < baseline_comparison).mean() if 'Fixed_SR_Withdrawal' in results_df.columns else 0.0
    pct_below_cape = (results_df['CAPE_Withdrawal'] < baseline_comparison).mean() if has_cape else None
    pct_below_vpw = (results_df['VPW_Withdrawal'] < baseline_comparison).mean() if has_vpw else None

    if fixed_monthly_withdrawal is not None:
        summary_rows = [
            {
                "Metric": "Average Annual Withdrawal",
                "Fixed": _fmt_currency(avg_yearly_fixed_withdrawal),
                "Guardrails": _fmt_currency(avg_yearly_guardrail_withdrawal),
                "% Diff": _fmt_pct_diff(avg_yearly_guardrail_withdrawal, avg_yearly_fixed_withdrawal) if withdrawal_diff_ratio is not None else "N/A",
                "CAPE": _fmt_currency(avg_yearly_cape_withdrawal) if has_cape else "N/A",
                "VPW": _fmt_currency(avg_yearly_vpw_withdrawal) if has_vpw else "N/A",
            },
            {
                "Metric": "Average Annual Spending",
                "Fixed": _fmt_currency(avg_yearly_fixed_spending),
                "Guardrails": _fmt_currency(avg_yearly_guardrail_spending),
                "% Diff": _fmt_pct_diff(avg_yearly_guardrail_spending, avg_yearly_fixed_spending) if spending_diff_ratio is not None else "N/A",
                "CAPE": _fmt_currency(avg_yearly_cape_spending) if has_cape else "N/A",
                "VPW": _fmt_currency(avg_yearly_vpw_spending) if has_vpw else "N/A",
            },
            {
                "Metric": "Duration Below",
                "Fixed": f"{pct_below_fixed:.1%}",
                "Guardrails": f"{pct_below_guardrails:.1%}",
                "% Diff": "—",
                "CAPE": f"{pct_below_cape:.1%}" if has_cape else "N/A",
                "VPW": f"{pct_below_vpw:.1%}" if has_vpw else "N/A",
            },
        ]
    else:
        summary_rows = [
            {
                "Metric": "Average Annual Withdrawal",
                "Guardrails": _fmt_currency(avg_yearly_guardrail_withdrawal),
                "CAPE": _fmt_currency(avg_yearly_cape_withdrawal) if has_cape else "N/A",
                "VPW": _fmt_currency(avg_yearly_vpw_withdrawal) if has_vpw else "N/A",
            },
            {
                "Metric": "Average Annual Spending",
                "Guardrails": _fmt_currency(avg_yearly_guardrail_spending),
                "CAPE": _fmt_currency(avg_yearly_cape_spending) if has_cape else "N/A",
                "VPW": _fmt_currency(avg_yearly_vpw_spending) if has_vpw else "N/A",
            },
            {
                "Metric": "Duration Below",
                "Guardrails": f"{pct_below_guardrails:.1%}",
                "CAPE": f"{pct_below_cape:.1%}" if has_cape else "N/A",
                "VPW": f"{pct_below_vpw:.1%}" if has_vpw else "N/A",
            },
        ]

    st.table(pd.DataFrame(summary_rows).set_index("Metric"))

    spending_rows = [
        {
            "Metric": "Start Spending",
            "Yearly": _fmt_currency(start_spending),
            "% Diff": "—",
            "Duration (months)": "—",
        },
        {
            "Metric": "Min Spending",
            "Yearly": _fmt_currency(min_spending),
            "% Diff": _fmt_pct_diff(min_spending, start_spending),
            "Duration (months)": f"{min_streak}" if min_streak else "—",
        },
        {
            "Metric": "Max Spending",
            "Yearly": _fmt_currency(max_spending),
            "% Diff": _fmt_pct_diff(max_spending, start_spending),
            "Duration (months)": f"{max_streak}" if max_streak else "—",
        },
    ]

    if has_cape:
        spending_rows.extend([
            {
                "Metric": "CAPE Start Spending",
                "Yearly": _fmt_currency(cape_start_spending),
                "% Diff": "—",
                "Duration (months)": "—",
            },
            {
                "Metric": "CAPE Min Spending",
                "Yearly": _fmt_currency(cape_min_spending),
                "% Diff": _fmt_pct_diff(cape_min_spending, cape_start_spending),
                "Duration (months)": f"{cape_min_streak}" if cape_min_streak else "—",
            },
            {
                "Metric": "CAPE Max Spending",
                "Yearly": _fmt_currency(cape_max_spending),
                "% Diff": _fmt_pct_diff(cape_max_spending, cape_start_spending),
                "Duration (months)": f"{cape_max_streak}" if cape_max_streak else "—",
            },
        ])

    if has_vpw:
        spending_rows.extend([
            {
                "Metric": "VPW Start Spending",
                "Yearly": _fmt_currency(vpw_start_spending),
                "% Diff": "—",
                "Duration (months)": "—",
            },
            {
                "Metric": "VPW Min Spending",
                "Yearly": _fmt_currency(vpw_min_spending),
                "% Diff": _fmt_pct_diff(vpw_min_spending, vpw_start_spending),
                "Duration (months)": f"{vpw_min_streak}" if vpw_min_streak else "—",
            },
            {
                "Metric": "VPW Max Spending",
                "Yearly": _fmt_currency(vpw_max_spending),
                "% Diff": _fmt_pct_diff(vpw_max_spending, vpw_start_spending),
                "Duration (months)": f"{vpw_max_streak}" if vpw_max_streak else "—",
            },
        ])

    st.table(pd.DataFrame(spending_rows).set_index("Metric"))


def _percentile_summary(series: pd.Series, label: str, is_pct: bool = False) -> dict:
    """Build a row of percentile values for a numeric series."""
    clean = series.dropna().astype(float)
    if clean.empty:
        return {
            "Metric": label,
            "10th": "N/A",
            "25th": "N/A",
            "50th": "N/A",
            "75th": "N/A",
            "90th": "N/A",
            "Mean": "N/A",
        }

    def fmt(val):
        if is_pct:
            return f"{val * 100:.1f}%"
        return _fmt_currency(val)

    return {
        "Metric": label,
        "10th": fmt(float(clean.quantile(0.10))),
        "25th": fmt(float(clean.quantile(0.25))),
        "50th": fmt(float(clean.quantile(0.50))),
        "75th": fmt(float(clean.quantile(0.75))),
        "90th": fmt(float(clean.quantile(0.90))),
        "Mean": fmt(float(clean.mean())),
    }

def render_combo_results(
        results_dict: dict[str, pd.DataFrame],
        calculation_time_seconds: Optional[float] = None,
        fixed_monthly_withdrawal: Optional[float] = None,
) -> None:
    if not results_dict:
        st.info("No combo results to display.")
        return

    yearly_scale = float(MONTHS_PER_YEAR)

    st.subheader("Combo Analysis Results")
    if calculation_time_seconds is not None:
        elapsed = max(float(calculation_time_seconds), 0.0)
        minutes, seconds = divmod(elapsed, 60.0)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            elapsed_text = f"{hours:d}h {minutes:02d}m {seconds:04.1f}s"
        elif minutes:
            elapsed_text = f"{minutes:d}m {seconds:04.1f}s"
        else:
            elapsed_text = f"{seconds:0.1f}s"
        st.caption(f"Calculation time: {elapsed_text}")

    st.caption(f"Parameter combinations: {len(results_dict)}")
    first_df = next(iter(results_dict.values()), None)
    if first_df is not None:
        st.caption(f"Number of historical retirement periods per combination: {len(first_df)}")

    rows = []
    vpw_by_sp: dict[str, list] = {}
    for key, result_df in results_dict.items():
        try:
            parts = key.split("_")
            sp = float(parts[0].replace("sp=", "").replace("%", "")) / 100.0
            ugr = float(parts[1].replace("ugr=", "").replace("%", "")) / 100.0
            lgr = float(parts[2].replace("lgr=", "").replace("%", "")) / 100.0
            tsr = float(parts[3].replace("tsr=", "").replace("%", "")) / 100.0
        except (ValueError, IndexError):
            continue

        if result_df is None or result_df.empty:
            rows.append({
                "Stock %": f"{sp:.0%}",
                "Upper GR": f"{ugr:.0%}",
                "Target SR": f"{tsr:.0%}",
                "Lower GR": f"{lgr:.0%}",
                "GR Success": "N/A",
                "N Runs": 0,
            })
            continue

        try:
            guardrail_success = float(result_df["Guardrail_Success"].mean())
        except Exception:
            guardrail_success = 0.0

        has_fixed = fixed_monthly_withdrawal is not None and "Fixed_Success" in result_df.columns
        fixed_success = float(result_df["Fixed_Success"].mean()) if has_fixed else None

        try:
            avg_sp_val = pd.to_numeric(result_df.get("avg_spending", pd.Series(dtype=float)), errors="coerce").mean()
            avg_spending = float(avg_sp_val) * yearly_scale if not pd.isna(avg_sp_val) else None
        except Exception:
            avg_spending = None

        try:
            med_sp_val = pd.to_numeric(result_df.get("median_spending", pd.Series(dtype=float)), errors="coerce").mean()
            median_spending = float(med_sp_val) * yearly_scale if not pd.isna(med_sp_val) else None
        except Exception:
            median_spending = None

        try:
            min_sp_val = pd.to_numeric(result_df.get("min_spending", pd.Series(dtype=float)), errors="coerce").mean()
            min_spending = float(min_sp_val) * yearly_scale if not pd.isna(min_sp_val) else None
        except Exception:
            min_spending = None
        try:
            max_sp_val = pd.to_numeric(result_df.get("max_spending", pd.Series(dtype=float)), errors="coerce").mean()
            max_spending = float(max_sp_val) * yearly_scale if not pd.isna(max_sp_val) else None
        except Exception:
            max_spending = None

        try:
            pct_below_val = pd.to_numeric(result_df.get("pct_below_initial", pd.Series(dtype=float)), errors="coerce").mean()
            pct_below = float(pct_below_val) if not pd.isna(pct_below_val) else None
        except Exception:
            pct_below = None

        try:
            sorted_result = result_df.sort_values(
                by=["pct_below_initial", "min_spending"],
                ascending=[False, True]
            ).reset_index(drop=True)
            N_runs = len(sorted_result)

            def _pct_at_q(col, q):
                if N_runs == 0:
                    return None
                idx = int(q * (N_runs - 1))
                val = sorted_result.iloc[idx][col]
                return float(val) if pd.notna(val) else None

            pct_below_10 = _pct_at_q("pct_below_initial", 0.10)
            pct_below_25 = _pct_at_q("pct_below_initial", 0.25)
            pct_below_50 = _pct_at_q("pct_below_initial", 0.50)
        except Exception:
            pct_below_10 = pct_below_25 = pct_below_50 = None

        # VPW metrics
        try:
            vpw_success = float(pd.to_numeric(result_df.get("VPW_Success", pd.Series(dtype=float)), errors="coerce").mean())
        except Exception:
            vpw_success = None
        try:
            vpw_avg = float(pd.to_numeric(result_df.get("VPW_Average_Withdrawal", pd.Series(dtype=float)), errors="coerce").mean())
        except Exception:
            vpw_avg = None
        try:
            vpw_median = float(pd.to_numeric(result_df.get("VPW_Median_Withdrawal", pd.Series(dtype=float)), errors="coerce").mean())
        except Exception:
            vpw_median = None
        try:
            vpw_min = float(pd.to_numeric(result_df.get("VPW_Minimum_Withdrawal", pd.Series(dtype=float)), errors="coerce").mean())
        except Exception:
            vpw_min = None
        try:
            vpw_max = float(pd.to_numeric(result_df.get("VPW_Maximum_Withdrawal", pd.Series(dtype=float)), errors="coerce").mean())
        except Exception:
            vpw_max = None
        try:
            vpw_pct_below = float(pd.to_numeric(result_df.get("VPW_Pct_Below_Fixed", pd.Series(dtype=float)), errors="coerce").mean())
        except Exception:
            vpw_pct_below = None

        try:
            vpw_pct_below_10 = _pct_at_q("VPW_Pct_Below_Fixed", 0.10)
            vpw_pct_below_25 = _pct_at_q("VPW_Pct_Below_Fixed", 0.25)
            vpw_pct_below_50 = _pct_at_q("VPW_Pct_Below_Fixed", 0.50)
        except Exception:
            vpw_pct_below_10 = vpw_pct_below_25 = vpw_pct_below_50 = None

        sp_key = f"{sp:.0%}"
        rows.append({
            "Stock %": sp_key,
            "Upper GR": f"{ugr:.0%}",
            "Target SR": f"{tsr:.0%}",
            "Lower GR": f"{lgr:.0%}",
            "GR Success": guardrail_success * 100, #f"{guardrail_success:.1%}",
            "Fixed Success": fixed_success * 100 if fixed_success is not None else np.nan, #f"{fixed_success:.1%}" if fixed_success is not None else "N/A",
            "Mean Avg": _fmt_currency(avg_spending) if avg_spending is not None else "N/A",
            "Mean Median": _fmt_currency(median_spending) if median_spending is not None else "N/A",
            "Mean Min": _fmt_currency(min_spending) if min_spending is not None else "N/A",
            "Mean Max": max_spending, #_fmt_currency(min_spending) if min_spending is not None else "N/A",

            "Below Fixed 10th": pct_below_10 * 100 if pct_below_10 is not None else np.nan, # f"{pct_below_10:04.1%}" if pct_below_10 is not None else "N/A",
            "Below Fixed 25th": pct_below_25 * 100 if pct_below_25 is not None else np.nan, #f"{pct_below_25:04.1%}" if pct_below_25 is not None else "N/A",
            "Below Fixed 50th": pct_below_50 * 100 if pct_below_50 is not None else np.nan, #f"{pct_below_50:04.1%}" if pct_below_50 is not None else "N/A",
            "Below Fixed Mean": pct_below * 100 if pct_below is not None else np.nan, #f"{pct_below:04.1%}" if pct_below is not None else "N/A",
        })

        # Accumulate by stock pct for the VPW aggregate table
        vpw_by_sp.setdefault(sp_key, []).append({
            "vpw_avg": vpw_avg,
            "vpw_median": vpw_median,
            "vpw_min": vpw_min,
            "vpw_max": vpw_max,
            "vpw_success": vpw_success,
            "vpw_pct_below": vpw_pct_below,
            "vpw_pct_below_10": vpw_pct_below_10,
            "vpw_pct_below_25": vpw_pct_below_25,
            "vpw_pct_below_50": vpw_pct_below_50,
        })

    if not rows:
        st.info("No valid combination results to display.")
        return

    summary_df = pd.DataFrame(rows)
    st.markdown("**Per-combination summary**")
    sorted_df = summary_df.sort_values(
        by=["Below Fixed 50th", "Below Fixed 25th", "Below Fixed 10th"],
        ascending=[True, True, True]
    )
    config_dict = {}
    for name in ["Mean Avg", "Mean Median", "Mean Min", "Mean Max"]:
        if name in sorted_df.columns:
            config_dict[name] = st.column_config.NumberColumn(format="dollar")
    for name in ["GR Success", "Fixed Success", "Below Fixed 10th", "Below Fixed 25th", "Below Fixed 50th", "Below Fixed Mean"]:
        if name in sorted_df.columns:
            config_dict[name] = st.column_config.NumberColumn(
                name,
                format="%.1f%%",
            )
    st.dataframe(sorted_df, use_container_width=True, hide_index=True,
                 column_config=config_dict)

    # VPW aggregate table: one row per stock percentage
    if vpw_by_sp:
        st.markdown("**VPW Summary by Stock %**")
        vpw_rows = []
        for sp_key, entries in sorted(vpw_by_sp.items(), key=lambda x: float(x[0].rstrip("%")) / 100):
            vals = [e for e in entries if e["vpw_avg"] is not None]
            if not vals:
                continue
            avg_avg = np.mean([e["vpw_avg"] for e in vals])
            avg_median = np.mean([e["vpw_median"] for e in vals])
            avg_min = np.mean([e["vpw_min"] for e in vals])
            avg_max = np.mean([e["vpw_max"] for e in vals])
            avg_success = np.mean([e["vpw_success"] for e in vals if e["vpw_success"] is not None])
            avg_pct_below = np.mean([e["vpw_pct_below"] for e in vals if e["vpw_pct_below"] is not None])
            vals_pct10 = [e["vpw_pct_below_10"] for e in vals if e["vpw_pct_below_10"] is not None]
            vals_pct25 = [e["vpw_pct_below_25"] for e in vals if e["vpw_pct_below_25"] is not None]
            vals_pct50 = [e["vpw_pct_below_50"] for e in vals if e["vpw_pct_below_50"] is not None]
            avg_pct_below_10 = np.mean(vals_pct10) if vals_pct10 else None
            avg_pct_below_25 = np.mean(vals_pct25) if vals_pct25 else None
            avg_pct_below_50 = np.mean(vals_pct50) if vals_pct50 else None
            vpw_rows.append({
                "Stock %": sp_key,
                "Avg Spend": avg_avg, #_fmt_currency(avg_avg),
                "Median Spend": avg_median, #_fmt_currency(avg_median),
                "Min Spend": avg_min, #_fmt_currency(avg_min),
                "Max Spend": avg_max, #_fmt_currency(avg_max),
                "Success Rate": avg_success * 100, #f"{avg_success:.1%}",
                "Below Fixed 10th": avg_pct_below_10 * 100 if avg_pct_below_10 is not None else np.nan, #f"{avg_pct_below_10:.1%}" if avg_pct_below_10 is not None else "N/A",
                "Below Fixed 25th": avg_pct_below_25 * 100 if avg_pct_below_25 is not None else np.nan, #f"{avg_pct_below_25:.1%}" if avg_pct_below_25 is not None else "N/A",
                "Below Fixed 50th": avg_pct_below_50 * 100 if avg_pct_below_50 is not None else np.nan, #f"{avg_pct_below_50:.1%}" if avg_pct_below_50 is not None else "N/A",
                "Below Fixed Mean": avg_pct_below * 100 if avg_pct_below is not None else np.nan, #f"{avg_pct_below:.1%}" if avg_pct_below is not None else "N/A",
            })
        if vpw_rows:
            df_to_show = pd.DataFrame(vpw_rows).set_index("Stock %")
            config_dict = {}
            for name in ["Avg Spend", "Median Spend", "Min Spend", "Max Spend"]:
                if name in df_to_show.columns:
                    config_dict[name] = st.column_config.NumberColumn(format="dollar")
            for name in ["Success Rate", "Below Fixed 10th", "Below Fixed 25th", "Below Fixed 50th", "Below Fixed Mean"]:
                if name in df_to_show.columns:
                    config_dict[name] = st.column_config.NumberColumn(
                        name,
                        format="%.1f%%",
                    )

            st.dataframe(df_to_show, use_container_width=True, column_config=config_dict)

    # Heatmap: pivot pct_below_initial over stock_pct x target_success_rate
    try:
        valid = [r for r in rows if r["Below Fixed Mean"] != "N/A"]
        stock_vals = sorted(set(r["Stock %"] for r in valid), key=lambda x: float(x.strip("%")) / 100)
        tsr_vals = sorted(set(r["Target SR"] for r in valid), key=lambda x: float(x.strip("%")) / 100)
        lookup = {(r["Stock %"], r["Target SR"]): float(r["Below Fixed Mean"].rstrip("%")) / 100.0 for r in valid}

        if len(stock_vals) > 1 and len(tsr_vals) > 1:
            st.markdown("**Duration Below by Stock % and Target Success Rate**")
            heat_data = []
            for tsr in tsr_vals:
                row_data = []
                for sp in stock_vals:
                    val = lookup.get((sp, tsr))
                    row_data.append(val if val is not None else None)
                heat_data.append(row_data)

            fig = go.Figure(data=go.Heatmap(
                z=heat_data,
                x=stock_vals,
                y=tsr_vals,
                text=[[f"{v:.1%}" if v is not None else "" for v in row] for row in heat_data],
                texttemplate="%{text}",
                colorscale="RdYlGn_r",
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="Below Initial"),
            ))
            fig.update_layout(
                title="Duration Below",
                xaxis_title="Stock %",
                yaxis_title="Target Success Rate",
                height=500,
                margin=dict(l=10, r=10, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        print(f"Error building heatmap: {e}")
        pass

def render_historical_results(
    results_df: pd.DataFrame,
    calculation_time_seconds: Optional[float] = None,
    fixed_monthly_withdrawal: Optional[float] = None,
) -> None:
    """Render summary, charts, and per-retirement table for Historical Mode."""
    if results_df is None or results_df.empty:
        st.info("No historical results to display.")
        return

    yearly_scale = float(MONTHS_PER_YEAR)
    fixed_yearly_withdrawal = monthly_to_yearly(fixed_monthly_withdrawal)

    st.subheader("Historical Mode Summary")
    st.caption(
        "The spending columns below are averaged across retirement start years. "
        "'Avg Yearly Spending' is the mean of each retirement's average yearly spending; "
        "'Median Yearly Spending' is the mean of each retirement's median yearly spending; "
        "'Min Yearly Spending' is the mean of each retirement's minimum yearly spending; "
        "'Duration Below Initial' is the mean share of months below the initial withdrawal target."
    )
    if calculation_time_seconds is not None:
        elapsed = max(float(calculation_time_seconds), 0.0)
        minutes, seconds = divmod(elapsed, 60.0)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            elapsed_text = f"{hours:d}h {minutes:02d}m {seconds:04.1f}s"
        elif minutes:
            elapsed_text = f"{minutes:d}m {seconds:04.1f}s"
        else:
            elapsed_text = f"{seconds:0.1f}s"
        st.caption(f"Calculation time: {elapsed_text}")

    total_runs = len(results_df)
    guardrail_success_rate = float(results_df["Guardrail_Success"].mean())
    has_fixed = fixed_yearly_withdrawal is not None and "Fixed_Success" in results_df.columns
    has_cape = fixed_yearly_withdrawal is not None and "CAPE_Success" in results_df.columns
    has_vpw = fixed_yearly_withdrawal is not None and "VPW_Success" in results_df.columns

    def fmt_mean_currency(col_name, scale=1.0):
        if col_name not in results_df.columns:
            return "N/A"
        series = pd.to_numeric(results_df[col_name], errors="coerce").dropna()
        if series.empty:
            return "N/A"
        return _fmt_currency(float(series.mean()) * scale)

    def fmt_mean_pct(col_name):
        if col_name not in results_df.columns:
            return "N/A"
        series = pd.to_numeric(results_df[col_name], errors="coerce").dropna()
        if series.empty:
            return "N/A"
        return f"{float(series.mean()):.1%}"

    summary_rows = [
        {
            "Strategy": "Guardrails",
            "Success Rate": f"{guardrail_success_rate:.1%}",
            "Successful Periods": f"{int(results_df['Guardrail_Success'].sum())} / {total_runs}",
            "Mean Avg Yearly Spending": fmt_mean_currency("avg_spending", yearly_scale),
            "Mean Median Yearly Spending": fmt_mean_currency("median_spending", yearly_scale),
            "Mean Min Yearly Spending": fmt_mean_currency("min_spending", yearly_scale),
            "Mean Duration Below Initial": fmt_mean_pct("pct_below_initial"),
        },
    ]

    if has_fixed:
        fixed_success_rate = float(results_df["Fixed_Success"].mean())
        summary_rows.append(
            {
                "Strategy": f"Fixed Withdrawal ({_fmt_currency(fixed_yearly_withdrawal)}/yr)",
                "Success Rate": f"{fixed_success_rate:.1%}",
                "Successful Periods": f"{int(results_df['Fixed_Success'].sum())} / {total_runs}",
                "Mean Avg Yearly Spending": _fmt_currency(fixed_yearly_withdrawal),
                "Mean Median Yearly Spending": _fmt_currency(fixed_yearly_withdrawal),
                "Mean Min Yearly Spending": _fmt_currency(fixed_yearly_withdrawal),
                "Mean Duration Below Initial": "N/A",
            }
        )
    else:
        summary_rows.append(
            {
                "Strategy": f"Fixed Withdrawal",
                "Success Rate": "N/A",
                "Successful Periods": "N/A",
                "Mean Avg Yearly Spending": "N/A",
                "Mean Median Yearly Spending": "N/A",
                "Mean Min Yearly Spending": "N/A",
                "Mean Duration Below Initial": "N/A",
            }
        )
    if has_cape:
        cape_success_rate = float(results_df["CAPE_Success"].mean())
        summary_rows.append(
            {
                "Strategy": f"CAPE Withdrawal",
                "Success Rate": f"{cape_success_rate:.1%}",
                "Successful Periods": f"{int(results_df['CAPE_Success'].sum())} / {total_runs}",
                "Mean Avg Yearly Spending": fmt_mean_currency("CAPE_Average_Withdrawal"),
                "Mean Median Yearly Spending": fmt_mean_currency("CAPE_Median_Withdrawal"),
                "Mean Min Yearly Spending": fmt_mean_currency("CAPE_Minimum_Withdrawal"),
                "Mean Duration Below Initial": fmt_mean_pct("CAPE_Pct_Below_Fixed"),
            }
        )
    else:
       summary_rows.append(
            {
                "Strategy": f"CAPE Withdrawal",
                "Success Rate": "N/A",
                "Successful Periods": "N/A",
                "Mean Avg Yearly Spending": "N/A",
                "Mean Median Yearly Spending": "N/A",
                "Mean Min Yearly Spending": "N/A",
                "Mean Duration Below Initial": "N/A",
            }
        ) 

    if has_vpw:
        vpw_success_rate = float(results_df["VPW_Success"].mean())
        summary_rows.append(
            {
                "Strategy": f"VPW Withdrawal",
                "Success Rate": f"{vpw_success_rate:.1%}",
                "Successful Periods": f"{int(results_df['VPW_Success'].sum())} / {total_runs}",
                "Mean Avg Yearly Spending": fmt_mean_currency("VPW_Average_Withdrawal"),
                "Mean Median Yearly Spending": fmt_mean_currency("VPW_Median_Withdrawal"),
                "Mean Min Yearly Spending": fmt_mean_currency("VPW_Minimum_Withdrawal"),
                "Mean Duration Below Initial": fmt_mean_pct("VPW_Pct_Below_Fixed"),
            }
        )
    else:
        summary_rows.append(
            {
                "Strategy": f"VPW Withdrawal",
                "Success Rate": "N/A",
                "Successful Periods": "N/A",
                "Mean Avg Yearly Spending": "N/A",
                "Mean Median Yearly Spending": "N/A",
                "Mean Min Yearly Spending": "N/A",
                "Mean Duration Below Initial": "N/A",
            }
        )

    st.table(pd.DataFrame(summary_rows).set_index("Strategy"))

    # Sort the runs from worst to best based on:
    # 1. pct_below_initial descending (worst outcomes have more duration below initial)
    # 2. min_spending ascending (worst outcomes have lower minimum yearly spending)
    sorted_df = results_df.sort_values(
        by=["pct_below_initial", "min_spending"],
        ascending=[False, True]
    ).reset_index(drop=True)

    N = len(sorted_df)
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]

    def get_val_at_quantile(col_name, q):
        if N == 0:
            return np.nan
        idx = int(q * (N - 1))
        return sorted_df.loc[idx, col_name]

    def get_flipped_pct_below_at_quantile(col_name, q):
        if N == 0:
            return np.nan
        idx = int(q * (N - 1))
        sum = sorted_df.loc[:idx, col_name].sum()
        return (idx - sum + 1) / N
    def get_flipped_pct_at_quantile(col_name, q):
        if N == 0:
            return np.nan
        idx = int(q * (N - 1))
        sum = sorted_df.loc[:idx, col_name].sum()
        denom = idx + 1
        if denom <= 0:
            return np.nan
        return (idx - sum + 1) / denom

    def get_mean(col_name):
        return results_df[col_name].mean() if not results_df.empty else np.nan

    def set_col(row, col_key, val, is_pct=False, is_bool=False, is_pct_below_flipped=False, is_pct_flipped=False, scale=1.0):
        if pd.isna(val) or val is None:
            row[col_key] = "N/A"
        elif is_bool:
            row[col_key] = "Yes" if val else "No"
        elif is_pct_below_flipped or is_pct_flipped or is_pct:
            row[col_key] = f"{val * scale * 100:.1f}%"
        else:
            row[col_key] = _fmt_currency(float(val) * scale)


    def build_percentile_row(col_name, label, is_pct=False, is_bool=False, is_pct_flipped=False, is_pct_below_flipped=False, scale=1.0):
        row = {"Metric": label}
        for q in quantiles:
            if is_pct_below_flipped:
                val = get_flipped_pct_below_at_quantile(col_name, q)
            elif is_pct_flipped:
                val = get_flipped_pct_at_quantile(col_name, q)
            else:
                val = get_val_at_quantile(col_name, q)
            col_key = f"{int(q*100)}th"
            set_col(row, col_key, val, is_pct, is_bool, is_pct_below_flipped, is_pct_flipped, scale)

        if is_pct_below_flipped or is_pct_flipped:
            mean_val = 1.0 - get_mean(col_name)
        else:
            mean_val = get_mean(col_name)
        set_col(row, "Mean", mean_val, is_pct, is_bool, is_pct_below_flipped, is_pct_flipped, scale)
        return row

    percentile_rows = [
        build_percentile_row("min_spending", "Min Yearly Spending", scale=yearly_scale),
        build_percentile_row("max_spending", "Max Yearly Spending", scale=yearly_scale),
        build_percentile_row("median_spending", "Median Yearly Spending", scale=yearly_scale),
        build_percentile_row("avg_spending", "Average Yearly Spending", scale=yearly_scale),
        build_percentile_row("pct_below_initial", "Duration Below", is_pct=True),
    ]

    if has_cape:
        percentile_rows.extend([
            build_percentile_row("CAPE_Minimum_Withdrawal", "CAPE Min Yearly Spending"),
            build_percentile_row("CAPE_Maximum_Withdrawal", "CAPE Max Yearly Spending"),
            build_percentile_row("CAPE_Median_Withdrawal", "CAPE Median Yearly Spending"),
            build_percentile_row("CAPE_Average_Withdrawal", "CAPE Average Yearly Spending"),
            build_percentile_row("CAPE_Pct_Below_Fixed", "CAPE Duration Below", is_pct=True),
        ])
    if has_vpw:
        percentile_rows.extend([
            build_percentile_row("VPW_Minimum_Withdrawal", "VPW Min Yearly Spending"),
            build_percentile_row("VPW_Maximum_Withdrawal", "VPW Max Yearly Spending"),
            build_percentile_row("VPW_Median_Withdrawal", "VPW Median Yearly Spending"),
            build_percentile_row("VPW_Average_Withdrawal", "VPW Average Yearly Spending"),
            build_percentile_row("VPW_Pct_Below_Fixed", "VPW Duration Below", is_pct=True),
        ])

    percentile_rows.append(build_percentile_row("Guardrail_Success", "Guardrail Failure %", is_pct_flipped=True))
    if has_fixed:
        percentile_rows.append(build_percentile_row("Fixed_Success", "Fixed Withdrawal Failure %", is_pct_flipped=True))
    if has_cape:
        percentile_rows.append(build_percentile_row("CAPE_Success", "CAPE Withdrawal Failure %", is_pct_flipped=True))
    if has_vpw:
        percentile_rows.append(build_percentile_row("VPW_Success", "VPW Withdrawal Failure %", is_pct_flipped=True))
 
    st.markdown("**Guardrail spending percentiles across historical retirements**")
    st.table(pd.DataFrame(percentile_rows).set_index("Metric"))

    st.subheader("Charts")

    fig_success = go.Figure()
    fig_success.add_trace(
        go.Scatter(
            x=results_df["Start_Year"],
            y=results_df["Guardrail_Success"].astype(int) * 100,
            mode="lines+markers",
            name="Guardrails Success",
            line=dict(color="#1f77b4"),
            hovertemplate="Start: %{x}<br>Success: %{y:.0f}%<extra></extra>",
        )
    )
    if has_fixed:
        fig_success.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df["Fixed_Success"].astype(int) * 100,
                mode="lines+markers",
                name="Fixed Withdrawal Success",
                line=dict(color="#7f7f7f", dash="dash"),
                hovertemplate="Start: %{x}<br>Success: %{y:.0f}%<extra></extra>",
            )
        )
    if has_cape:
        fig_success.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df["CAPE_Success"].astype(int) * 100,
                mode="lines+markers",
                name="CAPE Withdrawal Success",
                line=dict(color="#1f77b4", dash="dash"),
                hovertemplate="Start: %{x}<br>Success: %{y:.0f}%<extra></extra>",
            )
        )
    if has_vpw:
        fig_success.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df["VPW_Success"].astype(int) * 100,
                mode="lines+markers",
                name="VPW Withdrawal Success",
                line=dict(color="#2ca02c", dash="dash"),
                hovertemplate="Start: %{x}<br>Success: %{y:.0f}%<extra></extra>",
            )
        )
    fig_success.update_layout(
        title="Success by Retirement Start Year",
        xaxis_title="Retirement Start Year",
        yaxis_title="Success (100 = success, 0 = failure)",
        yaxis=dict(tickvals=[0, 100], ticktext=["Fail", "Guardrail_Success"]),
        hovermode="x unified",
        height=400,
        margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig_success, use_container_width=True)

    fig_duration = go.Figure()
    fig_duration.add_trace(
        go.Scatter(
            x=results_df["Start_Year"],
            y=results_df["pct_below_initial"] * 100,
            mode="lines+markers",
            name="Duration Below Initial",
            line=dict(color="#d62728"),
            hovertemplate="Start: %{x}<br>Duration Below Initial: %{y:.1f}%<extra></extra>",
        )
    )
    if has_cape and "CAPE_Pct_Below_Fixed" in results_df.columns:
        fig_duration.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df["CAPE_Pct_Below_Fixed"] * 100,
                mode="lines+markers",
                name="CAPE Duration Below Fixed",
                line=dict(color="#1f77b4", dash="dash"),
                hovertemplate="Start: %{x}<br>Duration Below Fixed: %{y:.1f}%<extra></extra>",
            )
        )
    if has_vpw and "VPW_Pct_Below_Fixed" in results_df.columns:
        fig_duration.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df["VPW_Pct_Below_Fixed"] * 100,
                mode="lines+markers",
                name="VPW Duration Below Fixed",
                line=dict(color="#2ca02c", dash="dash"),
                hovertemplate="Start: %{x}<br>Duration Below Fixed: %{y:.1f}%<extra></extra>",
            )
        )
    fig_duration.update_layout(
        title="Percentage of Duration Below by Start Year",
        xaxis_title="Retirement Start Year",
        yaxis_title="Duration Below Initial (%)",
        yaxis=dict(ticksuffix="%"),
        hovermode="x unified",
        height=400,
        margin=dict(l=10, r=10, t=60, b=40),
    )
    st.plotly_chart(fig_duration, use_container_width=True)

    fig_spending = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Min Yearly Spending",
            "Max Yearly Spending",
            "Median Yearly Spending",
            "Average Yearly Spending",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    spending_specs = [
        ("min_spending", "CAPE_Minimum_Withdrawal", "VPW_Minimum_Withdrawal", 1, 1, "#d62728"),
        ("max_spending", "CAPE_Maximum_Withdrawal", "VPW_Maximum_Withdrawal", 1, 2, "#2ca02c"),
        ("median_spending", "CAPE_Median_Withdrawal", "VPW_Median_Withdrawal", 2, 1, "#9467bd"),
        ("avg_spending", "CAPE_Average_Withdrawal", "VPW_Average_Withdrawal", 2, 2, "#1f77b4"),
    ]
    for column, cape_column, vpw_column, row, col, color in spending_specs:
        fig_spending.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df[column] * yearly_scale,
                mode="lines",
                name=f"Guardrails {column.replace('_', ' ').title()}",
                line=dict(color=color),
                hovertemplate="Start: %{x}<br>$%{y:,.0f}/yr<extra></extra>",
                showlegend=True,
            ),
            row=row,
            col=col,
        )
        if has_cape and cape_column in results_df.columns:
            fig_spending.add_trace(
                go.Scatter(
                    x=results_df["Start_Year"],
                    y=results_df[cape_column],
                    mode="lines",
                    name=f"CAPE {column.replace('_', ' ').title()}",
                    line=dict(color=color, dash="dash"),
                    hovertemplate="Start: %{x}<br>$%{y:,.0f}/yr<extra></extra>",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
        if has_vpw and vpw_column in results_df.columns:
            fig_spending.add_trace(
                go.Scatter(
                    x=results_df["Start_Year"],
                    y=results_df[vpw_column],
                    mode="lines",
                    name=f"VPW {column.replace('_', ' ').title()}",
                    line=dict(color=color, dash="dot"),
                    hovertemplate="Start: %{x}<br>$%{y:,.0f}/yr<extra></extra>",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
    fig_spending.update_layout(
        height=700,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    for i in range(1, 5):
        fig_spending.update_yaxes(tickprefix="$", tickformat=",.0f", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
    fig_spending.update_xaxes(title_text="Retirement Start Year", row=2, col=1)
    fig_spending.update_xaxes(title_text="Retirement Start Year", row=2, col=2)
    st.plotly_chart(fig_spending, use_container_width=True)

    st.subheader("Per-Retirement Results")
    display_df = results_df.copy()
    display_df["Guardrail_Success"] = display_df["Guardrail_Success"].map({True: "Yes", False: "No"})
    if "Fixed_Success" in display_df.columns:
        display_df["Fixed_Success"] = display_df["Fixed_Success"].map(
            lambda x: "Yes" if x is True else ("No" if x is False else "")
        )
    if "CAPE_Success" in display_df.columns:
        display_df["CAPE_Success"] = display_df["CAPE_Success"].map(
            lambda x: "Yes" if x is True else ("No" if x is False else "")
        )
    if "VPW_Success" in display_df.columns:
        display_df["VPW_Success"] = display_df["VPW_Success"].map(
            lambda x: "Yes" if x is True else ("No" if x is False else "")
        )

    currency_cols = [
        "Ending_Portfolio",
        "min_spending",
        "max_spending",
        "median_spending",
        "avg_spending",
        "Fixed_Ending_Portfolio",
        "CAPE_Ending_Portfolio",
        "CAPE_Minimum_Withdrawal",
        "CAPE_Maximum_Withdrawal",
        "CAPE_Average_Withdrawal",
        "CAPE_Median_Withdrawal",
        "VPW_Ending_Portfolio",
        "VPW_Minimum_Withdrawal",
        "VPW_Maximum_Withdrawal",
        "VPW_Average_Withdrawal",
        "VPW_Median_Withdrawal",
    ]
    spending_display_cols = {
        "min_spending",
        "max_spending",
        "median_spending",
        "avg_spending",
    }
    for col in currency_cols:
        if col in display_df.columns:
            if col in spending_display_cols:
                display_df[col] = display_df[col].astype(float) * yearly_scale
            else:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce")

    if "pct_below_initial" in display_df.columns:
        display_df["pct_below_initial"] = pd.to_numeric(display_df["pct_below_initial"], errors="coerce") * 100.0
    if "CAPE_Pct_Below_Fixed" in display_df.columns:
        display_df["CAPE_Pct_Below_Fixed"] = pd.to_numeric(display_df["CAPE_Pct_Below_Fixed"], errors="coerce") * 100.0
    if "VPW_Pct_Below_Fixed" in display_df.columns:
        display_df["VPW_Pct_Below_Fixed"] = pd.to_numeric(display_df["VPW_Pct_Below_Fixed"], errors="coerce") * 100.0

    if "Start_Date" in display_df.columns:
        display_df["Start_Date"] = pd.to_datetime(display_df["Start_Date"]).dt.strftime("%Y-%m")
    if "End_Date" in display_df.columns:
        display_df["End_Date"] = pd.to_datetime(display_df["End_Date"]).dt.strftime("%Y-%m")

    column_rename = {
        "Start_Year": "Start Year",
        "Start_Date": "Start Date",
        "End_Date": "End Date",
        "Guardrail_Success": "GR Success",
        "Fixed_Success": "Fixed Success",
        "CAPE_Success": "CAPE Success",
        "Ending_Portfolio": "GR Ending Portfolio",
        "min_spending": "GR Min Withdrawal",
        "max_spending": "GR Max Withdrawal",
        "median_spending": "GR Median Withdrawal",
        "avg_spending": "GR Avg Withdrawal",
        "pct_below_initial": "GR Below Fixed (%)",
        "Fixed_Ending_Portfolio": "Fixed Ending Portfolio",
        "CAPE_Ending_Portfolio": "CAPE Ending Portfolio",
        "CAPE_Minimum_Withdrawal": "CAPE Minimum Withdrawal",
        "CAPE_Maximum_Withdrawal": "CAPE Maximum Withdrawal",
        "CAPE_Average_Withdrawal": "CAPE Average Withdrawal",
        "CAPE_Median_Withdrawal": "CAPE Median Withdrawal",
        "CAPE_Pct_Below_Fixed": "CAPE Below Fixed (%)",
        "VPW_Success": "VPW Success",
        "VPW_Ending_Portfolio": "VPW Ending Portfolio",
        "VPW_Minimum_Withdrawal": "VPW Minimum Withdrawal",
        "VPW_Maximum_Withdrawal": "VPW Maximum Withdrawal",
        "VPW_Average_Withdrawal": "VPW Average Withdrawal",
        "VPW_Median_Withdrawal": "VPW Median Withdrawal",
        "VPW_Pct_Below_Fixed": "VPW Below Fixed (%)",
    }
    display_df = display_df.rename(columns=column_rename)
    column_config = { "Start Date": None }
    for name in [
        "GR Ending Portfolio",
        "GR Min Withdrawal",
        "GR Max Withdrawal",
        "GR Median Withdrawal",
        "GR Avg Withdrawal",
        "Fixed Ending Portfolio",
        "CAPE Ending Portfolio",
        "CAPE Minimum Withdrawal",
        "CAPE Maximum Withdrawal",
        "CAPE Average Withdrawal",
        "CAPE Median Withdrawal",
        "VPW Ending Portfolio",
        "VPW Minimum Withdrawal",
        "VPW Maximum Withdrawal",
        "VPW Average Withdrawal",
        "VPW Median Withdrawal",
    ]:
        if name in display_df.columns:
            column_config[name] = st.column_config.NumberColumn(format="dollar") #format="$%.0f")

    for name in ["GR Below Fixed (%)", "CAPE Below Fixed (%)", "VPW Below Fixed (%)"]:
        if name in display_df.columns:
            column_config[name] = st.column_config.NumberColumn(format="%.1f%%")



    pinned_order = ["Start Year", "End Date", "GR Success", "Fixed Success", "CAPE Success",
                    "VPW Success", "GR Below Fixed (%)", "CAPE Below Fixed (%)", "VPW Below Fixed (%)"]
    # 2. Automatically grab all other columns that aren't in your pinned list
    remaining_cols = [col for col in display_df.columns if col not in pinned_order]
    # 3. Combine them together into one complete list
    final_column_order = pinned_order + remaining_cols
    st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config, column_order=final_column_order)


def render_guidance_results(snap: dict):
    """Summarize the guardrail guidance snapshot as advisor-friendly bullet points."""

    def fmt_money(x):
        return _fmt_currency(x, escape_for_markdown=True)

    def fmt_pct(p):
        return f"{p * 100:+.0f}%" if p is not None else "N/A"

    # Get spending rate (rate at which spending is calculated from portfolio)
    start_sr = st.session_state.get("isr_value", snap.get("target_spending_rate"))
    # Get actual withdrawal rate (true portfolio withdrawal rate after cashflows)
    target_wr = snap.get("target_withdrawal_rate")

    upper_val = snap.get("upper_guardrail_value")
    lower_val = snap.get("lower_guardrail_value")
    up_adj_pct = snap.get("upper_adjustment_pct")
    low_adj_pct = snap.get("lower_adjustment_pct")

    start_yearly_spending = monthly_to_yearly(snap.get("target_monthly_spending"))
    start_yearly_withdrawal = monthly_to_yearly(snap.get("target_monthly_withdrawal"))
    up_adj_yearly = monthly_to_yearly(snap.get("upper_adjusted_monthly"))
    low_adj_yearly = monthly_to_yearly(snap.get("lower_adjusted_monthly"))
    cashflow_yearly = monthly_to_yearly(snap.get("current_cashflow"))
    annual_cashflow_total = snap.get("annual_cashflow_total")

    cape_sr = snap.get("cape_spending_rate")
    cape_yearly_spending = monthly_to_yearly(snap.get("cape_monthly_spending"))
    cape_yearly_withdrawal = monthly_to_yearly(snap.get("cape_monthly_withdrawal"))
    cape_annual_withdrawal = snap.get("cape_annual_withdrawal")
    cape_wr = snap.get("cape_withdrawal_rate")

    st.subheader("Guidance Mode")
    st.markdown(
        "Use this mode to generate forward-looking guidance for a client who is retired today and in drawdown.\n\n"
        "All dollar amounts shown are in real (constant) dollars, net of inflation. For more details, see the [documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")

    # Notes for spending row shows spending rate
    spending_notes = (
        f"{start_sr * 100:.2f}% spending rate" if start_sr is not None else "Based on Initial Portfolio Value"
    )

    # Notes for withdrawal row shows actual withdrawal rate
    withdrawal_notes = (
        f"{target_wr * 100:.2f}% withdrawal rate" if target_wr is not None else "Net amount after cashflows"
    )

    summary_rows = [
        {
            "Metric": "Target Spending",
            "Yearly": fmt_money(start_yearly_spending),
            "Notes": spending_notes,
        },
        {
            "Metric": "Portfolio Withdrawal After Cashflows",
            "Yearly": fmt_money(start_yearly_withdrawal),
            "Notes": withdrawal_notes,
        },
        {
            "Metric": "Current Cashflow",
            "Yearly": fmt_money(cashflow_yearly),
            "Notes": "First month, shown as yearly equivalent",
        },
    ]

    if annual_cashflow_total is not None:
        summary_rows.append(
            {
                "Metric": "First-Year Cashflow Total",
                "Yearly": fmt_money(annual_cashflow_total),
                "Notes": "Sum of configured cashflows over the first 12 months",
            }
        )

    if cape_sr is not None:
        cape_spending_notes = f"{cape_sr * 100:.2f}% CAPE-based spending rate"
        cape_withdrawal_notes = (
            f"{cape_wr * 100:.2f}% CAPE-based withdrawal rate" if cape_wr is not None else "Net amount after cashflows"
        )
        summary_rows.append(
            {
                "Metric": "CAPE Target Spending",
                "Yearly": fmt_money(cape_yearly_spending),
                "Notes": cape_spending_notes,
            }
        )
        summary_rows.append(
            {
                "Metric": "CAPE Portfolio Withdrawal",
                "Yearly": fmt_money(cape_yearly_withdrawal),
                "Notes": cape_withdrawal_notes,
            }
        )
        if cape_annual_withdrawal is not None:
            summary_rows.append(
                {
                    "Metric": "CAPE First-Year Withdrawal",
                    "Yearly": fmt_money(cape_annual_withdrawal),
                    "Notes": "Sum of CAPE withdrawals over the first 12 months",
                }
            )

    st.table(pd.DataFrame(summary_rows).set_index("Metric"))

    guardrail_rows = [
        {
            "Guardrail": "Upper",
            "Trigger (Portfolio Value)": fmt_money(upper_val),
            "Adjustment": fmt_pct(up_adj_pct),
            "New Spending (Yearly)": fmt_money(up_adj_yearly),
        },
        {
            "Guardrail": "Lower",
            "Trigger (Portfolio Value)": fmt_money(lower_val),
            "Adjustment": fmt_pct(low_adj_pct),
            "New Spending (Yearly)": fmt_money(low_adj_yearly),
        },
    ]

    st.table(pd.DataFrame(guardrail_rows).set_index("Guardrail"))
