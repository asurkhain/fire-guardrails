import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from typing import Optional, Tuple
import shiller_utils
import utils
from tax_utils import summarize_tax_metrics_from_results

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


def render_simulation_results(
    results_df: pd.DataFrame,
    fixed_monthly_withdrawal: Optional[float] = None,
    current_age: Optional[float] = None,
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

    initial_portfolio_value = (
        float(results_df['Portfolio_Value'].iloc[0] + results_df['Withdrawal'].iloc[0])
        if not results_df.empty else None
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.40, 0.30, 0.30],
        subplot_titles=("Portfolio Value vs Guardrails", "Spend Rates Over Time", "Withdrawal Rates Over Time")
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
                line=dict(color='#e377c2'),
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
            name='Guardrail Portfolio Value',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=1,
        col=1
    )

    if 'Net_Cashflow' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Net_Cashflow'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='Net Cashflow Rate',
                line=dict(color='#17becf', dash='dash'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=2,
            col=1
        )
    if 'Total_Spending' in results_df.columns and initial_portfolio_value:
        total_spending_hovertemplate = '<b>%{fullData.name}</b>: %{y:.2%}'
        total_spending_hovertemplate += '<extra></extra>'
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Total_Spending'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='Guardrails Spend Rate',
                line=dict(color='#1f77b4', dash='dash'),
                hovertemplate=total_spending_hovertemplate
            ),
            row=2,
            col=1
        )
    if fixed_monthly_withdrawal is not None and 'Fixed_SR_Total_Spending' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Fixed_SR_Total_Spending'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='Fixed Spend Rate',
                line=dict(color='#7f7f7f', dash='dash'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=2,
            col=1
        )
    if 'CAPE_Total_Spending' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['CAPE_Total_Spending'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='CAPE Spend Rate',
                line=dict(color='#ff7f0e', dash='dash'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=2,
            col=1
        )
    if 'VPW_Total_Spending' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['VPW_Total_Spending'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='VPW Spend Rate',
                line=dict(color='#e377c2', dash='dash'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=2,
            col=1
        )
    guardrail_wd = (
        results_df['Tax_Inclusive_Withdrawal']
        if 'Tax_Inclusive_Withdrawal' in results_df.columns
        else results_df['Withdrawal']
    )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=guardrail_wd * yearly_scale / initial_portfolio_value,
            mode='lines',
            name='Guardrail W/D Rate',
            line=dict(color='#1f77b4', dash='dot'),
            hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
        ),
        row=3,
        col=1
    )
    if initial_portfolio_value:
        if 'Fixed_Tax_Inclusive_Withdrawal' in results_df.columns:
            fixed_wd_series = results_df['Fixed_Tax_Inclusive_Withdrawal'] * yearly_scale / initial_portfolio_value
        elif 'Fixed_SR_Withdrawal' in results_df.columns:
            fixed_wd_series = results_df['Fixed_SR_Withdrawal'] * yearly_scale / initial_portfolio_value
        else:
            first_wd = float(guardrail_wd.iloc[0]) if not results_df.empty else init_withdrawal
            fixed_wd_series = [first_wd * yearly_scale / initial_portfolio_value] * len(results_df)

        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=fixed_wd_series,
                mode='lines',
                name='Initial W/D Rate' if fixed_monthly_withdrawal is None else 'Fixed W/D Rate',
                line=dict(color='#7f7f7f', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=3,
            col=1
        )

    if 'CAPE_Tax_Inclusive_Withdrawal' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['CAPE_Tax_Inclusive_Withdrawal'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='CAPE W/D Rate',
                line=dict(color='#ff7f0e', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=3,
            col=1
        )
    elif 'CAPE_Withdrawal' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['CAPE_Withdrawal'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='CAPE W/D Rate',
                line=dict(color='#ff7f0e', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=3,
            col=1
        )

    if 'VPW_Tax_Inclusive_Withdrawal' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['VPW_Tax_Inclusive_Withdrawal'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='VPW W/D Rate',
                line=dict(color='#e377c2', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=3,
            col=1
        )
    elif 'VPW_Withdrawal' in results_df.columns and initial_portfolio_value:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['VPW_Withdrawal'] * yearly_scale / initial_portfolio_value,
                mode='lines',
                name='VPW W/D Rate',
                line=dict(color='#e377c2', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>'
            ),
            row=3,
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
                ),
                dict(
                    type='line', xref='x', yref='y3 domain',
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
        height=1100,
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
        type='date',
        hoverformat='%b %d, %Y',
        showticklabels=True,
        row=2,
        col=1
    )
    fig.update_xaxes(
        title_text='Date',
        type='date',
        hoverformat='%b %d, %Y',
        showticklabels=True,
        row=3,
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
        title_text='Spending Rate',
        title_font=dict(size=14),
        tickformat='.2%',
        tickfont=dict(size=12),
        automargin=True,
        rangemode='tozero',
        row=2,
        col=1
    )
    fig.update_yaxes(
        title_text='Rate',
        title_font=dict(size=14),
        tickformat='.2%',
        tickfont=dict(size=12),
        automargin=True,
        rangemode='tozero',
        row=3,
        col=1
    )

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False})

    if "Tax_Rate" in results_df.columns:
        strategy_summaries = _collect_strategy_tax_summaries(
            results_df,
            initial_portfolio=initial_portfolio_value,
            current_age=current_age,
        )
        if strategy_summaries:
            st.markdown("**Lifetime Tax & Efficiency**")
            st.table(
                _build_strategy_tax_metric_table(
                    strategy_summaries,
                    _SIM_LIFETIME_TAX_METRICS,
                )
            )

            st.markdown("**ACA & Healthcare Diagnostics**")
            st.table(
                _build_strategy_tax_metric_table(
                    strategy_summaries,
                    _SIM_ACA_TAX_METRICS,
                )
            )

    if "Tax_Rate" in results_df.columns:
        fig_tax_rate = go.Figure()
        tax_rate_strategies = (
            ("Guardrails", "Tax_Rate", "Tax_Paid", "#1f77b4"),
            ("Fixed", "Fixed_Tax_Rate", "Fixed_Tax_Paid", "#7f7f7f"),
            ("CAPE", "CAPE_Tax_Rate", "CAPE_Tax_Paid", "#ff7f0e"),
            ("VPW", "VPW_Tax_Rate", "VPW_Tax_Paid", "#e377c2"),
        )
        for name, rate_column, tax_column, color in tax_rate_strategies:
            if rate_column not in results_df.columns:
                continue
            annual_tax_data = results_df[["Date", rate_column, tax_column]].copy()
            annual_tax_data["Year"] = pd.to_datetime(annual_tax_data["Date"]).dt.year
            withdrawal_column = rate_column.replace("Tax_Rate", "Tax_Inclusive_Withdrawal")
            annual_tax_data["Tax_Inclusive_Withdrawal"] = pd.to_numeric(
                results_df.get(withdrawal_column, 0.0), errors="coerce"
            ).fillna(0.0)
            annual_tax_data[tax_column] = pd.to_numeric(annual_tax_data[tax_column], errors="coerce").fillna(0.0)
            annual_tax_data = annual_tax_data.groupby("Year", as_index=False).agg(
                {tax_column: "sum", "Tax_Inclusive_Withdrawal": "sum"}
            )
            annual_tax_data["Tax_Rate"] = np.where(
                annual_tax_data["Tax_Inclusive_Withdrawal"] > 0.0,
                annual_tax_data[tax_column] / annual_tax_data["Tax_Inclusive_Withdrawal"],
                0.0,
            )
            fig_tax_rate.add_trace(
                go.Scatter(
                    x=annual_tax_data["Year"],
                    y=annual_tax_data["Tax_Rate"],
                    mode="lines",
                    name=name,
                    line=dict(color=color),
                    customdata=np.column_stack([
                        annual_tax_data[tax_column],
                        annual_tax_data["Tax_Inclusive_Withdrawal"],
                    ]),
                    hovertemplate=(
                        f"<b>{name}</b><br>Year: %{{x}}<br>Tax rate: %{{y:.2%}}"
                        "<br>Taxes paid: $%{customdata[0]:,.0f}"
                        "<br>Tax-inclusive withdrawal: $%{customdata[1]:,.0f}<extra></extra>"
                    ),
                )
            )
        if "Taxable_Accounts_Balance" in results_df.columns:
            balances = pd.to_numeric(results_df["Taxable_Accounts_Balance"], errors="coerce").fillna(0.0)
            if float(balances.iloc[0]) > 1e-9:
                depleted_mask = balances <= 1e-9
                if depleted_mask.any():
                    depleted_year = int(pd.to_datetime(results_df.loc[depleted_mask, "Date"]).iloc[0].year)
                    fig_tax_rate.add_vline(
                        x=depleted_year,
                        line_dash="dash",
                        line_color="#9467bd",
                        annotation_text="Taxable Accounts Depleted",
                        annotation_position="top left",
                    )
        fig_tax_rate.update_layout(
            title="Tax Rate Over Time",
            xaxis_title="Year",
            yaxis_title="Taxes Paid / Tax-Inclusive Withdrawal",
            yaxis=dict(tickformat=".2%", rangemode="tozero"),
            hovermode="x unified",
            height=360,
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig_tax_rate, use_container_width=True, config={"scrollZoom": False})

        fig_aca = go.Figure()
        aca_strategies = (
            ("Guardrails", "ACA_Surcharge", "#1f77b4"),
            ("Fixed", "Fixed_ACA_Surcharge", "#7f7f7f"),
            ("CAPE", "CAPE_ACA_Surcharge", "#ff7f0e"),
            ("VPW", "VPW_ACA_Surcharge", "#e377c2"),
        )
        for name, surcharge_column, color in aca_strategies:
            if surcharge_column not in results_df.columns:
                continue
            annual_aca = pd.DataFrame({
                "Year": pd.to_datetime(results_df["Date"]).dt.year,
                "ACA_Surcharge": pd.to_numeric(results_df[surcharge_column], errors="coerce").fillna(0.0),
            }).groupby("Year", as_index=False)["ACA_Surcharge"].sum()
            fig_aca.add_trace(
                go.Scatter(
                    x=annual_aca["Year"],
                    y=annual_aca["ACA_Surcharge"],
                    mode="lines",
                    name=name,
                    line=dict(color=color),
                    hovertemplate=(
                        f"<b>{name}</b><br>Year: %{{x}}<br>ACA surcharge: $%{{y:,.0f}}<extra></extra>"
                    ),
                )
            )
        medicare_year = None
        if current_age is not None and "Date" in results_df.columns:
            start_year = int(pd.to_datetime(results_df["Date"]).iloc[0].year)
            medicare_year = start_year + max(0, 65 - int(current_age))
        if medicare_year is not None:
            fig_aca.add_vline(
                x=medicare_year,
                line_dash="dash",
                line_color="#444444",
                annotation_text="Age 65 (Medicare)",
                annotation_position="top right",
            )
        fig_aca.update_layout(
            title="ACA Surcharge Paid by Year",
            xaxis_title="Year",
            yaxis_title="ACA Surcharge Paid ($)",
            yaxis=dict(tickprefix="$", tickformat=",.0f", rangemode="tozero"),
            hovermode="x unified",
            height=360,
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig_aca, use_container_width=True, config={"scrollZoom": False})

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
    guardrail_wd_col = 'Tax_Inclusive_Withdrawal' if 'Tax_Inclusive_Withdrawal' in results_df.columns else 'Withdrawal'
    fixed_wd_col = 'Fixed_Tax_Inclusive_Withdrawal' if 'Fixed_Tax_Inclusive_Withdrawal' in results_df.columns else 'Fixed_SR_Withdrawal'
    cape_wd_col = 'CAPE_Tax_Inclusive_Withdrawal' if 'CAPE_Tax_Inclusive_Withdrawal' in results_df.columns else 'CAPE_Withdrawal'
    vpw_wd_col = 'VPW_Tax_Inclusive_Withdrawal' if 'VPW_Tax_Inclusive_Withdrawal' in results_df.columns else 'VPW_Withdrawal'

    def _get_active_df(df: pd.DataFrame, success_col: str) -> pd.DataFrame:
        if success_col in df.columns:
            succ = df[success_col].astype(bool)
            if not succ.all():
                first_fail = succ.idxmin()
                return df.loc[:first_fail]
        return df

    gr_df = _get_active_df(results_df, 'Guardrail_Success')
    fixed_df = _get_active_df(results_df, 'Fixed_Success')
    cape_df = _get_active_df(results_df, 'CAPE_Success')
    vpw_df = _get_active_df(results_df, 'VPW_Success')

    gr_wd = gr_df[guardrail_wd_col].astype(float) if guardrail_wd_col in gr_df.columns else gr_df['Withdrawal'].astype(float)
    gr_sp = gr_df['Total_Spending'].astype(float) if 'Total_Spending' in gr_df.columns else gr_wd

    avg_yearly_guardrail_withdrawal = float(gr_wd.mean() * yearly_scale) if not gr_wd.empty else None
    avg_yearly_guardrail_spending = float(gr_sp.mean() * yearly_scale) if not gr_sp.empty else None
    median_yearly_guardrail_withdrawal = float(gr_wd.median() * yearly_scale) if not gr_wd.empty else None
    median_yearly_guardrail_spending = float(gr_sp.median() * yearly_scale) if not gr_sp.empty else None

    if fixed_wd_col in fixed_df.columns:
        fixed_wd = fixed_df[fixed_wd_col].astype(float)
    elif 'Fixed_SR_Withdrawal' in fixed_df.columns:
        fixed_wd = fixed_df['Fixed_SR_Withdrawal'].astype(float)
    else:
        fixed_wd = pd.Series([init_withdrawal] * len(fixed_df))

    if 'Fixed_SR_Total_Spending' in fixed_df.columns:
        fixed_sp = fixed_df['Fixed_SR_Total_Spending'].astype(float)
    else:
        fixed_sp = fixed_wd

    avg_yearly_fixed_withdrawal = float(fixed_wd.mean() * yearly_scale) if not fixed_wd.empty else None
    avg_yearly_fixed_spending = float(fixed_sp.mean() * yearly_scale) if not fixed_sp.empty else None
    median_yearly_fixed_withdrawal = float(fixed_wd.median() * yearly_scale) if not fixed_wd.empty else None
    median_yearly_fixed_spending = float(fixed_sp.median() * yearly_scale) if not fixed_sp.empty else None

    has_cape = cape_wd_col in results_df.columns or 'CAPE_Withdrawal' in results_df.columns
    has_vpw = vpw_wd_col in results_df.columns or 'VPW_Withdrawal' in results_df.columns

    if has_cape:
        cape_wd = cape_df[cape_wd_col].astype(float) if cape_wd_col in cape_df.columns else cape_df['CAPE_Withdrawal'].astype(float)
        cape_sp = cape_df['CAPE_Total_Spending'].astype(float) if 'CAPE_Total_Spending' in cape_df.columns else cape_wd
        avg_yearly_cape_withdrawal = float(cape_wd.mean() * yearly_scale) if not cape_wd.empty else None
        avg_yearly_cape_spending = float(cape_sp.mean() * yearly_scale) if not cape_sp.empty else None
        median_yearly_cape_withdrawal = float(cape_wd.median() * yearly_scale) if not cape_wd.empty else None
        median_yearly_cape_spending = float(cape_sp.median() * yearly_scale) if not cape_sp.empty else None
    else:
        avg_yearly_cape_withdrawal = avg_yearly_cape_spending = median_yearly_cape_withdrawal = median_yearly_cape_spending = None

    if has_vpw:
        vpw_wd = vpw_df[vpw_wd_col].astype(float) if vpw_wd_col in vpw_df.columns else vpw_df['VPW_Withdrawal'].astype(float)
        vpw_sp = vpw_df['VPW_Total_Spending'].astype(float) if 'VPW_Total_Spending' in vpw_df.columns else vpw_wd
        avg_yearly_vpw_withdrawal = float(vpw_wd.mean() * yearly_scale) if not vpw_wd.empty else None
        avg_yearly_vpw_spending = float(vpw_sp.mean() * yearly_scale) if not vpw_sp.empty else None
        median_yearly_vpw_withdrawal = float(vpw_wd.median() * yearly_scale) if not vpw_wd.empty else None
        median_yearly_vpw_spending = float(vpw_sp.median() * yearly_scale) if not vpw_sp.empty else None
    else:
        avg_yearly_vpw_withdrawal = avg_yearly_vpw_spending = median_yearly_vpw_withdrawal = median_yearly_vpw_spending = None

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

    start_spending = float(gr_sp.iloc[0] * yearly_scale) if not gr_sp.empty else None
    median_spending = float(gr_sp.median() * yearly_scale) if not gr_sp.empty else None
    min_spending = float(gr_sp.min() * yearly_scale) if not gr_sp.empty else None
    max_spending = float(gr_sp.max() * yearly_scale) if not gr_sp.empty else None

    start_wd = float(gr_wd.iloc[0] * yearly_scale) if not gr_wd.empty else None
    median_wd = float(gr_wd.median() * yearly_scale) if not gr_wd.empty else None
    min_wd = float(gr_wd.min() * yearly_scale) if not gr_wd.empty else None
    max_wd = float(gr_wd.max() * yearly_scale) if not gr_wd.empty else None

    if has_cape:
        cape_start_spending = float(cape_sp.iloc[0] * yearly_scale) if not cape_sp.empty else None
        cape_median_spending = float(cape_sp.median() * yearly_scale) if not cape_sp.empty else None
        cape_min_spending = float(cape_sp.min() * yearly_scale) if not cape_sp.empty else None
        cape_max_spending = float(cape_sp.max() * yearly_scale) if not cape_sp.empty else None

        cape_start_wd = float(cape_wd.iloc[0] * yearly_scale) if not cape_wd.empty else None
        cape_median_wd = float(cape_wd.median() * yearly_scale) if not cape_wd.empty else None
        cape_min_wd = float(cape_wd.min() * yearly_scale) if not cape_wd.empty else None
        cape_max_wd = float(cape_wd.max() * yearly_scale) if not cape_wd.empty else None
    else:
        cape_start_spending = cape_median_spending = cape_min_spending = cape_max_spending = None
        cape_start_wd = cape_median_wd = cape_min_wd = cape_max_wd = None

    if has_vpw:
        vpw_start_spending = float(vpw_sp.iloc[0] * yearly_scale) if not vpw_sp.empty else None
        vpw_median_spending = float(vpw_sp.median() * yearly_scale) if not vpw_sp.empty else None
        vpw_min_spending = float(vpw_sp.min() * yearly_scale) if not vpw_sp.empty else None
        vpw_max_spending = float(vpw_sp.max() * yearly_scale) if not vpw_sp.empty else None

        vpw_start_wd = float(vpw_wd.iloc[0] * yearly_scale) if not vpw_wd.empty else None
        vpw_median_wd = float(vpw_wd.median() * yearly_scale) if not vpw_wd.empty else None
        vpw_min_wd = float(vpw_wd.min() * yearly_scale) if not vpw_wd.empty else None
        vpw_max_wd = float(vpw_wd.max() * yearly_scale) if not vpw_wd.empty else None
    else:
        vpw_start_spending = vpw_median_spending = vpw_min_spending = vpw_max_spending = None
        vpw_start_wd = vpw_median_wd = vpw_min_wd = vpw_max_wd = None

    def _longest_run(values, target):
        if values is None or values.empty or target is None:
            return 0
        longest = 0
        current = 0
        for val in values:
            if np.isclose(val, target, rtol=1e-9, atol=0.5):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    monthly_min = float(gr_sp.min()) if not gr_sp.empty else None
    monthly_med = float(gr_sp.median()) if not gr_sp.empty else None
    monthly_max = float(gr_sp.max()) if not gr_sp.empty else None
    min_streak = _longest_run(gr_sp, monthly_min)
    med_streak = _longest_run(gr_sp, monthly_med)
    max_streak = _longest_run(gr_sp, monthly_max)
    cape_min_streak = _longest_run(cape_sp, float(cape_sp.min())) if has_cape and not cape_sp.empty else 0
    cape_med_streak = _longest_run(cape_sp, float(cape_sp.median())) if has_cape and not cape_sp.empty else 0
    cape_max_streak = _longest_run(cape_sp, float(cape_sp.max())) if has_cape and not cape_sp.empty else 0
    vpw_min_streak = _longest_run(vpw_sp, float(vpw_sp.min())) if has_vpw and not vpw_sp.empty else 0
    vpw_med_streak = _longest_run(vpw_sp, float(vpw_sp.median())) if has_vpw and not vpw_sp.empty else 0
    vpw_max_streak = _longest_run(vpw_sp, float(vpw_sp.max())) if has_vpw and not vpw_sp.empty else 0

    def _fmt_pct_diff(new_value, baseline):
        if new_value is None or baseline in (None, 0):
            return "N/A"
        return f"{(new_value / baseline - 1.0):+.0%}"

    def _fmt_rate(value, base):
        if value is None or base in (None, 0):
            return "N/A"
        return f"{value / base:.2%}"

    baseline_comparison = (
        float(fixed_monthly_withdrawal)
        if fixed_monthly_withdrawal is not None
        else init_withdrawal
    )
    pct_below_guardrails = (results_df['Total_Spending'] < baseline_comparison).mean()
    pct_below_cape = (results_df['CAPE_Total_Spending'] < baseline_comparison).mean() if has_cape else None
    pct_below_vpw = (results_df['VPW_Total_Spending'] < baseline_comparison).mean() if has_vpw else None

    if fixed_monthly_withdrawal is not None:
        summary_rows = [
            {
                "Metric": "Avg Annual Withdrawal Rate",
                "Fixed": _fmt_rate(avg_yearly_fixed_withdrawal, initial_portfolio_value),
                "Guardrails": _fmt_rate(avg_yearly_guardrail_withdrawal, initial_portfolio_value),
                "CAPE": _fmt_rate(avg_yearly_cape_withdrawal, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(avg_yearly_vpw_withdrawal, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Median Annual Withdrawal Rate",
                "Fixed": _fmt_rate(median_yearly_fixed_withdrawal, initial_portfolio_value),
                "Guardrails": _fmt_rate(median_yearly_guardrail_withdrawal, initial_portfolio_value),
                "CAPE": _fmt_rate(median_yearly_cape_withdrawal, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(median_yearly_vpw_withdrawal, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Avg Annual Spending Rate",
                "Fixed": _fmt_rate(avg_yearly_fixed_spending, initial_portfolio_value),
                "Guardrails": _fmt_rate(avg_yearly_guardrail_spending, initial_portfolio_value),
                "CAPE": _fmt_rate(avg_yearly_cape_spending, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(avg_yearly_vpw_spending, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Median Annual Spending Rate",
                "Fixed": _fmt_rate(median_yearly_fixed_spending, initial_portfolio_value),
                "Guardrails": _fmt_rate(median_yearly_guardrail_spending, initial_portfolio_value),
                "CAPE": _fmt_rate(median_yearly_cape_spending, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(median_yearly_vpw_spending, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Duration Below",
                "Fixed": "N/A",
                "Guardrails": f"{pct_below_guardrails:.1%}",
                "CAPE": f"{pct_below_cape:.1%}" if has_cape else "N/A",
                "VPW": f"{pct_below_vpw:.1%}" if has_vpw else "N/A",
            },
        ]
    else:
        summary_rows = [
            {
                "Metric": "Avg Annual Withdrawal Rate",
                "Guardrails": _fmt_rate(avg_yearly_guardrail_withdrawal, initial_portfolio_value),
                "CAPE": _fmt_rate(avg_yearly_cape_withdrawal, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(avg_yearly_vpw_withdrawal, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Median Annual Withdrawal Rate",
                "Guardrails": _fmt_rate(median_yearly_guardrail_withdrawal, initial_portfolio_value),
                "CAPE": _fmt_rate(median_yearly_cape_withdrawal, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(median_yearly_vpw_withdrawal, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Avg Annual Spending Rate",
                "Guardrails": _fmt_rate(avg_yearly_guardrail_spending, initial_portfolio_value),
                "CAPE": _fmt_rate(avg_yearly_cape_spending, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(avg_yearly_vpw_spending, initial_portfolio_value) if has_vpw else "N/A",
            },
            {
                "Metric": "Median Annual Spending Rate",
                "Guardrails": _fmt_rate(median_yearly_guardrail_spending, initial_portfolio_value),
                "CAPE": _fmt_rate(median_yearly_cape_spending, initial_portfolio_value) if has_cape else "N/A",
                "VPW": _fmt_rate(median_yearly_vpw_spending, initial_portfolio_value) if has_vpw else "N/A",
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
            "Metric": "Guardrails Start Spending",
            "Yearly": _fmt_currency(start_spending),
            "Spend Rate": _fmt_rate(start_spending, initial_portfolio_value),
            "Withdrawal Rate": _fmt_rate(start_wd, initial_portfolio_value),
            "Duration (months)": "—",
        },
        {
            "Metric": "Guardrails Min Spending",
            "Yearly": _fmt_currency(min_spending),
            "Spend Rate": _fmt_rate(min_spending, initial_portfolio_value),
            "Withdrawal Rate": _fmt_rate(min_wd, initial_portfolio_value),
            "Duration (months)": f"{min_streak}" if min_streak else "—",
        },
        {
            "Metric": "Guardrails Median Spending",
            "Yearly": _fmt_currency(median_spending),
            "Spend Rate": _fmt_rate(median_spending, initial_portfolio_value),
            "Withdrawal Rate": _fmt_rate(median_wd, initial_portfolio_value),
            "Duration (months)": f"{med_streak}" if med_streak else "—",
        },
        {
            "Metric": "Guardrails Max Spending",
            "Yearly": _fmt_currency(max_spending),
            "Spend Rate": _fmt_rate(max_spending, initial_portfolio_value),
            "Withdrawal Rate": _fmt_rate(max_wd, initial_portfolio_value),
            "Duration (months)": f"{max_streak}" if max_streak else "—",
        },
    ]

    if has_cape:
        spending_rows.extend([
            {
                "Metric": "CAPE Start Spending",
                "Yearly": _fmt_currency(cape_start_spending),
                "Spend Rate": _fmt_rate(cape_start_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(cape_start_wd, initial_portfolio_value),
                "Duration (months)": "—",
            },
            {
                "Metric": "CAPE Min Spending",
                "Yearly": _fmt_currency(cape_min_spending),
                "Spend Rate": _fmt_rate(cape_min_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(cape_min_wd, initial_portfolio_value),
                "Duration (months)": f"{cape_min_streak}" if cape_min_streak else "—",
            },
            {
                "Metric": "CAPE Median Spending",
                "Yearly": _fmt_currency(cape_median_spending),
                "Spend Rate": _fmt_rate(cape_median_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(cape_median_wd, initial_portfolio_value),
                "Duration (months)": f"{cape_med_streak}" if cape_med_streak else "—",
            },
            {
                "Metric": "CAPE Max Spending",
                "Yearly": _fmt_currency(cape_max_spending),
                "Spend Rate": _fmt_rate(cape_max_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(cape_max_wd, initial_portfolio_value),
                "Duration (months)": f"{cape_max_streak}" if cape_max_streak else "—",
            },
        ])

    if has_vpw:
        spending_rows.extend([
            {
                "Metric": "VPW Start Spending",
                "Yearly": _fmt_currency(vpw_start_spending),
                "Spend Rate": _fmt_rate(vpw_start_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(vpw_start_wd, initial_portfolio_value),
                "Duration (months)": "—",
            },
            {
                "Metric": "VPW Min Spending",
                "Yearly": _fmt_currency(vpw_min_spending),
                "Spend Rate": _fmt_rate(vpw_min_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(vpw_min_wd, initial_portfolio_value),
                "Duration (months)": f"{vpw_min_streak}" if vpw_min_streak else "—",
            },
            {
                "Metric": "VPW Median Spending",
                "Yearly": _fmt_currency(vpw_median_spending),
                "Spend Rate": _fmt_rate(vpw_median_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(vpw_median_wd, initial_portfolio_value),
                "Duration (months)": f"{vpw_med_streak}" if vpw_med_streak else "—",
            },
            {
                "Metric": "VPW Max Spending",
                "Yearly": _fmt_currency(vpw_max_spending),
                "Spend Rate": _fmt_rate(vpw_max_spending, initial_portfolio_value),
                "Withdrawal Rate": _fmt_rate(vpw_max_wd, initial_portfolio_value),
                "Duration (months)": f"{vpw_max_streak}" if vpw_max_streak else "—",
            },
        ])

    st.table(pd.DataFrame(spending_rows).set_index("Metric"))


_HISTORICAL_TAX_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]

_HISTORICAL_TAX_METRICS = (
    ("Lifetime_Tax_Rate", "Lifetime Effective Tax Rate", "pct"),
    ("Pre65_Lifetime_Tax_Rate", "Pre-65 Effective Tax Rate", "pct"),
    ("Avg_Annual_Tax_Pct", "Avg Annual Tax (% of Starting Portfolio)", "pct"),
    ("Pre65_Avg_Annual_Tax_Pct", "Avg Pre-65 Annual Tax (% of Starting Portfolio)", "pct"),
    ("Pre65_Subsidized_Years", "Pre-65 Subsidized Years", "years"),
    ("Pre65_Cliff_Years", "Pre-65 Cliff Years", "years"),
)


def _fmt_historical_tax_percentile_value(
    value,
    metric_kind: str,
) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if metric_kind == "pct":
        return f"{float(value):.1%}"
    if metric_kind == "years":
        return f"{float(value):.1f}"
    return _fmt_currency(float(value))


def _build_historical_tax_percentile_row(
    sorted_df: pd.DataFrame,
    results_df: pd.DataFrame,
    col_name: str,
    label: str,
    metric_kind: str = "pct",
) -> dict:
    """Build one tax metric row using the shared historical cohort ordering."""
    row = {"Metric": label}
    quantile_cols = [f"{int(q * 100)}th" for q in _HISTORICAL_TAX_QUANTILES]
    n_runs = len(sorted_df)

    if col_name not in sorted_df.columns or n_runs == 0:
        for col_key in quantile_cols:
            row[col_key] = "N/A"
        row["Mean"] = "N/A"
        return row

    for q, col_key in zip(_HISTORICAL_TAX_QUANTILES, quantile_cols):
        idx = int(q * (n_runs - 1))
        row[col_key] = _fmt_historical_tax_percentile_value(
            sorted_df.loc[idx, col_name],
            metric_kind,
        )

    if col_name in results_df.columns:
        mean_series = pd.to_numeric(results_df[col_name], errors="coerce").dropna()
        mean_val = float(mean_series.mean()) if not mean_series.empty else np.nan
    else:
        mean_val = np.nan
    row["Mean"] = _fmt_historical_tax_percentile_value(mean_val, metric_kind)
    return row


_SIM_TAX_STRATEGIES = (
    ("", "Guardrails"),
    ("Fixed_", "Fixed"),
    ("CAPE_", "CAPE"),
    ("VPW_", "VPW"),
)

_SIM_LIFETIME_TAX_METRICS = (
    ("Lifetime_Tax_Rate", "Lifetime Effective Tax Rate", "pct"),
    ("Pre65_Lifetime_Tax_Rate", "Pre-65 Effective Tax Rate", "pct"),
    ("Avg_Annual_Tax_Pct", "Avg Annual Tax (% of Initial Portfolio)", "pct2"),
    ("Pre65_Avg_Annual_Tax_Pct", "Avg Pre-65 Annual Tax (% of Initial Portfolio)", "pct2"),
)

_SIM_ACA_TAX_METRICS = (
    ("Pre65_Subsidized_Years", "Pre-65 Subsidized Years", "int"),
    ("Pre65_Cliff_Years", "Pre-65 Cliff Years", "int"),
    ("Medicare_Transition_Year", "Medicare Transition Year", "year"),
)


def _fmt_sim_tax_metric(value, metric_kind: str) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if metric_kind == "pct":
        return f"{float(value):.1%}"
    if metric_kind == "pct2":
        return f"{float(value):.2%}"
    if metric_kind in ("int", "year"):
        return str(int(value))
    return str(value)


def _collect_strategy_tax_summaries(
    results_df: pd.DataFrame,
    initial_portfolio: float | None,
    current_age: float | None,
) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for prefix, strategy_label in _SIM_TAX_STRATEGIES:
        tax_col = f"{prefix}Tax_Paid" if prefix else "Tax_Paid"
        if tax_col not in results_df.columns:
            continue
        summary = summarize_tax_metrics_from_results(
            results_df,
            prefix=prefix,
            initial_portfolio=initial_portfolio,
            current_age=current_age,
        )
        if summary:
            summaries[strategy_label] = summary
    return summaries


def _build_strategy_tax_metric_table(
    summaries: dict[str, dict],
    metric_defs: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    rows = []
    strategy_labels = [label for _, label in _SIM_TAX_STRATEGIES]
    for metric_key, metric_label, metric_kind in metric_defs:
        row = {"Metric": metric_label}
        for strategy_label in strategy_labels:
            summary = summaries.get(strategy_label)
            if summary is None:
                row[strategy_label] = "N/A"
            else:
                row[strategy_label] = _fmt_sim_tax_metric(
                    summary.get(metric_key),
                    metric_kind,
                )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Metric")


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
    fixed_yearly_withdrawal_including_taxes: Optional[float] = None,
) -> None:
    """Render summary, charts, and per-retirement table for Historical Mode."""
    if results_df is None or results_df.empty:
        st.info("No historical results to display.")
        return

    yearly_scale = float(MONTHS_PER_YEAR)
    fixed_yearly_withdrawal = monthly_to_yearly(fixed_monthly_withdrawal)
    effective_fixed_yearly_withdrawal = (
        fixed_yearly_withdrawal_including_taxes
        if fixed_yearly_withdrawal_including_taxes is not None
        else fixed_yearly_withdrawal
    )

    st.subheader("Historical Mode Summary")
    st.caption(
        "Spend rates below are annualized and divided by the initial portfolio value. "
        "'Avg Spend Rate' is the mean of each retirement's average yearly spend rate; "
        "'Median Spend Rate' is the mean of each retirement's median yearly spend rate; "
        "'Min Spend Rate' is the mean of each retirement's minimum yearly spend rate; "
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

    starting_portfolio = float(
        pd.to_numeric(results_df["Starting_Portfolio"], errors="coerce").iloc[0]
    ) if "Starting_Portfolio" in results_df.columns else None

    def fmt_mean_rate(col_name, scale=1.0):
        if col_name not in results_df.columns or starting_portfolio is None or starting_portfolio == 0:
            return "N/A"
        series = pd.to_numeric(results_df[col_name], errors="coerce").dropna()
        if series.empty:
            return "N/A"
        return f"{float(series.mean() * scale / starting_portfolio):.2%}"

    summary_rows = [
        {
            "Strategy": "Guardrails",
            "Success Rate": f"{guardrail_success_rate:.1%}",
            "Successful Periods": f"{int(results_df['Guardrail_Success'].sum())} / {total_runs}",
            "Mean Avg Spend Rate": fmt_mean_rate("avg_spending", yearly_scale),
            "Mean Median Spend Rate": fmt_mean_rate("median_spending", yearly_scale),
            "Mean Min Spend Rate": fmt_mean_rate("min_spending", yearly_scale),
            "Mean Median Withdrawal Rate": fmt_mean_rate("median_withdrawal", yearly_scale),
            "Mean Duration Below Initial": fmt_mean_pct("pct_below_initial"),
        },
    ]

    if has_fixed:
        fixed_success_rate = float(results_df["Fixed_Success"].mean())
        fixed_spend_rate = (
            f"{fixed_yearly_withdrawal / starting_portfolio:.2%}"
            if starting_portfolio and fixed_yearly_withdrawal
            else "N/A"
        )

        # fmt_mean_rate("Fixed_Median_Actual_Withdrawal"") if "Fixed_Median_Actual_Withdrawal" in results_df.columns else fixed_withdrawal_rate,
        # results_df["Fixed_Median_Actual_Withdrawal"] if "Fixed_Median_Actual_Withdrawal" in results_df.columns else effective_fixed_yearly_withdrawal
        # build_percentile_row("Fixed_Median_WD_Rate", "Fixed — Median Withdrawal Rate", is_pct=True),
        fixed_withdrawal_rate = (
            f"{effective_fixed_yearly_withdrawal / starting_portfolio:.2%}"
            if starting_portfolio and effective_fixed_yearly_withdrawal
            else "N/A"
        )
        summary_rows.append(
            {
                "Strategy": f"Fixed Withdrawal",
                "Success Rate": f"{fixed_success_rate:.1%}",
                "Successful Periods": f"{int(results_df['Fixed_Success'].sum())} / {total_runs}",
                "Mean Avg Spend Rate": fixed_spend_rate,
                "Mean Median Spend Rate": fixed_spend_rate,
                "Mean Min Spend Rate": fixed_spend_rate,
                "Mean Median Withdrawal Rate": fmt_mean_rate("Fixed_Median_Actual_Withdrawal") if "Fixed_Median_Actual_Withdrawal" in results_df.columns else fixed_withdrawal_rate,
                "Mean Duration Below Initial": "N/A",
            }
        )
    else:
        summary_rows.append(
            {
                "Strategy": f"Fixed Withdrawal",
                "Success Rate": "N/A",
                "Successful Periods": "N/A",
                "Mean Avg Spend Rate": "N/A",
                "Mean Median Spend Rate": "N/A",
                "Mean Min Spend Rate": "N/A",
                "Mean Median Withdrawal Rate": "N/A",
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
                "Mean Avg Spend Rate": fmt_mean_rate("CAPE_Average_Withdrawal"),
                "Mean Median Spend Rate": fmt_mean_rate("CAPE_Median_Withdrawal"),
                "Mean Min Spend Rate": fmt_mean_rate("CAPE_Minimum_Withdrawal"),
                "Mean Median Withdrawal Rate": fmt_mean_rate("CAPE_Median_Actual_Withdrawal") if "CAPE_Median_Actual_Withdrawal" in results_df.columns else fmt_mean_rate("CAPE_Median_Withdrawal"),
                "Mean Duration Below Initial": fmt_mean_pct("CAPE_Pct_Below_Fixed"),
            }
        )
    else:
       summary_rows.append(
            {
                "Strategy": f"CAPE Withdrawal",
                "Success Rate": "N/A",
                "Successful Periods": "N/A",
                "Mean Avg Spend Rate": "N/A",
                "Mean Median Spend Rate": "N/A",
                "Mean Min Spend Rate": "N/A",
                "Mean Median Withdrawal Rate": "N/A",
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
                "Mean Avg Spend Rate": fmt_mean_rate("VPW_Average_Withdrawal"),
                "Mean Median Spend Rate": fmt_mean_rate("VPW_Median_Withdrawal"),
                "Mean Min Spend Rate": fmt_mean_rate("VPW_Minimum_Withdrawal"),
                "Mean Median Withdrawal Rate": fmt_mean_rate("VPW_Median_Actual_Withdrawal") if "VPW_Median_Actual_Withdrawal" in results_df.columns else fmt_mean_rate("VPW_Median_Withdrawal"),
                "Mean Duration Below Initial": fmt_mean_pct("VPW_Pct_Below_Fixed"),
            }
        )
    else:
        summary_rows.append(
            {
                "Strategy": f"VPW Withdrawal",
                "Success Rate": "N/A",
                "Successful Periods": "N/A",
                "Mean Avg Spend Rate": "N/A",
                "Mean Median Spend Rate": "N/A",
                "Mean Min Spend Rate": "N/A",
                "Mean Median Withdrawal Rate": "N/A",
                "Mean Duration Below Initial": "N/A",
            }
        )

    st.table(pd.DataFrame(summary_rows).set_index("Strategy"))

    # Compute rate columns on results_df first so both get_val_at_quantile
    # (which uses sorted_df) and get_mean (which uses results_df) can find them.
    if starting_portfolio:
        results_df["min_spending_rate"] = results_df["min_spending"].astype(float) * yearly_scale / starting_portfolio
        results_df["max_spending_rate"] = results_df["max_spending"].astype(float) * yearly_scale / starting_portfolio
        results_df["median_spending_rate"] = results_df["median_spending"].astype(float) * yearly_scale / starting_portfolio
        results_df["avg_spending_rate"] = results_df["avg_spending"].astype(float) * yearly_scale / starting_portfolio

        results_df["min_withdrawal_rate"] = pd.to_numeric(results_df.get("min_withdrawal"), errors="coerce") * yearly_scale / starting_portfolio
        results_df["max_withdrawal_rate"] = pd.to_numeric(results_df.get("max_withdrawal"), errors="coerce") * yearly_scale / starting_portfolio
        results_df["median_withdrawal_rate"] = pd.to_numeric(results_df.get("median_withdrawal"), errors="coerce") * yearly_scale / starting_portfolio
        results_df["avg_withdrawal_rate"] = pd.to_numeric(results_df.get("avg_withdrawal"), errors="coerce") * yearly_scale / starting_portfolio

        if has_fixed:
            fixed_act_min = results_df["Fixed_Min_Actual_Withdrawal"] if "Fixed_Min_Actual_Withdrawal" in results_df.columns else effective_fixed_yearly_withdrawal
            fixed_act_max = results_df["Fixed_Max_Actual_Withdrawal"] if "Fixed_Max_Actual_Withdrawal" in results_df.columns else effective_fixed_yearly_withdrawal
            fixed_act_med = results_df["Fixed_Median_Actual_Withdrawal"] if "Fixed_Median_Actual_Withdrawal" in results_df.columns else effective_fixed_yearly_withdrawal
            fixed_act_avg = results_df["Fixed_Average_Actual_Withdrawal"] if "Fixed_Average_Actual_Withdrawal" in results_df.columns else effective_fixed_yearly_withdrawal

            results_df["Fixed_Min_WD_Rate"] = pd.to_numeric(fixed_act_min, errors="coerce") / starting_portfolio
            results_df["Fixed_Max_WD_Rate"] = pd.to_numeric(fixed_act_max, errors="coerce") / starting_portfolio
            results_df["Fixed_Median_WD_Rate"] = pd.to_numeric(fixed_act_med, errors="coerce") / starting_portfolio
            results_df["Fixed_Avg_WD_Rate"] = pd.to_numeric(fixed_act_avg, errors="coerce") / starting_portfolio
            results_df["Fixed_Spend_Rate"] = pd.to_numeric(fixed_yearly_withdrawal, errors="coerce") / starting_portfolio

        if has_cape:
            results_df["CAPE_Min_Spend_Rate"] = pd.to_numeric(results_df["CAPE_Minimum_Withdrawal"], errors="coerce") / starting_portfolio
            results_df["CAPE_Max_Spend_Rate"] = pd.to_numeric(results_df["CAPE_Maximum_Withdrawal"], errors="coerce") / starting_portfolio
            results_df["CAPE_Median_Spend_Rate"] = pd.to_numeric(results_df["CAPE_Median_Withdrawal"], errors="coerce") / starting_portfolio
            results_df["CAPE_Avg_Spend_Rate"] = pd.to_numeric(results_df["CAPE_Average_Withdrawal"], errors="coerce") / starting_portfolio

            cape_act_min = results_df["CAPE_Min_Actual_Withdrawal"] if "CAPE_Min_Actual_Withdrawal" in results_df.columns else results_df["CAPE_Minimum_Withdrawal"]
            cape_act_max = results_df["CAPE_Max_Actual_Withdrawal"] if "CAPE_Max_Actual_Withdrawal" in results_df.columns else results_df["CAPE_Maximum_Withdrawal"]
            cape_act_med = results_df["CAPE_Median_Actual_Withdrawal"] if "CAPE_Median_Actual_Withdrawal" in results_df.columns else results_df["CAPE_Median_Withdrawal"]
            cape_act_avg = results_df["CAPE_Average_Actual_Withdrawal"] if "CAPE_Average_Actual_Withdrawal" in results_df.columns else results_df["CAPE_Average_Withdrawal"]

            results_df["CAPE_Min_WD_Rate"] = pd.to_numeric(cape_act_min, errors="coerce") / starting_portfolio
            results_df["CAPE_Max_WD_Rate"] = pd.to_numeric(cape_act_max, errors="coerce") / starting_portfolio
            results_df["CAPE_Median_WD_Rate"] = pd.to_numeric(cape_act_med, errors="coerce") / starting_portfolio
            results_df["CAPE_Avg_WD_Rate"] = pd.to_numeric(cape_act_avg, errors="coerce") / starting_portfolio

        if has_vpw:
            results_df["VPW_Min_Spend_Rate"] = pd.to_numeric(results_df["VPW_Minimum_Withdrawal"], errors="coerce") / starting_portfolio
            results_df["VPW_Max_Spend_Rate"] = pd.to_numeric(results_df["VPW_Maximum_Withdrawal"], errors="coerce") / starting_portfolio
            results_df["VPW_Median_Spend_Rate"] = pd.to_numeric(results_df["VPW_Median_Withdrawal"], errors="coerce") / starting_portfolio
            results_df["VPW_Avg_Spend_Rate"] = pd.to_numeric(results_df["VPW_Average_Withdrawal"], errors="coerce") / starting_portfolio

            vpw_act_min = results_df["VPW_Min_Actual_Withdrawal"] if "VPW_Min_Actual_Withdrawal" in results_df.columns else results_df["VPW_Minimum_Withdrawal"]
            vpw_act_max = results_df["VPW_Max_Actual_Withdrawal"] if "VPW_Max_Actual_Withdrawal" in results_df.columns else results_df["VPW_Maximum_Withdrawal"]
            vpw_act_med = results_df["VPW_Median_Actual_Withdrawal"] if "VPW_Median_Actual_Withdrawal" in results_df.columns else results_df["VPW_Median_Withdrawal"]
            vpw_act_avg = results_df["VPW_Average_Actual_Withdrawal"] if "VPW_Average_Actual_Withdrawal" in results_df.columns else results_df["VPW_Average_Withdrawal"]

            results_df["VPW_Min_WD_Rate"] = pd.to_numeric(vpw_act_min, errors="coerce") / starting_portfolio
            results_df["VPW_Max_WD_Rate"] = pd.to_numeric(vpw_act_max, errors="coerce") / starting_portfolio
            results_df["VPW_Median_WD_Rate"] = pd.to_numeric(vpw_act_med, errors="coerce") / starting_portfolio
            results_df["VPW_Avg_WD_Rate"] = pd.to_numeric(vpw_act_avg, errors="coerce") / starting_portfolio

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
        build_percentile_row("min_spending_rate", "Guardrails — Min Spend Rate", is_pct=True),
        build_percentile_row("max_spending_rate", "Guardrails — Max Spend Rate", is_pct=True),
        build_percentile_row("median_spending_rate", "Guardrails — Median Spend Rate", is_pct=True),
        build_percentile_row("avg_spending_rate", "Guardrails — Average Spend Rate", is_pct=True),
        build_percentile_row("min_withdrawal_rate", "Guardrails — Min Withdrawal Rate", is_pct=True),
        build_percentile_row("max_withdrawal_rate", "Guardrails — Max Withdrawal Rate", is_pct=True),
        build_percentile_row("median_withdrawal_rate", "Guardrails — Median Withdrawal Rate", is_pct=True),
        build_percentile_row("avg_withdrawal_rate", "Guardrails — Average Withdrawal Rate", is_pct=True),
        build_percentile_row("pct_below_initial", "Guardrails — Duration Below", is_pct=True),
    ]

    if has_fixed:
        percentile_rows.extend([
            build_percentile_row("Fixed_Spend_Rate", "Fixed — Spend Rate", is_pct=True),
            build_percentile_row("Fixed_Min_WD_Rate", "Fixed — Min Withdrawal Rate", is_pct=True),
            build_percentile_row("Fixed_Max_WD_Rate", "Fixed — Max Withdrawal Rate", is_pct=True),
            build_percentile_row("Fixed_Median_WD_Rate", "Fixed — Median Withdrawal Rate", is_pct=True),
            build_percentile_row("Fixed_Avg_WD_Rate", "Fixed — Average Withdrawal Rate", is_pct=True),
        ])

    if has_cape:
        percentile_rows.extend([
            build_percentile_row("CAPE_Min_Spend_Rate", "CAPE — Min Spend Rate", is_pct=True),
            build_percentile_row("CAPE_Max_Spend_Rate", "CAPE — Max Spend Rate", is_pct=True),
            build_percentile_row("CAPE_Median_Spend_Rate", "CAPE — Median Spend Rate", is_pct=True),
            build_percentile_row("CAPE_Avg_Spend_Rate", "CAPE — Average Spend Rate", is_pct=True),
            build_percentile_row("CAPE_Min_WD_Rate", "CAPE — Min Withdrawal Rate", is_pct=True),
            build_percentile_row("CAPE_Max_WD_Rate", "CAPE — Max Withdrawal Rate", is_pct=True),
            build_percentile_row("CAPE_Median_WD_Rate", "CAPE — Median Withdrawal Rate", is_pct=True),
            build_percentile_row("CAPE_Avg_WD_Rate", "CAPE — Average Withdrawal Rate", is_pct=True),
            build_percentile_row("CAPE_Pct_Below_Fixed", "CAPE — Duration Below", is_pct=True),
        ])
    if has_vpw:
        percentile_rows.extend([
            build_percentile_row("VPW_Min_Spend_Rate", "VPW — Min Spend Rate", is_pct=True),
            build_percentile_row("VPW_Max_Spend_Rate", "VPW — Max Spend Rate", is_pct=True),
            build_percentile_row("VPW_Median_Spend_Rate", "VPW — Median Spend Rate", is_pct=True),
            build_percentile_row("VPW_Avg_Spend_Rate", "VPW — Average Spend Rate", is_pct=True),
            build_percentile_row("VPW_Min_WD_Rate", "VPW — Min Withdrawal Rate", is_pct=True),
            build_percentile_row("VPW_Max_WD_Rate", "VPW — Max Withdrawal Rate", is_pct=True),
            build_percentile_row("VPW_Median_WD_Rate", "VPW — Median Withdrawal Rate", is_pct=True),
            build_percentile_row("VPW_Avg_WD_Rate", "VPW — Average Withdrawal Rate", is_pct=True),
            build_percentile_row("VPW_Pct_Below_Fixed", "VPW — Duration Below", is_pct=True),
        ])

    percentile_rows.append(build_percentile_row("Guardrail_Success", "Guardrails — Failure %", is_pct_flipped=True))
    if has_fixed:
        percentile_rows.append(build_percentile_row("Fixed_Success", "Fixed — Failure %", is_pct_flipped=True))
    if has_cape:
        percentile_rows.append(build_percentile_row("CAPE_Success", "CAPE — Failure %", is_pct_flipped=True))
    if has_vpw:
        percentile_rows.append(build_percentile_row("VPW_Success", "VPW — Failure %", is_pct_flipped=True))

    has_tax_summary = "Lifetime_Tax_Rate" in results_df.columns or any(
        f"{prefix}Lifetime_Tax_Rate" in results_df.columns
        for prefix in ("Fixed_", "CAPE_", "VPW_")
    )
    if has_tax_summary:
        st.markdown("**Historical Tax & ACA Efficiency Summary**")
        st.caption(
            "Percentiles use the same historical retirement ordering as the spend-rate table: "
            "worst to best by duration below initial, then minimum spending."
        )
        tax_strategy_prefixes = [
            ("", "Guardrails"),
            ("Fixed_", "Fixed"),
            ("CAPE_", "CAPE"),
            ("VPW_", "VPW"),
        ]
        tax_summary_rows = []
        for prefix, strategy_label in tax_strategy_prefixes:
            for metric_key, metric_label, metric_kind in _HISTORICAL_TAX_METRICS:
                col_name = f"{prefix}{metric_key}" if prefix else metric_key
                row_label = f"{strategy_label} — {metric_label}"
                tax_summary_rows.append(
                    _build_historical_tax_percentile_row(
                        sorted_df,
                        results_df,
                        col_name,
                        row_label,
                        metric_kind,
                    )
                )
        st.table(pd.DataFrame(tax_summary_rows).set_index("Metric"))
        if "Lifetime_Tax_Rate" in results_df.columns and "Pre65_Avg_Annual_Tax_Pct" not in results_df.columns:
            st.caption(
                "Re-run analysis to populate pre-65 average tax metrics and multi-strategy tax columns."
            )
        elif "Lifetime_Tax_Rate" in results_df.columns and not any(
            f"{prefix}Lifetime_Tax_Rate" in results_df.columns
            for prefix in ("Fixed_", "CAPE_", "VPW_")
        ):
            st.caption("Re-run analysis to populate Fixed, CAPE, and VPW tax metrics.")
 
    st.markdown("**Guardrail spend rate percentiles across historical retirements**")
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
                line=dict(color="#e377c2", dash="dash"),
                hovertemplate="Start: %{x}<br>Success: %{y:.0f}%<extra></extra>",
            )
        )
    fig_success.update_layout(
        title="Success by Retirement Start Year",
        xaxis_title="Retirement Start Year",
        yaxis_title="Success (100 = success, 0 = failure)",
        yaxis=dict(tickvals=[0, 100], ticktext=["Fail", "Success"]),
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
                line=dict(color="#e377c2", dash="dash"),
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
            "Min Spend Rate",
            "Max Spend Rate",
            "Median Spend Rate",
            "Average Spend Rate",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    spending_specs = [
        ("min_spending", "CAPE_Minimum_Withdrawal", "VPW_Minimum_Withdrawal", 1, 1, "#d62728"),
        ("max_spending", "CAPE_Maximum_Withdrawal", "VPW_Maximum_Withdrawal", 1, 2, "#e377c2"),
        ("median_spending", "CAPE_Median_Withdrawal", "VPW_Median_Withdrawal", 2, 1, "#9467bd"),
        ("avg_spending", "CAPE_Average_Withdrawal", "VPW_Average_Withdrawal", 2, 2, "#1f77b4"),
    ]
    for column, cape_column, vpw_column, row, col, color in spending_specs:
        fig_spending.add_trace(
            go.Scatter(
                x=results_df["Start_Year"],
                y=results_df[column] * yearly_scale / starting_portfolio,
                mode="lines",
                name=f"Guardrails {column.replace('_', ' ').title()}",
                line=dict(color=color),
                hovertemplate="Start: %{x}<br>%{y:.2%}<extra></extra>",
                showlegend=True,
            ),
            row=row,
            col=col,
        )
        if has_cape and cape_column in results_df.columns:
            fig_spending.add_trace(
                go.Scatter(
                    x=results_df["Start_Year"],
                    y=results_df[cape_column] / starting_portfolio,
                    mode="lines",
                    name=f"CAPE {column.replace('_', ' ').title()}",
                    line=dict(color=color, dash="dash"),
                    hovertemplate="Start: %{x}<br>%{y:.2%}<extra></extra>",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
        if has_vpw and vpw_column in results_df.columns:
            fig_spending.add_trace(
                go.Scatter(
                    x=results_df["Start_Year"],
                    y=results_df[vpw_column] / starting_portfolio,
                    mode="lines",
                    name=f"VPW {column.replace('_', ' ').title()}",
                    line=dict(color=color, dash="dot"),
                    hovertemplate="Start: %{x}<br>%{y:.2%}<extra></extra>",
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
        fig_spending.update_yaxes(tickformat=".2%", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
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
