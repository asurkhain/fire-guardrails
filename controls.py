import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app_settings import Settings
from utils import get_combo_tempdir

MONTHS_PER_YEAR = 12
DEFAULT_FIXED_WITHDRAWAL_RATE = 0.0375
DEFAULT_INITIAL_SPENDING_RATE = 0.04
DEFAULT_STOCK_PCT = 0.90
DEFAULT_SPENDING_CAP_OPTION = "150%"
DEFAULT_SPENDING_FLOOR_OPTION = "75%"
STARTING_ACCOUNTS_FILENAME = "starting_accounts.csv"
RECURRING_CASHFLOWS_FILENAME = "recurring_cashflows.csv"
RECURRING_CASHFLOW_COLUMNS = ("label", "start_month", "end_month", "amount")
TAXABLE_ACCOUNTS_DRAFT_KEY = "taxable_mode_accounts_draft"
VALID_ACCOUNT_TYPES = ("taxable", "retirement pretax", "retirement post tax")
VALID_ASSET_TYPES = ("equity", "equity ex-US", "bond")
DEFAULT_TAXABLE_MODE_ACCOUNTS = [
    {
        "account_type": "taxable",
        "asset_type": "equity",
        "cost_basis": 600_000.0,
        "total": 1_000_000.0,
    }
]

DIRTY_COLOR = "#8B0000"  # dark red

# Helpers for cashflow management
def sanitize_cashflows(raw_cashflows):
    """Convert raw cashflow inputs into validated dictionaries with numeric values."""

    sanitized = []
    for flow in raw_cashflows or []:
        try:
            start = int(flow.get("start_month", 0))
            end = int(flow.get("end_month", 0))
            amount = float(flow.get("amount", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        if end < start:
            continue

        label_raw = flow.get("label") if isinstance(flow, dict) else None
        label = str(label_raw).strip() if label_raw is not None else None

        sanitized.append({
            "start_month": start,
            "end_month": end,
            "amount": amount,
            "label": label,
        })
    return sanitized


def cashflows_to_tuple(cashflows):
    """Create a hashable snapshot of each cashflow's timing and amount."""

    return tuple((cf["start_month"], cf["end_month"], cf["amount"]) for cf in cashflows)


def clear_cashflow_widget_state(start_idx: int = 0) -> None:
    """Remove cached widget values for cashflow inputs from the specified starting index onward."""

    prefixes = (
        "cf_start_",
        "cf_end_",
        "cf_amount_",
        "cf_label_",
    )

    keys_to_drop = []
    for key in list(st.session_state.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                if suffix.isdigit() and int(suffix) >= start_idx:
                    keys_to_drop.append(key)
                break

    for key in keys_to_drop:
        del st.session_state[key]


# Ensure widget defaults for existing cashflows remain synchronized prior to rendering inputs
def ensure_cashflow_widget_state(idx: int, flow: dict) -> None:
    """Keep a cashflow row's widget state aligned with sanitized default values."""

    start_key = f"cf_start_{idx}"
    end_key = f"cf_end_{idx}"
    amount_key = f"cf_amount_{idx}"
    label_key = f"cf_label_{idx}"

    default_start = int(flow.get("start_month", 0))
    default_end = int(flow.get("end_month", default_start))
    default_amount = float(flow.get("amount", 0.0)) * MONTHS_PER_YEAR
    default_label = str(flow.get("label") or f"Cashflow {idx + 1}")

    if start_key not in st.session_state:
        st.session_state[start_key] = default_start

    if end_key not in st.session_state:
        st.session_state[end_key] = max(default_end, st.session_state[start_key])
    elif st.session_state[end_key] < st.session_state[start_key]:
        st.session_state[end_key] = st.session_state[start_key]

    if amount_key not in st.session_state:
        st.session_state[amount_key] = default_amount

    if label_key not in st.session_state or not str(st.session_state[label_key]).strip():
        st.session_state[label_key] = default_label


def sync_cashflows_from_widgets() -> None:
    """Write the current widget state back into the tracked cashflow configurations."""

    flows = st.session_state.get("cashflows")
    if not flows:
        return

    for idx, flow in enumerate(flows):
        start_key = f"cf_start_{idx}"
        end_key = f"cf_end_{idx}"
        amount_key = f"cf_amount_{idx}"
        label_key = f"cf_label_{idx}"

        try:
            start_val = int(st.session_state.get(start_key, flow.get("start_month", 0)))
        except (TypeError, ValueError):
            start_val = int(flow.get("start_month", 0))

        try:
            end_val = int(st.session_state.get(end_key, flow.get("end_month", start_val)))
        except (TypeError, ValueError):
            end_val = int(flow.get("end_month", start_val))

        if end_val < start_val:
            end_val = start_val

        try:
            amount_val = float(
                st.session_state.get(amount_key, float(flow.get("amount", 0.0)) * MONTHS_PER_YEAR)
            )
        except (TypeError, ValueError):
            amount_val = float(flow.get("amount", 0.0)) * MONTHS_PER_YEAR

        label_raw = st.session_state.get(label_key, flow.get("label") or f"Cashflow {idx + 1}")
        label_val = str(label_raw).strip() or f"Cashflow {idx + 1}"

        st.session_state[start_key] = start_val
        st.session_state[end_key] = end_val
        st.session_state[amount_key] = amount_val
        st.session_state[label_key] = label_val

        flows[idx] = {
            "start_month": start_val,
            "end_month": end_val,
            "amount": amount_val / MONTHS_PER_YEAR,
            "label": label_val,
        }

    save_recurring_cashflows_to_csv(flows)


def get_date_state(key: str, default: datetime.date) -> datetime.date:
    """Return a stored date value, falling back to the provided baseline when parsing fails."""

    value = st.session_state.get(key)
    if value is None:
        return default
    if isinstance(value, datetime.date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return default


def get_int_state(key: str, default: int) -> int:
    """Return an integer from session state, using the fallback when conversion is unsafe."""

    value = st.session_state.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_state(key: str, default: float) -> float:
    """Return a floating-point number from session state, with graceful fallback on errors."""

    value = st.session_state.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def init_start_date_field(today_date, sim_start_key, start_date_key, start_date_default, mode, is_guidance):
    """Manage the retirement start date widget when switching between modes."""

    previous_mode_key = "_previous_mode"
    is_historical = mode in ("Historical Mode", "Taxable Mode", "Taxable Historical Mode")
    is_combo = mode == mode == "Combo Mode"

    if sim_start_key not in st.session_state:
        st.session_state[sim_start_key] = get_date_state(start_date_key, start_date_default)

    previous_mode = st.session_state.get(previous_mode_key)
    if previous_mode != mode:
        if is_guidance:
            st.session_state[sim_start_key] = get_date_state(start_date_key, start_date_default)
            st.session_state[start_date_key] = today_date
        elif is_combo or is_historical:
            st.session_state[sim_start_key] = get_date_state(start_date_key, start_date_default)
        else:
            restored_start = st.session_state.get(sim_start_key, start_date_default)
            if not isinstance(restored_start, datetime.date):
                restored_start = get_date_state(start_date_key, start_date_default)
            st.session_state[start_date_key] = restored_start
        st.session_state[previous_mode_key] = mode

    if start_date_key not in st.session_state and not (is_combo or is_historical):
        default_start = today_date if is_guidance else st.session_state.get(sim_start_key, start_date_default)
        if not isinstance(default_start, datetime.date):
            default_start = start_date_default
        st.session_state[start_date_key] = default_start

def get_starting_accounts_sum() -> float:
    """Return the sum of accounts in starting_accounts.csv, or default total."""

    accounts = load_starting_accounts_from_csv()
    if accounts:
        return float(sum(acc["total"] for acc in accounts))
    return float(sum(acc["total"] for acc in DEFAULT_TAXABLE_MODE_ACCOUNTS))


def init_initial_portfolio_value(mode: str) -> None:
    """Manage the Initial Portfolio Value state when switching between modes."""

    is_taxable = (mode in ("Taxable Mode", "Taxable Historical Mode"))
    is_taxable_simulation = (mode == "Taxable Simulation Mode")
    is_tax_aware = is_taxable or is_taxable_simulation

    if not is_tax_aware:
        previous_mode_key = "_previous_mode"
        previous_mode = st.session_state.get(previous_mode_key)

        was_tax_aware = False
        if previous_mode is not None:
            was_taxable = (previous_mode in ("Taxable Mode", "Taxable Historical Mode"))
            was_taxable_simulation = (previous_mode == "Taxable Simulation Mode")
            was_tax_aware = was_taxable or was_taxable_simulation

        if "initial_portfolio_value" not in st.session_state or (previous_mode != mode and was_tax_aware):
            st.session_state["initial_portfolio_value"] = get_starting_accounts_sum()


def initialize_display():
    """Populate session state with default values required for the control widgets."""

    if "show_advanced_modal" not in st.session_state:
        st.session_state["show_advanced_modal"] = False
    if "spending_cap_option" not in st.session_state:
        st.session_state["spending_cap_option"] = DEFAULT_SPENDING_CAP_OPTION
    if "spending_floor_option" not in st.session_state:
        st.session_state["spending_floor_option"] = DEFAULT_SPENDING_FLOOR_OPTION
    if "stock_pct" in st.session_state:
        st.session_state["_backup_stock_pct"] = st.session_state["stock_pct"]
    else:
        if "_backup_stock_pct" in st.session_state:
            st.session_state["stock_pct"] = st.session_state["_backup_stock_pct"]
        elif "taxable_mode_stock_pct" in st.session_state:
            st.session_state["stock_pct"] = float(st.session_state["taxable_mode_stock_pct"])
            st.session_state["_backup_stock_pct"] = st.session_state["stock_pct"]
        else:
            st.session_state["stock_pct"] = DEFAULT_STOCK_PCT
            st.session_state["_backup_stock_pct"] = DEFAULT_STOCK_PCT
    if "cashflows" not in st.session_state:
        loaded_cashflows = load_recurring_cashflows_from_csv()
        st.session_state["cashflows"] = loaded_cashflows if loaded_cashflows is not None else []
    if "conditional_cashflows" not in st.session_state:
        st.session_state["conditional_cashflows"] = []
    if "final_value_target" not in st.session_state:
        st.session_state["final_value_target"] = 0.0
    if "retirement_duration_years" not in st.session_state:
        st.session_state["retirement_duration_years"] = 50
    if "fixed_monthly_withdrawal" not in st.session_state:
        st.session_state["fixed_monthly_withdrawal"] = None
    if "shiller_extension_mode" not in st.session_state:
        st.session_state["shiller_extension_mode"] = "hard_code_date"
    if "final_value_target_ramp_down" not in st.session_state:
        st.session_state["final_value_target_ramp_down"] = True
    st.session_state.pop("historical_parallel_backend", None)
    if "stock_pct_range" not in st.session_state:
        st.session_state["stock_pct_range"] = (0.50, 0.90)
    if "upper_guardrail_success_range" not in st.session_state:
        st.session_state["upper_guardrail_success_range"] = (0.80, 1.00)
    if "lower_guardrail_success_range" not in st.session_state:
        st.session_state["lower_guardrail_success_range"] = (0.25, 0.45)
    if "target_success_rate_range" not in st.session_state:
        st.session_state["target_success_rate_range"] = (0.70, 0.90)
    if "initial_portfolio_value" not in st.session_state:
        st.session_state["initial_portfolio_value"] = get_starting_accounts_sum()
    if "fixed_yearly_withdrawal_input" not in st.session_state:
        _default_fixed_yearly_input = DEFAULT_FIXED_WITHDRAWAL_RATE * 100.0
        st.session_state["fixed_yearly_withdrawal_input"] = _default_fixed_yearly_input
    if "initial_yearly_spending" not in st.session_state:
        _default_initial_yearly_spending = DEFAULT_INITIAL_SPENDING_RATE * 100.0
        st.session_state["initial_yearly_spending"] = _default_initial_yearly_spending
    


def draw_cashflow_widget_rows():
    """Render editable cashflow rows and keep their widgets synchronized."""

    for idx, flow in enumerate(st.session_state["cashflows"]):
        ensure_cashflow_widget_state(idx, flow)

        name_col, remove_col = st.columns([1, 0.15])
        label_key = f"cf_label_{idx}"
        name_col.text_input(
            "Cashflow Name",
            key=label_key,
            label_visibility="collapsed",
            placeholder="Cashflow name",
        )
        if remove_col.button("✕", key=f"cf_remove_{idx}"):
            st.session_state["cashflows"].pop(idx)
            clear_cashflow_widget_state(idx)
            save_recurring_cashflows_to_csv(st.session_state["cashflows"])
            st.rerun()

        col_start, col_end, col_amount = st.columns(3)
        col_start.number_input(
            "Start Month",
            min_value=0,
            step=1,
            key=f"cf_start_{idx}",
        )
        col_end.number_input(
            "End Month",
            min_value=0,
            step=1,
            key=f"cf_end_{idx}",
        )
        col_amount.number_input(
            "Amount ($/yr)",
            step=600.0,
            format="%0.0f",
            key=f"cf_amount_{idx}",
        )


# Helpers for conditional cashflow management
CONDITIONAL_THRESHOLD_OPTIONS = tuple(f"{pct}%" for pct in range(5, 100, 5))
CONDITIONAL_TYPE_OPTIONS = ("One-time", "Recurring")


def sanitize_conditional_cashflows(raw_cashflows):
    """Convert raw conditional cashflow inputs into validated dictionaries."""

    sanitized = []
    for flow in raw_cashflows or []:
        try:
            cashflow_type = str(flow.get("cashflow_type", "one_time"))
            if cashflow_type not in ("one_time", "recurring"):
                continue
            trigger_threshold = str(flow.get("trigger_threshold", "50%"))
            amount = float(flow.get("amount", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        label_raw = flow.get("label") if isinstance(flow, dict) else None
        label = str(label_raw).strip() if label_raw is not None else None

        sanitized.append({
            "cashflow_type": cashflow_type,
            "trigger_threshold": trigger_threshold,
            "amount": amount,
            "label": label,
        })
    return sanitized


def clear_conditional_cashflow_widget_state(start_idx: int = 0) -> None:
    """Remove cached widget values for conditional cashflow inputs from the specified index onward."""

    prefixes = (
        "ccf_type_",
        "ccf_threshold_",
        "ccf_amount_",
        "ccf_label_",
    )

    keys_to_drop = []
    for key in list(st.session_state.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix.isdigit() and int(suffix) >= start_idx:
                    keys_to_drop.append(key)
                break

    for key in keys_to_drop:
        del st.session_state[key]


def ensure_conditional_cashflow_widget_state(idx: int, flow: dict) -> None:
    """Keep a conditional cashflow row's widget state aligned with sanitized default values."""

    type_key = f"ccf_type_{idx}"
    threshold_key = f"ccf_threshold_{idx}"
    amount_key = f"ccf_amount_{idx}"
    label_key = f"ccf_label_{idx}"

    default_type = str(flow.get("cashflow_type", "one_time"))
    # Convert internal value to display value
    default_type_display = "One-time" if default_type == "one_time" else "Recurring"
    default_threshold = str(flow.get("trigger_threshold", "50%"))
    default_amount = float(flow.get("amount", 0.0)) * MONTHS_PER_YEAR
    default_label = str(flow.get("label") or f"Conditional {idx + 1}")

    if type_key not in st.session_state:
        st.session_state[type_key] = default_type_display

    if threshold_key not in st.session_state:
        st.session_state[threshold_key] = default_threshold

    if amount_key not in st.session_state:
        st.session_state[amount_key] = default_amount

    if label_key not in st.session_state or not str(st.session_state[label_key]).strip():
        st.session_state[label_key] = default_label


def sync_conditional_cashflows_from_widgets() -> None:
    """Write the current widget state back into the tracked conditional cashflow configurations."""

    flows = st.session_state.get("conditional_cashflows")
    if not flows:
        return

    for idx, flow in enumerate(flows):
        type_key = f"ccf_type_{idx}"
        threshold_key = f"ccf_threshold_{idx}"
        amount_key = f"ccf_amount_{idx}"
        label_key = f"ccf_label_{idx}"

        type_display = st.session_state.get(type_key, "One-time")
        # Convert display value to internal value
        type_val = "one_time" if type_display == "One-time" else "recurring"

        threshold_val = st.session_state.get(threshold_key, flow.get("trigger_threshold", "50%"))

        try:
            amount_val = float(
                st.session_state.get(amount_key, float(flow.get("amount", 0.0)) * MONTHS_PER_YEAR)
            )
        except (TypeError, ValueError):
            amount_val = float(flow.get("amount", 0.0)) * MONTHS_PER_YEAR

        label_raw = st.session_state.get(label_key, flow.get("label") or f"Conditional {idx + 1}")
        label_val = str(label_raw).strip() or f"Conditional {idx + 1}"

        st.session_state[type_key] = type_display
        st.session_state[threshold_key] = threshold_val
        st.session_state[amount_key] = amount_val
        st.session_state[label_key] = label_val

        flows[idx] = {
            "cashflow_type": type_val,
            "trigger_threshold": threshold_val,
            "amount": amount_val / MONTHS_PER_YEAR,
            "label": label_val,
        }


def draw_conditional_cashflow_widget_rows():
    """Render editable conditional cashflow rows and keep their widgets synchronized."""

    for idx, flow in enumerate(st.session_state.get("conditional_cashflows", [])):
        ensure_conditional_cashflow_widget_state(idx, flow)

        # Row 1: Name and remove button
        name_col, remove_col = st.columns([1, 0.15])
        label_key = f"ccf_label_{idx}"
        name_col.text_input(
            "Conditional Cashflow Name",
            key=label_key,
            label_visibility="collapsed",
            placeholder="Conditional cashflow name",
        )
        if remove_col.button("✕", key=f"ccf_remove_{idx}"):
            st.session_state["conditional_cashflows"].pop(idx)
            clear_conditional_cashflow_widget_state(idx)
            st.rerun()

        # Row 2: Type, Threshold, Amount
        col_type, col_threshold, col_amount = st.columns(3)
        col_type.selectbox(
            "Type",
            options=CONDITIONAL_TYPE_OPTIONS,
            key=f"ccf_type_{idx}",
        )
        col_threshold.selectbox(
            "Trigger Below",
            options=CONDITIONAL_THRESHOLD_OPTIONS,
            key=f"ccf_threshold_{idx}",
            help="Cashflow triggers when spending falls below this % of initial spending. Recurring cashflows "
                 "will then end if and when spending increases back to 100% of initial spending.",
        )
        col_amount.number_input(
            "Amount ($/yr)",
            step=600.0,
            format="%0.0f",
            key=f"ccf_amount_{idx}",
        )


def render_dirty_banner():
    """Display a warning banner indicating that the inputs changed and a rerun is needed."""

    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            margin: 0 0 0.75rem 0;
            border: 1px solid {DIRTY_COLOR};
            background: rgba(139,0,0,0.08);
            color: {DIRTY_COLOR};
            border-radius: 6px;
            font-weight: 600;">
            Inputs changed, please rerun
        </div>
        """,
        unsafe_allow_html=True
    )

def draw_dirty_border():
    """Add a temporary border effect around the app to reinforce the rerun prompt."""

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            border: 3px solid {DIRTY_COLOR};
            border-radius: 8px;
            padding: 6px;
            filter: grayscale(30%) brightness(0.95);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def starting_accounts_csv_path() -> Path:
    """Return the path used to persist starting account rows between sessions."""

    return Path(get_combo_tempdir()) / STARTING_ACCOUNTS_FILENAME


def recurring_cashflows_csv_path() -> Path:
    """Return the path used to persist recurring cashflows between sessions."""

    return Path(get_combo_tempdir()) / RECURRING_CASHFLOWS_FILENAME


def _coerce_recurring_cashflows(rows: list[dict]) -> list[dict]:
    """Validate and normalise recurring cashflow rows from CSV or session state."""

    coerced = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            coerced.append({
                "label": str(row.get("label") or "").strip() or "Cashflow",
                "start_month": int(float(row.get("start_month", 0) or 0)),
                "end_month": int(float(row.get("end_month", 0) or 0)),
                "amount": float(row.get("amount", 0.0) or 0.0),
            })
        except (TypeError, ValueError):
            continue
    return coerced


def load_recurring_cashflows_from_csv() -> list[dict] | None:
    """Load recurring cashflows from the temp-folder CSV, if present.

    Returns None if the file does not exist or contains no valid rows.
    Amounts in the returned dicts are in **monthly** terms.
    """

    path = recurring_cashflows_csv_path()
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    if df.empty:
        return None
    missing = set(RECURRING_CASHFLOW_COLUMNS) - set(df.columns)
    if missing:
        return None
    rows = _coerce_recurring_cashflows(df.to_dict("records"))
    return rows if rows else None


def save_recurring_cashflows_to_csv(cashflows: list[dict]) -> None:
    """Persist recurring cashflows to the temp-folder CSV.

    Amounts are stored in **monthly** terms (the internal representation).
    """

    coerced = _coerce_recurring_cashflows(cashflows)
    path = recurring_cashflows_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if coerced:
        pd.DataFrame(coerced)[list(RECURRING_CASHFLOW_COLUMNS)].to_csv(path, index=False)
    else:
        # Write an empty file with headers so load can distinguish "no rows" from "not saved yet"
        pd.DataFrame(columns=list(RECURRING_CASHFLOW_COLUMNS)).to_csv(path, index=False)


def _coerce_starting_accounts(rows: list[dict]) -> list[dict]:
    """Coerce account rows for persistence without dropping in-progress edits."""

    coerced = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            coerced.append({
                "account_type": str(row.get("account_type", "")).strip() or "taxable",
                "asset_type": str(row.get("asset_type", "")).strip() or "equity",
                "cost_basis": float(row.get("cost_basis", 0.0) or 0.0),
                "total": float(row.get("total", 0.0) or 0.0),
            })
        except (TypeError, ValueError):
            continue
    return coerced


ACCOUNT_TABLE_COLUMNS = ("account_type", "asset_type", "cost_basis", "total")


def _apply_data_editor_delta(base_df: pd.DataFrame, delta: dict) -> pd.DataFrame:
    """Apply Streamlit data_editor delta payloads onto a base table."""

    df = base_df.copy().reset_index(drop=True)
    for row_idx, updates in sorted((int(k), v) for k, v in delta.get("edited_rows", {}).items()):
        if row_idx < 0 or row_idx >= len(df):
            continue
        for col, val in updates.items():
            if col in df.columns:
                df.at[row_idx, col] = val
    for row in delta.get("added_rows", []):
        if isinstance(row, dict) and row:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    for row_idx in sorted((int(i) for i in delta.get("deleted_rows", [])), reverse=True):
        if 0 <= row_idx < len(df):
            df = df.drop(index=row_idx)
    return df.reset_index(drop=True)


def _accounts_editor_to_dataframe(editor_state) -> pd.DataFrame | None:
    """Normalize data_editor state into a flat accounts table."""

    if editor_state is None:
        return None
    if isinstance(editor_state, pd.DataFrame):
        return editor_state.copy()
    if isinstance(editor_state, list):
        if not editor_state:
            return pd.DataFrame(columns=list(ACCOUNT_TABLE_COLUMNS))
        return pd.DataFrame(editor_state)
    if isinstance(editor_state, dict):
        if {"edited_rows", "added_rows", "deleted_rows"}.intersection(editor_state):
            return None
        if not editor_state:
            return pd.DataFrame(columns=list(ACCOUNT_TABLE_COLUMNS))
        sample_value = next(iter(editor_state.values()))
        if isinstance(sample_value, dict):
            return pd.DataFrame.from_dict(editor_state, orient="index").reset_index(drop=True)
        if isinstance(sample_value, (list, tuple, pd.Series)):
            return pd.DataFrame(editor_state)
        return None
    return None


def accounts_records_from_editor(
    editor_state,
    base_accounts: list[dict] | None = None,
) -> list[dict]:
    """Read the latest account rows from a data_editor widget state."""

    editor_df = _accounts_editor_to_dataframe(editor_state)
    if editor_df is None and isinstance(editor_state, dict):
        delta_keys = {"edited_rows", "added_rows", "deleted_rows"}
        if delta_keys.intersection(editor_state) and base_accounts is not None:
            base_df = pd.DataFrame(_coerce_starting_accounts(base_accounts))
            editor_df = _apply_data_editor_delta(base_df, editor_state)
        else:
            return []
    if editor_df is None or editor_df.empty:
        return []
    missing_columns = set(ACCOUNT_TABLE_COLUMNS) - set(editor_df.columns)
    if missing_columns:
        return []
    editor_df = editor_df.loc[:, list(ACCOUNT_TABLE_COLUMNS)]
    return _coerce_starting_accounts(editor_df.to_dict("records"))


def init_taxable_accounts_widget_state() -> None:
    """Initialize committed and draft account tables in session state."""

    if "taxable_mode_accounts" not in st.session_state:
        st.session_state["taxable_mode_accounts"] = default_starting_accounts()
    if TAXABLE_ACCOUNTS_DRAFT_KEY not in st.session_state:
        st.session_state[TAXABLE_ACCOUNTS_DRAFT_KEY] = pd.DataFrame(
            _coerce_starting_accounts(st.session_state["taxable_mode_accounts"])
        )


def commit_taxable_accounts_from_editor(editor_state) -> tuple[bool, str]:
    """Persist the edited table to session state and CSV."""

    records = accounts_records_from_editor(
        editor_state,
        base_accounts=st.session_state.get("taxable_mode_accounts"),
    )
    if not records:
        return False, "No account rows to save."
    committed_df = pd.DataFrame(records)
    st.session_state["taxable_mode_accounts"] = records
    st.session_state[TAXABLE_ACCOUNTS_DRAFT_KEY] = committed_df
    st.session_state.pop("taxable_mode_accounts_editor", None)
    save_starting_accounts_to_csv(records)
    return True, f"Saved {len(records)} account row(s)."


def _sanitize_starting_accounts(rows: list[dict]) -> list[dict]:
    """Validate and normalize account rows loaded from CSV or session state."""

    sanitized = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        account_type = str(row.get("account_type", "")).strip()
        asset_type = str(row.get("asset_type", "")).strip()
        if account_type not in VALID_ACCOUNT_TYPES or asset_type not in VALID_ASSET_TYPES:
            continue
        try:
            cost_basis = float(row.get("cost_basis", 0.0))
            total = float(row.get("total", 0.0))
        except (TypeError, ValueError):
            continue
        if total < 0.0 or cost_basis < 0.0:
            continue
        sanitized.append({
            "account_type": account_type,
            "asset_type": asset_type,
            "cost_basis": cost_basis,
            "total": total,
        })
    return sanitized


def load_starting_accounts_from_csv() -> list[dict] | None:
    """Load starting accounts from the temp-folder CSV, if present."""

    path = starting_accounts_csv_path()
    if not path.is_file():
        return None
    try:
        accounts_df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    if accounts_df.empty:
        return None
    accounts = _sanitize_starting_accounts(accounts_df.to_dict("records"))
    return accounts or None


def save_starting_accounts_to_csv(accounts: list[dict]) -> None:
    """Persist starting accounts to the temp-folder CSV."""

    coerced = _coerce_starting_accounts(accounts)
    if not coerced:
        return
    path = starting_accounts_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coerced).to_csv(path, index=False)


def default_starting_accounts() -> list[dict]:
    """Return the default starting account rows for taxable modes."""

    loaded = load_starting_accounts_from_csv()
    if loaded:
        return loaded
    return [dict(account) for account in DEFAULT_TAXABLE_MODE_ACCOUNTS]


def hydrate_settings():
    """Load shared configuration details from the URL and apply them to the session."""

    query_params = st.query_params
    config_value = query_params.get("config") if query_params is not None else None
    if config_value:
        try:
            loaded_settings = Settings.from_base64(config_value)
            loaded_settings.apply_to_session_state(st.session_state)
            st.session_state["settings"] = loaded_settings
            st.session_state["_settings_loaded_from_query"] = True
        except Exception as exc:
            st.session_state["_settings_error"] = str(exc)
    st.session_state["_settings_initialized"] = True
