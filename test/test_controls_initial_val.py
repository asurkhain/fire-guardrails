from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
import streamlit as st

import controls


def test_get_starting_accounts_sum(tmp_path, monkeypatch):
    # Test when csv is empty or doesn't exist
    monkeypatch.setattr(controls, "starting_accounts_csv_path", lambda: tmp_path / "starting_accounts.csv")
    
    # Should fall back to default taxable accounts total (1,000,000)
    assert controls.get_starting_accounts_sum() == 1_000_000.0

    # Write a test CSV
    df = pd.DataFrame([
        {"account_type": "taxable", "asset_type": "equity", "cost_basis": 1000.0, "total": 5000.0},
        {"account_type": "retirement pretax", "asset_type": "bond", "cost_basis": 2000.0, "total": 3000.0},
    ])
    df.to_csv(tmp_path / "starting_accounts.csv", index=False)
    
    # Should sum the totals (5000 + 3000 = 8000)
    assert controls.get_starting_accounts_sum() == 8000.0


def test_init_initial_portfolio_value(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "starting_accounts_csv_path", lambda: tmp_path / "starting_accounts.csv")
    
    # Write a test CSV
    df = pd.DataFrame([
        {"account_type": "taxable", "asset_type": "equity", "cost_basis": 1000.0, "total": 5000.0},
    ])
    df.to_csv(tmp_path / "starting_accounts.csv", index=False)

    # Mock st.session_state
    session_state = {}
    monkeypatch.setattr(st, "session_state", session_state)

    # 1. Initial run (no previous mode)
    controls.init_initial_portfolio_value("Simulation Mode")
    assert session_state["initial_portfolio_value"] == 5000.0

    # 2. Rerun in same mode after user changes value to 10000.0
    session_state["initial_portfolio_value"] = 10000.0
    session_state["_previous_mode"] = "Simulation Mode"
    controls.init_initial_portfolio_value("Simulation Mode")
    assert session_state["initial_portfolio_value"] == 10000.0  # Should NOT reset!

    # 3. Switch between two non-taxable modes (Simulation Mode -> Historical Mode)
    # Should preserve the edited value
    controls.init_initial_portfolio_value("Historical Mode")
    assert session_state["initial_portfolio_value"] == 10000.0  # Should NOT reset!

    # 4. Switch from taxable to non-taxable (Taxable Simulation Mode -> Simulation Mode)
    session_state["_previous_mode"] = "Taxable Simulation Mode"
    # Should reset to the sum from starting_accounts.csv
    controls.init_initial_portfolio_value("Simulation Mode")
    assert session_state["initial_portfolio_value"] == 5000.0


def test_stock_pct_backup_and_restore(monkeypatch):
    # Mock st.session_state
    session_state = {}
    monkeypatch.setattr(st, "session_state", session_state)

    # 1. First run, stock_pct not in session state, should default to DEFAULT_STOCK_PCT (0.90)
    controls.initialize_display()
    assert session_state["stock_pct"] == 0.90
    assert session_state["_backup_stock_pct"] == 0.90

    # 2. User changes stock_pct to 0.75
    session_state["stock_pct"] = 0.75
    controls.initialize_display()
    assert session_state["_backup_stock_pct"] == 0.75

    # 3. Simulate mode switch where Streamlit deletes widget key "stock_pct"
    del session_state["stock_pct"]
    controls.initialize_display()
    # Should restore from backup
    assert session_state["stock_pct"] == 0.75


# ---------------------------------------------------------------------------
# Recurring cashflow CSV tests
# ---------------------------------------------------------------------------

def test_recurring_cashflows_csv_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "recurring_cashflows_csv_path", lambda: tmp_path / "recurring_cashflows.csv")

    cashflows = [
        {"label": "Social Security", "start_month": 0, "end_month": 240, "amount": 2500.0},
        {"label": "Part-time work", "start_month": 0, "end_month": 60, "amount": 1500.0},
    ]

    controls.save_recurring_cashflows_to_csv(cashflows)

    loaded = controls.load_recurring_cashflows_from_csv()
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0]["label"] == "Social Security"
    assert loaded[0]["start_month"] == 0
    assert loaded[0]["end_month"] == 240
    assert loaded[0]["amount"] == 2500.0
    assert loaded[1]["label"] == "Part-time work"
    assert loaded[1]["amount"] == 1500.0


def test_recurring_cashflows_csv_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "recurring_cashflows_csv_path", lambda: tmp_path / "recurring_cashflows.csv")

    # File doesn't exist yet
    assert controls.load_recurring_cashflows_from_csv() is None


def test_recurring_cashflows_csv_returns_none_when_empty(tmp_path, monkeypatch):
    path = tmp_path / "recurring_cashflows.csv"
    monkeypatch.setattr(controls, "recurring_cashflows_csv_path", lambda: path)

    # Save empty list -> writes header-only CSV
    controls.save_recurring_cashflows_to_csv([])
    assert path.exists()

    # Should return None because no rows
    assert controls.load_recurring_cashflows_from_csv() is None


def test_recurring_cashflows_csv_skips_malformed_rows(tmp_path, monkeypatch):
    path = tmp_path / "recurring_cashflows.csv"
    monkeypatch.setattr(controls, "recurring_cashflows_csv_path", lambda: path)

    # Write a CSV with one good row and one row with a bad amount
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "start_month", "end_month", "amount"])
        writer.writeheader()
        writer.writerow({"label": "Good", "start_month": 0, "end_month": 12, "amount": 500.0})
        writer.writerow({"label": "Bad", "start_month": "x", "end_month": "y", "amount": "not_a_number"})

    loaded = controls.load_recurring_cashflows_from_csv()
    # Good row should load; bad row should be skipped
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0]["label"] == "Good"


def test_initialize_display_loads_cashflows_from_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "recurring_cashflows_csv_path", lambda: tmp_path / "recurring_cashflows.csv")
    monkeypatch.setattr(controls, "starting_accounts_csv_path", lambda: tmp_path / "starting_accounts.csv")

    # Pre-populate the CSV
    cashflows = [{"label": "Pension", "start_month": 12, "end_month": 120, "amount": 3000.0}]
    controls.save_recurring_cashflows_to_csv(cashflows)

    session_state = {}
    monkeypatch.setattr(st, "session_state", session_state)

    controls.initialize_display()

    assert "cashflows" in session_state
    assert len(session_state["cashflows"]) == 1
    assert session_state["cashflows"][0]["label"] == "Pension"
    assert session_state["cashflows"][0]["amount"] == 3000.0


def test_initialize_display_defaults_empty_cashflows_when_no_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "recurring_cashflows_csv_path", lambda: tmp_path / "recurring_cashflows.csv")
    monkeypatch.setattr(controls, "starting_accounts_csv_path", lambda: tmp_path / "starting_accounts.csv")

    session_state = {}
    monkeypatch.setattr(st, "session_state", session_state)

    controls.initialize_display()

    assert "cashflows" in session_state
    assert session_state["cashflows"] == []

