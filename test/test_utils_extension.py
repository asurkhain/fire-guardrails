import pandas as pd

from app_settings import Settings
import shiller_utils


def _make_shiller_frame(start="2006-09-01", periods=30):
    dates = pd.date_range(start=start, periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "Date": dates,
            "Real Total Return Price": [100.0 + i for i in range(periods)],
            "Real Total Bond Returns": [50.0 + i * 0.5 for i in range(periods)],
            "CAPE": [20.0 for _ in range(periods)],
            "TR CAPE": [22.0 for _ in range(periods)],
        }
    )


def test_hard_code_extension_uses_anchor_date_for_covering_range(monkeypatch):
    df = _make_shiller_frame()
    monkeypatch.setattr(shiller_utils, "SHILLER_EXTENSION_MODE", "hard_code_date")

    end_date = pd.Timestamp("2010-12-01")
    extended = shiller_utils.extend_shiller_data_to_cover_date(df, end_date=end_date)

    assert pd.to_datetime(extended["Date"].max()) >= end_date
    assert len(extended) >= len(df)
    assert extended["Date"].iloc[: len(df)].tolist() == df["Date"].tolist()
    assert pd.to_datetime(extended["Date"].iloc[len(df)]).year >= 2009


def test_settings_carries_shiller_extension_mode():
    settings = Settings(
        mode="Simulation Mode",
        start_date=pd.Timestamp("1985-01-01").date(),
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
        shiller_extension_mode="average_return",
    )

    assert settings.to_dict()["shiller_extension_mode"] == "average_return"
    session_state = {}
    settings.apply_to_session_state(session_state)
    assert session_state["shiller_extension_mode"] == "average_return"


def test_settings_carries_final_value_target_ramp_down_flag():
    settings = Settings(
        mode="Simulation Mode",
        start_date=pd.Timestamp("1985-01-01").date(),
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
        shiller_extension_mode="average_return",
        final_value_target=100_000,
        final_value_target_ramp_down=True,
    )

    assert settings.to_dict()["final_value_target_ramp_down"] is True
    round_tripped = Settings.from_dict(settings.to_dict())
    assert round_tripped.final_value_target_ramp_down is True
    session_state = {}
    settings.apply_to_session_state(session_state)
    assert session_state["final_value_target_ramp_down"] is True


def test_final_value_target_ramp_down_defaults_to_true_in_settings_state():
    session_state = {}
    Settings(
        mode="Simulation Mode",
        start_date=pd.Timestamp("1985-01-01").date(),
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
        shiller_extension_mode="average_return",
    ).apply_to_session_state(session_state)

    assert session_state["final_value_target_ramp_down"] is True


def test_settings_uses_loky_only_for_historical_parallel():
    settings = Settings(
        mode="Historical Mode",
        start_date=pd.Timestamp("1985-01-01").date(),
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

    assert "historical_parallel_backend" not in settings.to_dict()
    session_state = {}
    settings.apply_to_session_state(session_state)
    assert "historical_parallel_backend" not in session_state


def test_settings_ignores_legacy_historical_parallel_backend_field():
    settings = Settings.from_dict(
        {
            "mode": "Historical Mode",
            "start_date": "1985-01-01",
            "retirement_duration_months": 12,
            "analysis_start_date": "1871-01-01",
            "initial_value": 1_000_000,
            "stock_pct": 0.9,
            "target_success_rate": 0.8,
            "initial_monthly_spending": 4000,
            "initial_spending_overridden": False,
            "upper_guardrail_success": 1.0,
            "lower_guardrail_success": 0.25,
            "upper_adjustment_fraction": 1.0,
            "lower_adjustment_fraction": 0.35,
            "adjustment_threshold": 0.05,
            "adjustment_frequency": "Monthly",
            "spending_cap_option": "Unlimited",
            "spending_floor_option": "Unlimited",
            "shiller_extension_mode": "hard_code_date",
            "historical_parallel_backend": "threading",
        }
    )

    assert settings.shiller_extension_mode == "hard_code_date"
    assert "historical_parallel_backend" not in settings.to_dict()


def test_missing_cape_values_default_to_15():
    df = pd.DataFrame(
        {
            "Date": ["1871.01", "1871.02"],
            "Real Total Return Price": [100.0, 101.0],
            "Real Total Bond Returns": [50.0, 50.5],
            "CAPE": [pd.NA, 16.0],
            "TR CAPE": [None, 17.0],
        }
    )

    result = shiller_utils._fill_missing_cape_values(df)

    assert result.loc[0, "CAPE"] == 15
    assert result.loc[0, "TR CAPE"] == 15
    assert result.loc[1, "CAPE"] == 16
    assert result.loc[1, "TR CAPE"] == 17


def test_extension_mode_changes_synthetic_tail():
    dates = pd.date_range(start="2006-01-01", periods=36, freq="MS")
    stock_prices = [100.0 + i for i in range(20)] + [60.0, 55.0, 50.0, 45.0, 40.0, 38.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.0, 27.0]
    bond_prices = [50.0 + i * 0.5 for i in range(36)]
    cape_values = [18.0 + i * 0.2 for i in range(20)] + [25.0, 24.0, 22.0, 20.0, 18.0, 17.0, 16.5, 16.0, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0, 12.8, 12.5]
    tr_cape_values = [20.0 + i * 0.25 for i in range(20)] + [28.0, 27.0, 25.0, 23.0, 21.0, 20.0, 19.0, 18.5, 18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 15.2, 15.0]
    df = pd.DataFrame(
        {
            "Date": dates,
            "Real Total Return Price": stock_prices,
            "Real Total Bond Returns": bond_prices,
            "CAPE": cape_values,
            "TR CAPE": tr_cape_values,
        }
    )

    end_date = pd.Timestamp("2011-12-01")
    hard_code = shiller_utils.extend_shiller_data_to_cover_date(
        df,
        end_date=end_date,
        extension_mode="hard_code_date",
    )

    assert len(hard_code) > len(df)
    assert hard_code["Real Total Return Price"].iloc[len(df)] < hard_code["Real Total Return Price"].iloc[len(df) - 1]
    assert hard_code["CAPE"].iloc[len(df)] != hard_code["CAPE"].iloc[len(df) - 1]
    assert hard_code["TR CAPE"].iloc[len(df)] != hard_code["TR CAPE"].iloc[len(df) - 1]
    assert hard_code["Real Total Return Price"].iloc[len(df) + 12] != hard_code["Real Total Return Price"].iloc[len(df)]
