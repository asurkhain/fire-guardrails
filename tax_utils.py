import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from joblib import dump, load

import numba as nb
import numpy as np
import pandas as pd

from app_settings import Settings
from cape_utils import calculate_cape_withdrawal_rate, simulate_cape_withdrawal_retirement
from vpw_utils import _apply_vpw_comparison_step, simulate_vpw_withdrawal_retirement
from shiller_utils import *  # noqa: F401,F403


# 2026 federal brackets supplied for the tax-mode calculation.  Thresholds are
# the lower edge of each bracket, which makes them convenient for progressive
# tax calculations.
ORDINARY_INCOME_BRACKETS = {
    "single": ((0.0, 0.10), (12_400.0, 0.12), (50_400.0, 0.22),
               (105_700.0, 0.24), (201_775.0, 0.32), (256_225.0, 0.35),
               (640_600.0, 0.37)),
    "mfj": ((0.0, 0.10), (24_800.0, 0.12), (100_800.0, 0.22),
            (211_400.0, 0.24), (403_550.0, 0.32), (512_450.0, 0.35),
            (768_700.0, 0.37)),
}
STANDARD_DEDUCTIONS = {"single": 16_100.0, "mfj": 32_200.0}
NIIT_THRESHOLDS = {"single": 200_000.0, "mfj": 250_000.0}
NIIT_RATE = 0.038
FPL_400_THRESHOLDS = {"single": 63_840.0, "mfj": 86_560.0}
LTCG_BRACKETS = {
    "single": ((0.0, 0.00), (49_450.0, 0.15), (545_500.0, 0.20)),
    "mfj": ((0.0, 0.00), (98_900.0, 0.15), (613_700.0, 0.20)),
}


@dataclass
class PortfolioAccount:
    """A tax-mode account balance.

    ``total`` is the current market value. ``cost_basis`` is used for taxable
    accounts and for the basis-first phase of post-tax retirement accounts.
    """

    account_type: str
    asset_type: str
    cost_basis: float
    total: float
    account_id: str = ""


def _canonical_marital_status(marital_status: str) -> str:
    value = str(marital_status).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"single": "single", "s": "single", "mfj": "mfj", "married_filing_jointly": "mfj"}
    if value not in aliases:
        raise ValueError("Marital status must be 'single' or 'mfj'.")
    return aliases[value]


def _canonical_account_type(account_type: str) -> str:
    value = str(account_type).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "taxable": "taxable",
        "retirement_pretax": "retirement_pretax",
        "pretax": "retirement_pretax",
        "pre_tax": "retirement_pretax",
        "traditional": "retirement_pretax",
        "retirement_post_tax": "retirement_post_tax",
        "posttax": "retirement_post_tax",
        "post_tax": "retirement_post_tax",
        "roth": "retirement_post_tax",
    }
    if value not in aliases:
        raise ValueError("Account type must be taxable, retirement pretax, or retirement post tax.")
    return aliases[value]


def _canonical_asset_type(asset_type: str) -> str:
    value = str(asset_type).strip().lower()
    if value in ("equity ex-us", "equity ex us", "equity_ex_us", "ex_us", "ex-us"):
        return "equity ex-US"
    if value in ("equity", "stock", "stocks"):
        return "equity"
    if value in ("bond", "bonds"):
        return "bond"
    raise ValueError("Asset type must be equity, equity ex-US, or bond.")


def normalize_accounts(accounts: Iterable[PortfolioAccount | Mapping[str, Any]]) -> list[PortfolioAccount]:
    """Validate and copy account input into the internal account representation."""

    normalized: list[PortfolioAccount] = []
    for index, raw_account in enumerate(accounts):
        if isinstance(raw_account, PortfolioAccount):
            raw_account = raw_account.__dict__
        if not isinstance(raw_account, Mapping):
            raise ValueError(f"Account {index + 1} must be an account object.")
        try:
            account = PortfolioAccount(
                account_type=_canonical_account_type(raw_account["account_type"]),
                asset_type=_canonical_asset_type(raw_account["asset_type"]),
                cost_basis=float(raw_account["cost_basis"]),
                total=float(raw_account["total"]),
                account_id=str(raw_account.get("account_id", raw_account.get("id", index))),
            )
        except KeyError as exc:
            raise ValueError(f"Account {index + 1} is missing required field {exc.args[0]!r}.") from exc
        if not np.isfinite(account.total) or account.total < 0.0:
            raise ValueError(f"Account {index + 1} total must be a non-negative finite number.")
        if not np.isfinite(account.cost_basis) or account.cost_basis < 0.0:
            raise ValueError(f"Account {index + 1} cost basis must be a non-negative finite number.")
        normalized.append(account)
    return normalized


def _taxable_account_balance(accounts: Iterable[PortfolioAccount]) -> float:
    """Return the total balance across all taxable accounts."""

    return sum(
        float(account.total)
        for account in accounts
        if account.account_type == "taxable"
    )


def settle_taxable_bond_gains(
    accounts: Iterable[PortfolioAccount | Mapping[str, Any]],
) -> dict[str, Any]:
    """Recognize accrued taxable-bond gains now and add them to cost basis.

    This is the tax mode's simplified bond rule: positive return in a taxable
    bond account is ordinary investment income in the period it occurs, rather
    than income deferred until the account is sold.  The recognized gain is
    added to basis, preventing the same return from being taxed again later.
    """

    working_accounts = normalize_accounts(accounts)
    ordinary_income = 0.0
    settlements: list[dict[str, float | str]] = []
    for account in working_accounts:
        if account.account_type != "taxable" or account.asset_type != "bond":
            continue
        gain = max(0.0, account.total - account.cost_basis)
        if gain > 0.0:
            account.cost_basis += gain
            ordinary_income += gain
            settlements.append({
                "account_id": account.account_id,
                "account_type": account.account_type,
                "asset_type": account.asset_type,
                "amount": gain,
                "ordinary_income": gain,
            })
    return {
        "accounts": working_accounts,
        "taxable_bond_income": ordinary_income,
        "settlements": settlements,
    }


def apply_account_return_multipliers(
    accounts: Iterable[PortfolioAccount | Mapping[str, Any]],
    equity_return_multiplier: float,
    bond_return_multiplier: float,
    ex_us_equity_return_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Apply one period's market returns and recognize taxable-bond income.

    Return values are multipliers, matching :func:`compute_portfolio_returns`:
    ``1.05`` represents a 5% return. Taxable bond appreciation is settled
    immediately; equity gains remain unrealized until withdrawal.
    """

    equity_multiplier = max(0.0, float(equity_return_multiplier))
    bond_multiplier = max(0.0, float(bond_return_multiplier))
    ex_us_equity_multiplier = max(0.0, float(ex_us_equity_return_multiplier))
    updated_accounts = normalize_accounts(accounts)
    for account in updated_accounts:
        if account.asset_type == "equity":
            multiplier = equity_multiplier
        elif account.asset_type == "equity ex-US":
            multiplier = ex_us_equity_multiplier
        else:
            multiplier = bond_multiplier
        account.total *= multiplier
    return settle_taxable_bond_gains(updated_accounts)


def rebalance_retirement_accounts(
    accounts: Iterable[PortfolioAccount | Mapping[str, Any]], target_stock_pct: float
) -> dict[str, Any]:
    """Best-effort rebalance retirement holdings while leaving taxable accounts intact.

    When more bonds are needed, pre-tax equity is converted to bonds before
    post-tax equity. When more equity is needed, post-tax bonds are converted
    first. A partial conversion creates (or adds to) the matching holding in
    the same account type; no taxable-account trades are made.
    """

    target_stock_pct = float(target_stock_pct)
    if not 0.0 <= target_stock_pct <= 1.0:
        raise ValueError("Target stock percentage must be between 0 and 1.")
    working_accounts = normalize_accounts(accounts)
    total_value = sum(account.total for account in working_accounts)
    current_equity = sum(account.total for account in working_accounts if account.asset_type in ("equity", "equity ex-US"))
    target_equity = total_value * target_stock_pct
    amount_to_reclassify = abs(target_equity - current_equity)
    target_asset = "equity" if target_equity > current_equity else "bond"

    if amount_to_reclassify <= 1e-9:
        return {
            "accounts": working_accounts,
            "rebalanced_amount": 0.0,
            "target_stock_pct": target_stock_pct,
            "achieved_stock_pct": current_equity / total_value if total_value else 0.0,
            "transactions": [],
        }

    source_asset = "bond" if target_asset == "equity" else "equity"
    # Required priority: bonds go to pre-tax first; equity goes to post-tax first.
    account_priority = (
        ("retirement_post_tax", "retirement_pretax")
        if target_asset == "equity"
        else ("retirement_pretax", "retirement_post_tax")
    )
    transactions: list[dict[str, float | str]] = []

    def add_destination(account_type: str, account_id: str, amount: float, basis: float) -> None:
        for account in working_accounts:
            if (
                account.account_type == account_type
                and account.account_id == account_id
                and account.asset_type == target_asset
            ):
                account.total += amount
                account.cost_basis += basis
                return
        working_accounts.append(PortfolioAccount(
            account_type=account_type,
            asset_type=target_asset,
            cost_basis=basis,
            total=amount,
            account_id=account_id,
        ))

    remaining = amount_to_reclassify
    for account_type in account_priority:
        for source in list(working_accounts):
            if remaining <= 1e-9:
                break
            if source.account_type != account_type or source.asset_type != source_asset or source.total <= 0.0:
                continue
            amount = min(remaining, source.total)
            basis_transfer = source.cost_basis * amount / source.total
            source.total -= amount
            source.cost_basis = max(0.0, source.cost_basis - basis_transfer)
            add_destination(account_type, source.account_id, amount, basis_transfer)
            transactions.append({
                "account_id": source.account_id,
                "account_type": account_type,
                "from_asset_type": source_asset,
                "to_asset_type": target_asset,
                "amount": amount,
            })
            remaining -= amount
        if remaining <= 1e-9:
            break

    working_accounts = [account for account in working_accounts if account.total > 1e-9]
    achieved_equity = sum(account.total for account in working_accounts if account.asset_type in ("equity", "equity ex-US"))
    return {
        "accounts": working_accounts,
        "rebalanced_amount": amount_to_reclassify - remaining,
        "target_stock_pct": target_stock_pct,
        "achieved_stock_pct": achieved_equity / total_value if total_value else 0.0,
        "transactions": transactions,
    }


def _progressive_tax(income: float, brackets: tuple[tuple[float, float], ...]) -> float:
    income = max(0.0, float(income))
    tax = 0.0
    for index, (lower, rate) in enumerate(brackets):
        upper = brackets[index + 1][0] if index + 1 < len(brackets) else np.inf
        tax += max(0.0, min(income, upper) - lower) * rate
    return tax


def calculate_fpl_health_expense(total_income: float, marital_status: str, age: float | None) -> float:
    """Return the annual under-65 health expense triggered above 400% FPL.

    The supplied values define a linear age curve from $8,000 at age 42 to
    $21,140 at age 64. Ages outside that range use the nearest endpoint, which
    avoids extrapolating an implausible expense for younger retirees.
    """

    if age is None or float(age) >= 65.0:
        return 0.0
    status = _canonical_marital_status(marital_status)
    if float(total_income) <= FPL_400_THRESHOLDS[status]:
        return 0.0
    bounded_age = min(64.0, max(42.0, float(age)))
    individual_expense = 8_000.0 + (bounded_age - 42.0) * (21_140.0 - 8_000.0) / (64.0 - 42.0)
    return individual_expense * (2.0 if status == "mfj" else 1.0)


def calculate_federal_income_tax(
    ordinary_income: float,
    long_term_capital_gains: float,
    marital_status: str,
    net_investment_income: float | None = None,
    age: float | None = None,
) -> dict[str, float]:
    """Calculate federal income tax after the standard deduction.

    Long-term gains are stacked above taxable ordinary income, as required by
    the supplied LTCG brackets. NIIT is 3.8% of the lesser of net investment
    income and MAGI above the status-specific NIIT threshold.  This model uses
    ordinary income plus long-term gains as MAGI, without rare MAGI adjustments.
    """

    status = _canonical_marital_status(marital_status)
    ordinary = max(0.0, float(ordinary_income))
    gains = max(0.0, float(long_term_capital_gains))
    # Long-term capital gains are investment income by definition here.  Callers
    # can supply bond interest/gains or other investment income explicitly.
    investment_income = gains if net_investment_income is None else max(0.0, float(net_investment_income))
    deduction = STANDARD_DEDUCTIONS[status]
    taxable_ordinary = max(ordinary - deduction, 0.0)
    deduction_left = max(deduction - ordinary, 0.0)
    taxable_gains = max(gains - deduction_left, 0.0)
    ordinary_tax = _progressive_tax(taxable_ordinary, ORDINARY_INCOME_BRACKETS[status])

    # LTCG brackets apply to total taxable income.  Tax only the incremental
    # gains, starting at the taxable ordinary-income level.
    gains_tax = (
        _progressive_tax(taxable_ordinary + taxable_gains, LTCG_BRACKETS[status])
        - _progressive_tax(taxable_ordinary, LTCG_BRACKETS[status])
    )
    modified_adjusted_gross_income = ordinary + gains
    niit_threshold = NIIT_THRESHOLDS[status]
    niit_taxable_income = min(
        investment_income,
        max(0.0, modified_adjusted_gross_income - niit_threshold),
    )
    niit = niit_taxable_income * NIIT_RATE
    fpl_health_expense = calculate_fpl_health_expense(modified_adjusted_gross_income, status, age)
    return {
        "ordinary_income": ordinary,
        "long_term_capital_gains": gains,
        "standard_deduction": deduction,
        "taxable_ordinary_income": taxable_ordinary,
        "taxable_long_term_capital_gains": taxable_gains,
        "ordinary_income_tax": ordinary_tax,
        "long_term_capital_gains_tax": gains_tax,
        "net_investment_income": investment_income,
        "modified_adjusted_gross_income": modified_adjusted_gross_income,
        "niit_threshold": niit_threshold,
        "niit_taxable_income": niit_taxable_income,
        "net_investment_income_tax": niit,
        "fpl_health_expense": fpl_health_expense,
        "total_tax": ordinary_tax + gains_tax + niit,
        "total_tax_and_fpl_health_expense": ordinary_tax + gains_tax + niit + fpl_health_expense,
    }


def _withdraw_from_account(account: PortfolioAccount, amount: float, phase: str) -> dict[str, float | str]:
    """Withdraw an amount from one account, returning its taxable character."""

    amount = min(max(float(amount), 0.0), account.total)
    if amount <= 0.0:
        return {}
    gain = max(account.total - account.cost_basis, 0.0)
    ordinary_income = 0.0
    long_term_capital_gains = 0.0
    if account.account_type == "taxable" and account.total > 0.0:
        if account.asset_type in ("equity", "equity ex-US"):
            realized_gain = amount * gain / account.total
            long_term_capital_gains = realized_gain
            account.cost_basis = max(0.0, account.cost_basis - (amount - realized_gain))
        else:
            # Taxable bond returns are settled as ordinary investment income as
            # they arise and added to basis, so a later withdrawal is principal.
            account.cost_basis = max(0.0, account.cost_basis - amount)
    elif account.account_type == "retirement_pretax":
        ordinary_income = amount
        # Cost basis does not affect tax treatment of a pre-tax account.
        account.cost_basis = max(0.0, account.cost_basis - min(account.cost_basis, amount))
    else:  # post-tax retirement distributions are tax-free
        if phase == "post_tax_basis":
            account.cost_basis = max(0.0, account.cost_basis - amount)
    account.total = max(0.0, account.total - amount)
    return {
        "account_id": account.account_id,
        "account_type": account.account_type,
        "asset_type": account.asset_type,
        "phase": phase,
        "amount": amount,
        "ordinary_income": ordinary_income,
        "long_term_capital_gains": long_term_capital_gains,
    }


def allocate_account_withdrawal(
    accounts: Iterable[PortfolioAccount | Mapping[str, Any]],
    gross_withdrawal: float,
    fpl_income_limit: float | None = None,
) -> dict[str, Any]:
    """Allocate a gross withdrawal using the tax-mode account order.

    Taxable accounts are drawn as newly recognized bond-return cash, equity
    sales up to ``fpl_income_limit``, then taxable-bond basis. The retirement
    order is post-tax basis, pre-tax, then post-tax growth. The returned
    accounts are copies, leaving the caller's input unchanged.
    """

    remaining = max(0.0, float(gross_withdrawal))
    bond_settlement = settle_taxable_bond_gains(accounts)
    working_accounts = bond_settlement["accounts"]
    transactions: list[dict[str, float | str]] = []
    settled_bond_gains = {
        str(item["account_id"]): float(item["amount"])
        for item in bond_settlement["settlements"]
    }
    remaining_fpl_income = (
        np.inf if fpl_income_limit is None else max(0.0, float(fpl_income_limit))
    )
    remaining_fpl_income = max(
        0.0, remaining_fpl_income - float(bond_settlement["taxable_bond_income"])
    )

    # The just-recognized taxable bond return is used first. Its income was
    # recorded by the settlement step, so this cash withdrawal has no new tax.
    for account in working_accounts:
        if remaining <= 1e-9:
            break
        if account.account_type == "taxable" and account.asset_type == "bond":
            bond_gain_cash = settled_bond_gains.get(account.account_id, 0.0)
            transaction = _withdraw_from_account(account, min(remaining, bond_gain_cash), "taxable_bond_gain")
            if transaction:
                transactions.append(transaction)
                remaining -= float(transaction["amount"])

    # Realize equity gains only until the income room below 400% FPL is used.
    for account in working_accounts:
        if remaining <= 1e-9 or remaining_fpl_income <= 1e-9:
            break
        if account.account_type != "taxable" or account.asset_type not in ("equity", "equity ex-US") or account.total <= 0.0:
            continue
        gain_ratio = (max(account.total - account.cost_basis, 0.0) / account.total) if account.total > 0.0 else 0.0
        maximum_equity_draw = account.total if gain_ratio <= 0.0 else remaining_fpl_income / gain_ratio
        transaction = _withdraw_from_account(
            account,
            min(remaining, maximum_equity_draw, account.total),
            "taxable_equity_to_fpl_limit",
        )
        if transaction:
            transactions.append(transaction)
            remaining -= float(transaction["amount"])
            remaining_fpl_income = max(
                0.0, remaining_fpl_income - float(transaction["long_term_capital_gains"])
            )

    # Check if drawing taxable bond basis will allow staying under the FPL cap.
    # If an FPL limit is in effect and remaining withdrawal exceeds available taxable bond basis,
    # then drawing bond basis will not prevent exceeding FPL (since subsequent equity/pretax draws
    # will push income over FPL anyway). In that case, preserve bond basis and draw taxable equity first.
    available_taxable_bond_basis = sum(
        account.total for account in working_accounts
        if account.account_type == "taxable" and account.asset_type == "bond"
    )

    can_stay_under_fpl_with_bond_basis = (
        fpl_income_limit is None
        or fpl_income_limit == np.inf
        or remaining <= available_taxable_bond_basis + 1e-9
    )

    if can_stay_under_fpl_with_bond_basis:
        for account in working_accounts:
            if remaining <= 1e-9:
                break
            if account.account_type == "taxable" and account.asset_type == "bond" and account.total > 0:
                transaction = _withdraw_from_account(account, min(remaining, account.total), "taxable_bond_basis")
                if transaction:
                    transactions.append(transaction)
                    remaining -= float(transaction["amount"])

    # The FPL target is a preference, not a spending cap. If the lower-income
    # sources are exhausted (or cannot prevent crossing FPL), equity sales continue to fund the withdrawal.
    for account in working_accounts:
        if remaining <= 1e-9:
            break
        if account.account_type == "taxable" and account.asset_type in ("equity", "equity ex-US"):
            transaction = _withdraw_from_account(account, min(remaining, account.total), "taxable_equity_after_fpl_limit")
            if transaction:
                transactions.append(transaction)
                remaining -= float(transaction["amount"])

    # If bond basis was deferred because it couldn't keep us under FPL, but taxable equity
    # is now exhausted and we still have remaining withdrawal before retirement accounts:
    if not can_stay_under_fpl_with_bond_basis:
        for account in working_accounts:
            if remaining <= 1e-9:
                break
            if account.account_type == "taxable" and account.asset_type == "bond" and account.total > 0:
                transaction = _withdraw_from_account(account, min(remaining, account.total), "taxable_bond_basis")
                if transaction:
                    transactions.append(transaction)
                    remaining -= float(transaction["amount"])

    retirement_phases = (
        ("post_tax_basis", lambda account: account.account_type == "retirement_post_tax", lambda account: min(account.cost_basis, account.total)),
        ("pretax", lambda account: account.account_type == "retirement_pretax", lambda account: account.total),
        ("post_tax_growth", lambda account: account.account_type == "retirement_post_tax", lambda account: max(account.total - account.cost_basis, 0.0)),
    )
    for phase, eligible, available in retirement_phases:
        for account in working_accounts:
            if remaining <= 1e-9:
                break
            if eligible(account):
                transaction = _withdraw_from_account(account, min(remaining, available(account)), phase)
                if transaction:
                    transactions.append(transaction)
                    remaining -= float(transaction["amount"])
        if remaining <= 1e-9:
            break

    withdrawal_ordinary_income = sum(float(item["ordinary_income"]) for item in transactions)
    ordinary_income = float(bond_settlement["taxable_bond_income"]) + withdrawal_ordinary_income
    gains = sum(float(item["long_term_capital_gains"]) for item in transactions)
    withdrawn = max(0.0, float(gross_withdrawal) - remaining)
    return {
        "accounts": working_accounts,
        "transactions": transactions,
        "gross_withdrawal": withdrawn,
        "ordinary_income": ordinary_income,
        "taxable_bond_income": float(bond_settlement["taxable_bond_income"]),
        "taxable_bond_settlements": bond_settlement["settlements"],
        "long_term_capital_gains": gains,
        "unfunded_withdrawal": remaining,
    }


def fund_spending_with_tax(
    accounts: Iterable[PortfolioAccount | Mapping[str, Any]],
    spending: float,
    marital_status: str,
    other_ordinary_income: float = 0.0,
    other_long_term_capital_gains: float = 0.0,
    other_net_investment_income: float = 0.0,
    current_age: float | None = None,
    target_stock_pct: float | None = None,
) -> dict[str, Any]:
    """Determine the account draw needed to deliver ``spending`` after tax.

    The method solves for the smallest gross draw under the required account
    order.  It returns updated account copies, detailed withdrawals, federal
    tax (including NIIT), FPL-triggered health expense, and any spending
    shortfall if the portfolio is exhausted. ``other_net_investment_income``
    covers investment income that is not represented by account withdrawals.
    """

    target = max(0.0, float(spending))
    status = _canonical_marital_status(marital_status)
    base_ordinary = max(0.0, float(other_ordinary_income))
    base_gains = max(0.0, float(other_long_term_capital_gains))
    base_investment_income = max(0.0, float(other_net_investment_income)) + base_gains
    source_accounts = normalize_accounts(accounts)
    maximum_draw = sum(account.total for account in source_accounts)
    fpl_income_limit = None
    if current_age is not None and float(current_age) < 65.0:
        fpl_income_limit = max(
            0.0,
            FPL_400_THRESHOLDS[status] - base_ordinary - base_gains,
        )

    def result_for(draw: float) -> dict[str, Any]:
        working = [
            PortfolioAccount(
                account_type=a.account_type,
                asset_type=a.asset_type,
                cost_basis=a.cost_basis,
                total=a.total,
                account_id=a.account_id,
            )
            for a in source_accounts
        ]
        allocation = allocate_account_withdrawal(
            working,
            draw,
            fpl_income_limit=fpl_income_limit,
        )
        tax = calculate_federal_income_tax(
            base_ordinary + allocation["ordinary_income"],
            base_gains + allocation["long_term_capital_gains"],
            status,
            net_investment_income=(
                base_investment_income
                + allocation["taxable_bond_income"]
                + allocation["long_term_capital_gains"]
            ),
            age=current_age,
        )
        allocation["tax"] = tax
        allocation["after_tax_cash_before_fpl_health_expense"] = allocation["gross_withdrawal"] - tax["total_tax"]
        allocation["after_tax_cash"] = (
            allocation["after_tax_cash_before_fpl_health_expense"] - tax["fpl_health_expense"]
        )
        return allocation

    maximum_result = result_for(maximum_draw)
    if maximum_result["after_tax_cash"] < target:
        result = maximum_result
    else:
        # The FPL expense is a one-time step up. First see if spending can be
        # funded below the threshold, where the normal bisection is monotonic.
        low, high = 0.0, maximum_draw
        for _ in range(80):
            midpoint = (low + high) / 2.0
            if result_for(midpoint)["after_tax_cash_before_fpl_health_expense"] >= target:
                high = midpoint
            else:
                low = midpoint
        no_expense_candidate = result_for(high)
        if no_expense_candidate["tax"]["fpl_health_expense"] == 0.0:
            result = no_expense_candidate
        else:
            # The no-expense solution is above the FPL threshold, so every
            # feasible solution must fund the health expense as well.
            low, high = 0.0, maximum_draw
            for _ in range(80):
                midpoint = (low + high) / 2.0
                if result_for(midpoint)["after_tax_cash"] >= target:
                    high = midpoint
                else:
                    low = midpoint
            result = result_for(high)

    result["requested_spending"] = target
    result["funded_spending"] = min(target, max(0.0, result["after_tax_cash"]))
    result["spending_shortfall"] = max(0.0, target - result["funded_spending"])
    if target_stock_pct is not None:
        rebalancing = rebalance_retirement_accounts(result["accounts"], target_stock_pct)
        result["accounts"] = rebalancing["accounts"]
        result["rebalancing"] = rebalancing
    return result


def add_tax_metrics_to_withdrawal_results(
    results_df: pd.DataFrame,
    accounts: Iterable[PortfolioAccount | Mapping[str, Any]],
    marital_status: str,
    current_age: float | None,
    target_stock_pct: float | None = None,
    market_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach tax-inclusive draw and tax-rate diagnostics to a simulation path.

    The guardrail engine supplies the monthly net cash need. This helper funds
    it from the configured accounts in sequence and records the resulting tax
    and gross draw, allowing tax-aware simulation charts without changing the
    established guardrail controls.
    """

    enriched = results_df.copy()
    if enriched.empty:
        return enriched

    dates = pd.to_datetime(enriched["Date"])
    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]

    # Pre-calculate monthly market returns if market_data is provided
    stock_returns: np.ndarray = np.ones(len(enriched))
    bond_returns: np.ndarray = np.ones(len(enriched))
    ex_us_returns: np.ndarray = np.ones(len(enriched))
    if market_data is not None and not market_data.empty:
        m_df = market_data.copy()
        m_df["Date"] = pd.to_datetime(m_df["Date"])
        cols = ["Date", "Real Total Return Price", "Real Total Bond Returns"]
        if "Real Total Ex-US Return Price" in m_df.columns:
            cols.append("Real Total Ex-US Return Price")
        merged = pd.merge(
            enriched[["Date"]].copy(),
            m_df[cols],
            on="Date",
            how="left"
        )
        s_prices = merged["Real Total Return Price"].ffill().bfill().to_numpy(dtype=float)
        b_prices = merged["Real Total Bond Returns"].ffill().bfill().to_numpy(dtype=float)
        if len(s_prices) > 1:
            stock_returns[:-1] = np.where(s_prices[:-1] > 0.0, s_prices[1:] / s_prices[:-1], 1.0)
            bond_returns[:-1] = np.where(b_prices[:-1] > 0.0, b_prices[1:] / b_prices[:-1], 1.0)
        if "Real Total Ex-US Return Price" in merged.columns:
            ex_prices = merged["Real Total Ex-US Return Price"].ffill().bfill().to_numpy(dtype=float)
            if len(ex_prices) > 1:
                ex_us_returns[:-1] = np.where(ex_prices[:-1] > 0.0, ex_prices[1:] / ex_prices[:-1], 1.0)

    strategies = (
        ("", "Withdrawal", "Total_Spending"),
        ("Fixed_", "Fixed_SR_Withdrawal", "Fixed_SR_Total_Spending"),
        ("CAPE_", "CAPE_Withdrawal", "CAPE_Total_Spending"),
        ("VPW_", "VPW_Withdrawal", "VPW_Total_Spending"),
    )

    for prefix, withdrawal_column, spending_column in strategies:
        if withdrawal_column not in enriched.columns:
            continue
        working_accounts = normalize_accounts(accounts)
        taxes: list[float] = []
        aca_surcharges: list[float] = []
        gross_draws: list[float] = []
        tax_rates: list[float] = []
        taxable_balances: list[float] | None = [] if prefix == "" else None
        track_taxable_balances = (
            prefix == ""
            and _taxable_account_balance(working_accounts) > 1e-9
        )
        if spending_column in enriched.columns:
            cashflow = pd.to_numeric(enriched.get("Net_Cashflow", 0.0), errors="coerce").fillna(0.0)
            spending_needs = (pd.to_numeric(enriched[spending_column], errors="coerce").fillna(0.0) - cashflow).clip(lower=0.0)
        else:
            cashflow = pd.Series(0.0, index=enriched.index)
            spending_needs = pd.to_numeric(enriched[withdrawal_column], errors="coerce").fillna(0.0)

        working_df = pd.DataFrame({
            "Date": dates,
            "_spending": spending_needs,
            "_cashflow": cashflow,
            "_orig_idx": np.arange(len(enriched)),
        })
        working_df["_year"] = working_df["Date"].dt.year
        start_year = int(working_df["_year"].iloc[0])

        for year, year_group in working_df.groupby("_year", sort=False):
            num_months_in_year = len(year_group)
            year_spending_total = float(year_group["_spending"].sum())
            year_cashflow_total = float(year_group["_cashflow"].sum())
            sim_age = None if current_age is None else float(current_age) + (int(year) - start_year)

            if year_spending_total <= 1e-9:
                annual_tax = 0.0
                annual_aca = 0.0
                annual_gross_draw = 0.0
                annual_tax_rate = 0.0
            else:
                funding = fund_spending_with_tax(
                    working_accounts,
                    year_spending_total,
                    marital_status,
                    other_ordinary_income=year_cashflow_total,
                    current_age=sim_age,
                    target_stock_pct=target_stock_pct,
                )

                annual_tax = float(funding["tax"]["total_tax"])
                annual_aca = float(funding["tax"]["fpl_health_expense"])
                annual_gross_draw = float(funding["gross_withdrawal"])
                annual_tax_rate = annual_tax / annual_gross_draw if annual_gross_draw > 0.0 else 0.0

            for monthly_need in year_group["_spending"]:
                if year_spending_total > 1e-9:
                    weight = max(0.0, float(monthly_need)) / year_spending_total
                else:
                    weight = 1.0 / num_months_in_year if num_months_in_year > 0 else 0.0

                m_gross = annual_gross_draw * weight
                m_tax = annual_tax * weight
                m_aca = annual_aca * weight

                gross_draws.append(m_gross)
                taxes.append(m_tax)
                aca_surcharges.append(m_aca)
                tax_rates.append(annual_tax_rate)

            # Monthly simulation of accounts
            gross_factor = annual_gross_draw / year_spending_total if year_spending_total > 0.0 else 1.0
            for idx in year_group["_orig_idx"]:
                m_spend = float(spending_needs.iloc[idx])
                m_gross = m_spend * gross_factor
                working_accounts = allocate_account_withdrawal(working_accounts, m_gross)["accounts"]
                s_ret = float(stock_returns[idx])
                b_ret = float(bond_returns[idx])
                ex_us_ret = float(ex_us_returns[idx])
                for acc in working_accounts:
                    if acc.asset_type == "equity":
                        acc.total *= s_ret
                    elif acc.asset_type == "equity ex-US":
                        acc.total *= ex_us_ret
                    else:
                        acc.total *= b_ret

                if track_taxable_balances and taxable_balances is not None:
                    taxable_balances.append(_taxable_account_balance(working_accounts))

            if target_stock_pct is not None:
                working_accounts = rebalance_retirement_accounts(working_accounts, target_stock_pct)["accounts"]

        enriched[f"{prefix}Tax_Paid"] = taxes
        enriched[f"{prefix}ACA_Surcharge"] = aca_surcharges
        enriched[f"{prefix}Tax_Inclusive_Withdrawal"] = gross_draws
        enriched[f"{prefix}Tax_Rate"] = tax_rates
        if track_taxable_balances and taxable_balances is not None:
            enriched["Taxable_Accounts_Balance"] = taxable_balances
    return enriched


def summarize_tax_metrics_from_results(
    results_df: pd.DataFrame,
    prefix: str = "",
    initial_portfolio: float | None = None,
    current_age: float | None = None,
) -> dict[str, float | int | None]:
    """Compute lifetime tax and ACA summary metrics for one strategy path."""

    tax_col = f"{prefix}Tax_Paid"
    aca_col = f"{prefix}ACA_Surcharge"
    gross_col = f"{prefix}Tax_Inclusive_Withdrawal"
    if tax_col not in results_df.columns or results_df.empty:
        return {}

    total_federal_tax = float(pd.to_numeric(results_df[tax_col], errors="coerce").fillna(0.0).sum())
    total_aca = float(pd.to_numeric(results_df.get(aca_col, 0.0), errors="coerce").fillna(0.0).sum())
    total_tax_and_aca = total_federal_tax + total_aca
    total_gross = float(pd.to_numeric(results_df.get(gross_col, 0.0), errors="coerce").fillna(0.0).sum())

    lifetime_tax_rate = total_federal_tax / total_gross if total_gross > 0.0 else 0.0

    if initial_portfolio is None:
        if "Portfolio_Value" in results_df.columns:
            withdrawal = float(results_df.get("Withdrawal", pd.Series([0.0])).iloc[0])
            initial_portfolio = float(results_df["Portfolio_Value"].iloc[0]) + withdrawal
        else:
            initial_portfolio = None

    num_years = 1
    if "Date" in results_df.columns:
        num_years = max(1, len(pd.to_datetime(results_df["Date"]).dt.year.unique()))

    avg_annual_tax_pct = (
        (total_tax_and_aca / num_years) / float(initial_portfolio)
        if initial_portfolio is not None and initial_portfolio > 0.0
        else None
    )

    pre65_subsidized = 0
    pre65_cliff = 0
    medicare_transition_year: int | None = None
    pre65_lifetime_tax_rate: float | None = None
    pre65_avg_annual_tax_pct: float | None = None
    if "Date" in results_df.columns and current_age is not None:
        dates = pd.to_datetime(results_df["Date"])
        start_year = int(dates.iloc[0].year)
        start_month = int(dates.iloc[0].month)
        months_elapsed = (dates.dt.year - start_year) * 12 + (dates.dt.month - start_month)
        ages = float(current_age) + months_elapsed / 12.0
        pre65_mask = ages < 65.0
        if pre65_mask.any():
            pre65_tax_and_aca = float(
                pd.to_numeric(results_df.loc[pre65_mask, tax_col], errors="coerce").fillna(0.0).sum()
            )
            pre65_gross = float(
                pd.to_numeric(results_df.loc[pre65_mask, gross_col], errors="coerce").fillna(0.0).sum()
            )
            pre65_lifetime_tax_rate = (
                pre65_tax_and_aca / pre65_gross if pre65_gross > 0.0 else 0.0
            )
            if initial_portfolio is not None and initial_portfolio > 0.0:
                pre65_years = max(1, len(dates.loc[pre65_mask].dt.year.unique()))
                pre65_avg_annual_tax_pct = (
                    (pre65_tax_and_aca / pre65_years) / float(initial_portfolio)
                )

        if aca_col in results_df.columns:
            annual_aca = pd.DataFrame({
                "Year": dates.dt.year,
                "ACA_Surcharge": pd.to_numeric(results_df[aca_col], errors="coerce").fillna(0.0),
            }).groupby("Year", as_index=False)["ACA_Surcharge"].sum()

            for _, year_row in annual_aca.iterrows():
                year = int(year_row["Year"])
                age = float(current_age) + (year - start_year)
                aca_amount = float(year_row["ACA_Surcharge"])
                if age < 65.0:
                    if aca_amount <= 0.0:
                        pre65_subsidized += 1
                    else:
                        pre65_cliff += 1
                elif medicare_transition_year is None and aca_amount <= 0.0:
                    medicare_transition_year = year

    min_actual_wd = float(pd.to_numeric(results_df[gross_col], errors="coerce").min() * 12.0) if gross_col in results_df.columns and not results_df.empty else None
    max_actual_wd = float(pd.to_numeric(results_df[gross_col], errors="coerce").max() * 12.0) if gross_col in results_df.columns and not results_df.empty else None
    median_actual_wd = float(pd.to_numeric(results_df[gross_col], errors="coerce").median() * 12.0) if gross_col in results_df.columns and not results_df.empty else None
    avg_actual_wd = float(pd.to_numeric(results_df[gross_col], errors="coerce").mean() * 12.0) if gross_col in results_df.columns and not results_df.empty else None

    return {
        "Total_Tax_Paid": total_federal_tax,
        "Total_ACA_Surcharge": total_aca,
        "Total_Gross_Withdrawal": total_gross,
        "Lifetime_Tax_Rate": lifetime_tax_rate,
        "Pre65_Lifetime_Tax_Rate": pre65_lifetime_tax_rate,
        "Avg_Annual_Tax_Pct": avg_annual_tax_pct,
        "Pre65_Avg_Annual_Tax_Pct": pre65_avg_annual_tax_pct,
        "Pre65_Subsidized_Years": pre65_subsidized,
        "Pre65_Cliff_Years": pre65_cliff,
        "Medicare_Transition_Year": medicare_transition_year,
        "Min_Actual_Withdrawal": min_actual_wd,
        "Max_Actual_Withdrawal": max_actual_wd,
        "Median_Actual_Withdrawal": median_actual_wd,
        "Average_Actual_Withdrawal": avg_actual_wd,
    }


TAX_SUMMARY_METRIC_KEYS = (
    "Total_Tax_Paid",
    "Total_ACA_Surcharge",
    "Total_Gross_Withdrawal",
    "Lifetime_Tax_Rate",
    "Pre65_Lifetime_Tax_Rate",
    "Avg_Annual_Tax_Pct",
    "Pre65_Avg_Annual_Tax_Pct",
    "Pre65_Subsidized_Years",
    "Pre65_Cliff_Years",
    "Medicare_Transition_Year",
    "Min_Actual_Withdrawal",
    "Max_Actual_Withdrawal",
    "Median_Actual_Withdrawal",
    "Average_Actual_Withdrawal",
)

TAX_SUMMARY_STRATEGY_PREFIXES = ("", "Fixed_", "CAPE_", "VPW_")


def apply_strategy_tax_summaries_to_row(
    row: dict,
    enriched_results: pd.DataFrame,
    initial_portfolio: float,
    current_age: float | None,
) -> None:
    """Attach per-strategy lifetime tax and ACA summary metrics to one historical row."""

    for prefix in TAX_SUMMARY_STRATEGY_PREFIXES:
        tax_col = f"{prefix}Tax_Paid"
        if tax_col not in enriched_results.columns:
            continue
        try:
            summary = summarize_tax_metrics_from_results(
                enriched_results,
                prefix=prefix,
                initial_portfolio=initial_portfolio,
                current_age=current_age,
            )
        except (ValueError, TypeError):
            summary = {}
        for key in TAX_SUMMARY_METRIC_KEYS:
            col_name = f"{prefix}{key}" if prefix else key
            row[col_name] = summary.get(key)


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
            'Fixed_SR_Total_Spending': 0.0 if fixed_depleted else fixed_total_spending,
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

    if "Guardrail_Success" in results_df.columns:
        success_col = results_df["Guardrail_Success"].astype(bool)
        if not success_col.all():
            first_failure = success_col.idxmin()
            results_df = results_df.loc[:first_failure]

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


def _withdrawal_stats_from_results(results_df: pd.DataFrame) -> dict:
    """Compute monthly withdrawal statistics from a guardrail simulation dataframe."""
    if results_df is None or results_df.empty:
        return {
            "min_withdrawal": np.nan,
            "max_withdrawal": np.nan,
            "median_withdrawal": np.nan,
            "avg_withdrawal": np.nan,
        }

    if "Guardrail_Success" in results_df.columns:
        success_col = results_df["Guardrail_Success"].astype(bool)
        if not success_col.all():
            first_failure = success_col.idxmin()
            results_df = results_df.loc[:first_failure]

    if "Tax_Inclusive_Withdrawal" in results_df.columns:
        withdrawal = results_df["Tax_Inclusive_Withdrawal"].astype(float)
    else:
        withdrawal = results_df["Withdrawal"].astype(float)

    return {
        "min_withdrawal": float(withdrawal.min()),
        "max_withdrawal": float(withdrawal.max()),
        "median_withdrawal": float(withdrawal.median()),
        "avg_withdrawal": float(withdrawal.mean()),
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
    withdrawal_stats = _withdrawal_stats_from_results(results_df)

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
        **withdrawal_stats,
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
        row["CAPE_Min_Actual_Withdrawal"] = cape_result.get("min_actual_withdrawal")
        row["CAPE_Max_Actual_Withdrawal"] = cape_result.get("max_actual_withdrawal")
        row["CAPE_Average_Actual_Withdrawal"] = cape_result.get("avg_actual_withdrawal")
        row["CAPE_Median_Actual_Withdrawal"] = cape_result.get("median_actual_withdrawal")
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
        row["VPW_Min_Actual_Withdrawal"] = vpw_result.get("min_actual_withdrawal")
        row["VPW_Max_Actual_Withdrawal"] = vpw_result.get("max_actual_withdrawal")
        row["VPW_Average_Actual_Withdrawal"] = vpw_result.get("avg_actual_withdrawal")
        row["VPW_Median_Actual_Withdrawal"] = vpw_result.get("median_actual_withdrawal")
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
        row["CAPE_Min_Actual_Withdrawal"] = None
        row["CAPE_Max_Actual_Withdrawal"] = None
        row["CAPE_Average_Actual_Withdrawal"] = None
        row["CAPE_Median_Actual_Withdrawal"] = None
        row["CAPE_Pct_Below_Fixed"] = None
        row["VPW_Success"] = None
        row["VPW_Ending_Portfolio"] = None
        row["VPW_Minimum_Withdrawal"] = None
        row["VPW_Maximum_Withdrawal"] = None
        row["VPW_Average_Withdrawal"] = None
        row["VPW_Median_Withdrawal"] = None
        row["VPW_Min_Actual_Withdrawal"] = None
        row["VPW_Max_Actual_Withdrawal"] = None
        row["VPW_Average_Actual_Withdrawal"] = None
        row["VPW_Median_Actual_Withdrawal"] = None
        row["VPW_Pct_Below_Fixed"] = None

    if settings.accounts or settings.mode in ("Taxable Mode", "Taxable Historical Mode"):
        try:
            normalized_accounts = normalize_accounts(settings.accounts)
            enriched_results = add_tax_metrics_to_withdrawal_results(
                results_df,
                normalized_accounts,
                settings.marital_status,
                settings.current_age,
                target_stock_pct=settings.stock_pct,
                market_data=df,
            )
            row.update(_withdrawal_stats_from_results(enriched_results))
            apply_strategy_tax_summaries_to_row(
                row,
                enriched_results,
                initial_portfolio=float(settings.initial_value),
                current_age=settings.current_age,
            )
        except (ValueError, TypeError):
            pass

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
    d = Path.cwd() / "temp"
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
