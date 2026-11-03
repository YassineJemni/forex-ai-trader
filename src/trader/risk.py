from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True)
class RiskSettings:
    account_balance: float
    risk_percent: float
    max_open_risk_percent: float
    account_currency: str = "USD"

    def __post_init__(self) -> None:
        if self.account_balance <= 0:
            raise ValueError("account_balance must be positive")
        if not 0 < self.risk_percent <= self.max_open_risk_percent <= 100:
            raise ValueError("risk percentages must satisfy 0 < per_trade <= max_open <= 100")
        if len(self.account_currency) != 3 or not self.account_currency.isalpha():
            raise ValueError("account_currency must be a three-letter currency code")


@dataclass(frozen=True)
class TradePlan:
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    risk_amount: float
    position_lots: float
    risk_percent: float


def build_trade_plan(*, direction: str, entry: float, stop_distance: float,
                     reward_multiple: float, settings: RiskSettings,
                     units_per_lot: float = 100_000.0,
                     quote_to_account_rate: float | None = None,
                     existing_open_risk_percent: float = 0.0,
                     lot_step: float = 0.01) -> TradePlan:
    """Calculate a conservative spot-FX plan; never sends an order.

    ``quote_to_account_rate`` converts P/L in the pair's quote currency to the
    account currency. Callers must provide it when currencies differ.
    """
    side = direction.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("direction must be BUY or SELL")
    if entry <= 0 or stop_distance <= 0 or reward_multiple <= 0:
        raise ValueError("entry, stop_distance and reward_multiple must be positive")
    if existing_open_risk_percent < 0:
        raise ValueError("existing_open_risk_percent cannot be negative")
    if existing_open_risk_percent + settings.risk_percent > settings.max_open_risk_percent:
        raise ValueError("trade would exceed the maximum open-risk limit")
    if units_per_lot <= 0 or lot_step <= 0:
        raise ValueError("units_per_lot and lot_step must be positive")
    if quote_to_account_rate is None or quote_to_account_rate <= 0:
        raise ValueError("a positive quote_to_account_rate is required for safe sizing")

    sign = 1 if side == "BUY" else -1
    stop = entry - sign * stop_distance
    target = entry + sign * stop_distance * reward_multiple
    risk_cash = settings.account_balance * settings.risk_percent / 100
    loss_per_lot = stop_distance * units_per_lot * quote_to_account_rate
    raw_lots = Decimal(str(risk_cash)) / Decimal(str(loss_per_lot))
    step = Decimal(str(lot_step))
    lots = (raw_lots / step).to_integral_value(rounding=ROUND_DOWN) * step
    if lots < lot_step:
        raise ValueError("calculated position is below the minimum lot step")
    lots = float(lots)
    return TradePlan(side, entry, stop, target, reward_multiple, risk_cash,
                     lots, settings.risk_percent)
