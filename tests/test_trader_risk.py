import pytest

from src.trader.risk import RiskSettings, build_trade_plan


def settings():
    return RiskSettings(10_000, 0.5, 2.0, "USD")


def test_buy_plan_has_correct_levels_and_conservative_lot_rounding():
    plan = build_trade_plan(direction="BUY", entry=1.1, stop_distance=.002,
                            reward_multiple=2, settings=settings(), quote_to_account_rate=1)
    assert plan.stop_loss == pytest.approx(1.098)
    assert plan.take_profit == pytest.approx(1.104)
    assert plan.risk_reward == 2
    assert plan.risk_amount == 50
    assert plan.position_lots == pytest.approx(.25)


def test_sell_plan_reverses_stop_and_target():
    plan = build_trade_plan(direction="SELL", entry=150, stop_distance=.5,
                            reward_multiple=1.5, settings=settings(), quote_to_account_rate=.01)
    assert plan.stop_loss == 150.5
    assert plan.take_profit == 149.25
    assert plan.position_lots == pytest.approx(.1)


def test_refuses_missing_currency_conversion_and_exposure_over_cap():
    with pytest.raises(ValueError, match="quote_to_account_rate"):
        build_trade_plan(direction="BUY", entry=1.1, stop_distance=.001,
                         reward_multiple=2, settings=settings())
    with pytest.raises(ValueError, match="maximum open-risk"):
        build_trade_plan(direction="BUY", entry=1.1, stop_distance=.001,
                         reward_multiple=2, settings=settings(), quote_to_account_rate=1,
                         existing_open_risk_percent=1.6)


def test_rejects_invalid_parameters_and_subminimum_size():
    with pytest.raises(ValueError):
        build_trade_plan(direction="HOLD", entry=1.1, stop_distance=.001,
                         reward_multiple=2, settings=settings(), quote_to_account_rate=1)
    with pytest.raises(ValueError, match="below the minimum"):
        build_trade_plan(direction="BUY", entry=1.1, stop_distance=100,
                         reward_multiple=2, settings=settings(), quote_to_account_rate=1)
