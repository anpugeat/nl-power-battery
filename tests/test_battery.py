import numpy as np
import pandas as pd
import pytest

import src.dispatch.battery as bat

DIP = [12, 13, 14]      # the three cheap hours of the stand-in day
PEAK = [18, 19]         # its two dear hours


def make_day(base=50.0, dip=10.0, peak=100.0):
    """24 made-up prices: flat, with three cheap hours at midday and two dear hours in the evening."""
    prices = np.full(24, float(base))
    prices[DIP] = dip
    prices[PEAK] = peak
    return prices


def make_forecasts(days=6, seed=0, missing=()):
    """A stand-in copy of the forecasts table: prices with a midday dip and an evening peak, and three forecasts of them."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-06-01", periods=24 * days, freq="h", name="delivery_time")
    shape = 60 - 50 * np.exp(-((hours.hour - 13) ** 2) / 8) + 60 * np.exp(-((hours.hour - 19) ** 2) / 6)
    table = pd.DataFrame({"price": shape + rng.normal(0, 8, len(hours))}, index=hours)
    table["naive"] = table["price"] + rng.normal(0, 30, len(hours))     # a poor forecast
    table["lear"] = table["price"] + rng.normal(0, 12, len(hours))      # a better one
    table["gbm"] = table["price"] + rng.normal(0, 6, len(hours))        # the best of the three
    return table[~table.index.normalize().isin(pd.to_datetime(list(missing)))]

def test_flat_prices_are_not_worth_trading():
    charge, discharge = bat.plan_battery(np.full(24, 50.0))
    assert charge.sum() == 0 and discharge.sum() == 0


def test_a_small_gap_between_cheap_and_dear_is_not_worth_the_losses():
    charge, discharge = bat.plan_battery(make_day(dip=50, peak=55))     # selling at 55 does not repay buying at 50
    assert charge.sum() == 0 and discharge.sum() == 0
    charge, discharge = bat.plan_battery(make_day(dip=50, peak=60))     # selling at 60 does
    assert discharge.sum() > 0


def test_it_buys_in_the_dip_and_sells_at_the_peak():
    charge, discharge = bat.plan_battery(make_day())
    assert list(np.flatnonzero(charge)) == DIP and list(np.flatnonzero(discharge)) == PEAK
    assert charge.sum() == pytest.approx(2 / 0.94)                      # it buys a little more than it can hold
    assert discharge.sum() == pytest.approx(2 * 0.94)                   # and gets a little less back out


def test_it_never_charges_and_discharges_in_the_same_hour():
    paid_to_take = np.array([-50.0] * 6 + [40.0] * 18)                  # six hours in which you are paid to take power
    for battery in [bat.BATTERY, {**bat.BATTERY, "max_cycles": 3.0}]:   # the second may fill three times a day
        charge, discharge = bat.plan_battery(paid_to_take, battery)
        assert np.minimum(charge, discharge).max() == 0
    charge, discharge = bat.plan_battery(paid_to_take)
    assert charge.sum() == pytest.approx(2 / 0.94)                      # with one cycle a day it fills once and no more


def test_wear_makes_it_trade_less_and_extra_cycles_let_it_trade_more():
    two_dips = make_day()
    two_dips[[3, 4, 5]], two_dips[[8, 9]] = 10, 100                     # a second cheap spell and a second dear one
    once = bat.plan_battery(two_dips)[1].sum()
    twice = bat.plan_battery(two_dips, {**bat.BATTERY, "max_cycles": 2.0})[1].sum()
    assert twice == pytest.approx(2 * once)
    worn = bat.plan_battery(make_day(), {**bat.BATTERY, "wear": 50.0})[1].sum()
    assert worn == 0                                                    # at 50 EUR per MWh moved, the trade no longer pays

def test_every_plan_stays_inside_the_limits():
    rng = np.random.default_rng(1)
    batteries = [bat.BATTERY, {**bat.BATTERY, "capacity": 4.0}, {**bat.BATTERY, "eta_c": 0.9, "eta_d": 0.9},
                 {**bat.BATTERY, "max_cycles": 2.0}, {**bat.BATTERY, "wear": 20.0}]
    for battery in batteries:
        for _ in range(10):
            prices = rng.normal(40, 60, 24)                             # wild days, with negative prices among them
            charge, discharge = bat.plan_battery(prices, battery)
            assert bat.broken_limits(charge, discharge, battery) == []


def test_a_plan_that_breaks_a_limit_is_named():
    def plan(charge=(), discharge=()):
        amounts = np.zeros((2, 24))
        for row, entries in enumerate([charge, discharge]):
            for hour, amount in entries:
                amounts[row, hour] = amount
        return amounts[0], amounts[1]

    assert bat.broken_limits(*plan()) == []                                                 # doing nothing is allowed
    assert bat.broken_limits(*plan(charge=[(2, 1.0)], discharge=[(5, 0.8836)])) == []       # one honest trade is too
    assert bat.broken_limits(*plan(charge=[(2, 1.0)], discharge=[(2, 0.8836)])) == ["charges and discharges in the same hour"]
    assert bat.broken_limits(*plan(charge=[(2, 2.0)], discharge=[(5, 1.0), (6, 0.7672)])) == ["moves more than its power"]
    assert bat.broken_limits(*plan(charge=[(5, 1.0)], discharge=[(2, 0.8836)])) == ["goes below empty"]
    assert bat.broken_limits(*plan(charge=[(2, 1.0)])) == ["does not end the day empty"]
    assert "goes above full" in bat.broken_limits(*plan(charge=[(1, 1.0), (2, 1.0), (3, 1.0)], discharge=[(5, 1.0), (6, 1.0), (7, 0.6508)]))

def test_the_till_pays_the_real_prices():
    prices, charge, discharge = np.array([10.0, 100.0, 50.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])
    assert bat.settle(prices, charge, discharge) == 40                  # 0.5 sold at 100, less 1 bought at 10
    assert bat.settle(prices, charge, discharge, wear=5.0) == 32.5      # less 5 for each of the 1.5 MWh moved
    best = bat.settle(make_day(), *bat.plan_battery(make_day()))
    assert best == pytest.approx(2 * (0.94 * 100 - 10 / 0.94))          # worked out by hand for a full 2 MWh battery


def test_the_crystal_ball_is_never_beaten():
    rng = np.random.default_rng(2)
    for _ in range(20):
        real = rng.normal(40, 60, 24)
        guess = real + rng.normal(0, 40, 24)
        ceiling = bat.settle(real, *bat.plan_battery(real))
        assert bat.settle(real, *bat.plan_battery(guess)) <= ceiling + 1e-6
        assert ceiling >= 0                                             # at worst, the crystal ball does nothing


def test_a_forecast_that_is_upside_down_loses_money():
    upside_down = make_day(dip=100, peak=10)                            # it expects the dear hours to be the cheap ones
    assert bat.settle(make_day(), *bat.plan_battery(upside_down)) < 0

def test_every_player_plans_every_day_and_is_paid_once_a_day():
    table = make_forecasts(missing=["2025-06-03"])
    plans, profits = bat.play_all(table)
    assert list(plans.columns) == ["price", "perfect", "gbm", "lear", "naive"]
    assert plans.index.equals(table.index) and plans.notna().all().all()
    assert (plans.drop(columns="price").abs() <= 1.0).all().all()       # never more than the battery's power, either way
    assert plans.loc["2025-06-01 13:00", "perfect"] < 0 < plans.loc["2025-06-01 19:00", "perfect"]   # buys in the dip, sells at the peak

    assert profits.index.name == "delivery_date" and len(profits) == 5
    assert pd.Timestamp("2025-06-03") not in profits.index              # the missing day is skipped, not invented
    assert (profits.drop(columns="perfect").sub(profits["perfect"], axis=0) <= 1e-6).all().all()

    with pytest.raises(AssertionError, match="does not have 24 rows"):
        bat.play(table.iloc[:-1], "gbm")


def test_a_player_who_sees_the_real_prices_earns_the_ceiling():
    table = make_forecasts(days=3)
    table["copy"] = table["price"]
    plans, profits = bat.play_all(table, players={"perfect": "price", "copy": "copy", "naive": "naive"})
    assert plans["copy"].equals(plans["perfect"]) and profits["copy"].equals(profits["perfect"])
    assert profits["naive"].sum() < profits["perfect"].sum()

def make_profits():
    """Two days of made-up daily profits, small enough to check by hand."""
    days = pd.DatetimeIndex(["2025-06-01", "2025-06-02"], name="delivery_date")
    return pd.DataFrame({"perfect": [100.0, 100.0], "gbm": [90.0, 60.0], "naive": [60.0, -20.0]}, index=days)


def test_the_scoreboard_adds_up():
    scores = bat.money_table(make_profits(), capacity=2.0)
    assert scores["profit"].to_dict() == {"perfect": 200, "gbm": 150, "naive": 40}
    assert scores["per_mwh_year"].to_dict() == {"perfect": 18250, "gbm": 13687.5, "naive": 3650}   # total / 2 days * 365 / 2 MWh
    assert scores["capture_rate"].to_dict() == {"perfect": 1, "gbm": 0.75, "naive": 0.2}
    assert scores["losing_days"].to_dict() == {"perfect": 0, "gbm": 0, "naive": 1}


def test_the_headline_names_the_numbers():
    profits = make_profits()
    scores = bat.money_table(profits, capacity=2.0)
    sentence = bat.headline(scores, profits.index)
    assert "Over 2 days from 2025-06-01 to 2025-06-02" in sentence
    assert "captured 75%" in sentence and "EUR 20,075 a year more than the naive forecast" in sentence
    assert "1 MW and 2 MWh" in sentence
    assert "a year less than the gbm forecast" in bat.headline(scores, profits.index, name="naive", baseline="gbm")

def test_the_costliest_days_come_first():
    worst = bat.worst_days(make_profits(), name="gbm", n=1)
    assert list(worst.index) == [pd.Timestamp("2025-06-02")]            # 40 short of the ceiling, against 10 the day before
    assert worst.loc["2025-06-02", "shortfall"] == 40 and worst.index.name == "delivery_date"
    assert list(bat.worst_days(make_profits(), name="naive")["shortfall"]) == [120, 40]