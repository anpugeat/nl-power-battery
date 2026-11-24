import numpy as np
import pandas as pd
import pytest

import src.dispatch.battery as bat
import src.doe.whatif as wi

FEW = {"duration": (1.0, 4.0), "efficiency": (0.85,), "wear": (0.0,), "cycles": (1.0,), "forecast": ("naive", "gbm")}   # four runs
TWO = {"duration": (1.0, 4.0), "forecast": ("naive", "gbm")}            # two factors, for sums that can be done by hand


def make_forecasts(start="2025-12-29", days=6, seed=0):
    """A stand-in copy of the forecasts table that runs across a new year: prices with a dip and a peak, and two forecasts."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=24 * days, freq="h", name="delivery_time")
    shape = 60 - 50 * np.exp(-((hours.hour - 13) ** 2) / 8) + 60 * np.exp(-((hours.hour - 19) ** 2) / 6)
    table = pd.DataFrame({"price": shape + rng.normal(0, 8, len(hours))}, index=hours)
    table["naive"] = table["price"] + rng.normal(0, 30, len(hours))     # a poor forecast
    table["gbm"] = table["price"] + rng.normal(0, 6, len(hours))        # a better one
    return table

def test_a_setting_becomes_a_battery_the_planner_accepts():
    battery = wi.battery_for({"duration": 4.0, "efficiency": 0.81, "wear": 20.0, "cycles": 2.0})
    assert battery == {"power": 1.0, "capacity": 4.0, "eta_c": 0.9, "eta_d": 0.9, "wear": 20.0, "max_cycles": 2.0}
    assert wi.battery_for(wi.BASE) == pytest.approx(bat.BATTERY)        # the base setting is Step 7's battery
    charge, discharge = bat.plan_battery(np.array([10.0] * 12 + [100.0] * 12), battery)
    assert charge.sum() == pytest.approx(4 / 0.9)                       # it fills all 4 MWh, through a 10% loss


def test_a_name_says_what_was_tried():
    setting = {"duration": 4.0, "efficiency": 0.92, "wear": 20.0, "cycles": 2.0}
    assert wi.name_of(setting) == "hours4_eff92_wear20_cycles2"
    assert wi.name_of({**setting, "forecast": "gbm"}) == "gbm_hours4_eff92_wear20_cycles2"
    assert wi.name_of({**wi.BASE, "duration": 1.75}) == "hours1.75_eff88.36_wear0_cycles1"


def test_the_plan_lists_every_combination_once():
    plan = wi.design()
    assert len(plan) == 32 and plan.index.is_unique and plan.index.name == "run"
    assert list(plan.columns) == ["duration", "efficiency", "wear", "cycles", "forecast"]
    assert plan.index[0] == "naive_hours1_eff80_wear0_cycles1" and plan.index[-1] == "gbm_hours4_eff92_wear20_cycles2"
    for factor, (low, high) in wi.LEVELS.items():
        assert (plan[factor] == low).sum() == 16 and (plan[factor] == high).sum() == 16   # every level is tried equally often
    assert len(wi.design(FEW)) == 4

def test_a_run_is_played_once_and_then_read_from_disk(tmp_path, monkeypatch):
    table, setting = make_forecasts(), {**wi.BASE, "forecast": "gbm"}
    played, real_play = [], bat.play
    monkeypatch.setattr(bat, "play", lambda *given: played.append(given[1]) or real_play(*given))

    first = wi.saved_play(table, setting, tmp_path)
    again = wi.saved_play(table, setting, tmp_path)
    assert played == ["gbm"] and again.equals(first)                    # the second time, the file was read
    assert first.index.name == "delivery_date" and len(first) == 6 and first.notna().all()

    changed = table.copy()
    changed.iloc[5, changed.columns.get_loc("gbm")] += 1.0              # one forecast moves by one euro
    wi.saved_play(changed, setting, tmp_path)
    assert played == ["gbm", "gbm"]                                     # so the old file no longer counts

    kept = tmp_path / f"{wi.name_of(setting)}_{wi.fingerprint(table[['price', 'gbm']])}.csv"
    kept.write_text("\n".join(kept.read_text().splitlines()[:3]) + "\n")    # a file cut short, as after a crash
    assert wi.saved_play(table, setting, tmp_path).equals(first) and len(played) == 3


def test_the_base_setting_plays_the_game_of_step_7(tmp_path):
    table = make_forecasts()
    here = wi.saved_play(table, {**wi.BASE, "forecast": "gbm"}, tmp_path)
    there = bat.play(table, "gbm")[1]
    assert here.to_numpy() == pytest.approx(there.to_numpy()) and list(here.index) == list(there.index)


def test_every_run_is_played_on_the_same_days(tmp_path):
    table = make_forecasts()
    plan = wi.design(FEW)
    daily = wi.play_design(table, plan, tmp_path)
    assert list(daily.columns) == list(plan.index) and daily.index.name == "delivery_date"
    assert len(daily) == 6 and daily.notna().all().all()
    assert len(list(tmp_path.glob("*.csv"))) == 4                       # one file per run

    seeing = wi.play_design(table, wi.design({**FEW, "forecast": ("price",)}), tmp_path)
    short, long = seeing["price_hours1_eff85_wear0_cycles1"], seeing["price_hours4_eff85_wear0_cycles1"]
    assert (long >= short - 1e-6).all() and long.sum() > short.sum()    # seeing the real prices, a bigger battery never earns less

def test_a_short_year_is_left_out():
    days = pd.date_range("2025-07-01", "2027-01-10", freq="D")          # 184 days, a whole year, then 10 days
    assert wi.long_years(days) == ([2025, 2026], [2027])
    assert wi.long_years(days, min_days=200) == ([2026], [2025, 2027])


def test_a_year_is_its_average_day_times_365():
    days = pd.DatetimeIndex(["2025-12-30", "2025-12-31", "2026-01-01"], name="delivery_date")
    daily = pd.DataFrame({"short": [10.0, 30.0, 5.0], "long": [0.0, 2.0, 1.0]}, index=days)
    per_year = wi.yearly(daily)
    assert per_year.loc[2025].to_dict() == {"short": 7300, "long": 365}     # 20 a day, and 1 a day
    assert per_year.loc[2026].to_dict() == {"short": 1825, "long": 365}

    plan = pd.DataFrame({"duration": [1.0, 4.0]}, index=pd.Index(["short", "long"], name="run"))
    runs = wi.run_table(plan, daily)
    assert len(runs) == 4 and list(runs.index.names) == ["run", "year"]
    assert runs.loc[("short", 2025)].to_dict() == {"duration": 1, "days": 2, "profit": 40, "per_year": 7300}

def make_plan():
    """Four made-up runs: a short and a long battery, each planned from two forecasts."""
    return pd.DataFrame({"duration": [1.0, 1.0, 4.0, 4.0], "forecast": ["naive", "gbm", "naive", "gbm"]},
                        index=pd.Index(["short_naive", "short_gbm", "long_naive", "long_gbm"], name="run"))


def make_daily(shake=0.0, seed=0):
    """Ten weeks of made-up daily profits for the four runs of make_plan: 10, 12, 30 and 32 EUR a day, plus noise."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=70, freq="D", name="delivery_date")
    steady = pd.DataFrame({"short_naive": 10.0, "short_gbm": 12.0, "long_naive": 30.0, "long_gbm": 32.0}, index=days)
    return steady + rng.normal(0, shake, steady.shape)


def test_the_signs_say_which_runs_are_high_and_which_are_low():
    plan = wi.design()
    signs = wi.contrasts(plan)
    assert len(signs.columns) == 15                                     # five factors and their ten pairs
    assert (signs.iloc[0, :5] == -1).all() and (signs.iloc[-1, :5] == 1).all()
    assert (signs.sum() == 0).all()                                     # every factor and pair is high in half the runs
    assert signs["duration x wear"].equals(signs["duration"] * signs["wear"])
    with pytest.raises(AssertionError, match="not one of its two levels"):
        wi.contrasts(plan.assign(wear=5.0))


def test_an_effect_is_the_high_runs_minus_the_low_runs():
    table = wi.effects(wi.yearly(make_daily()), make_plan(), TWO)
    assert table["effect"].to_dict() == pytest.approx({"duration": 7300, "forecast": 730, "duration x forecast": 0})
    assert table.index.name == "term" and list(table.columns) == ["effect", "in_2025"]

    signs = wi.contrasts(wi.design())
    one_year = 1000 + 150 * signs["duration"] - 25 * signs["wear"] + 10 * signs["duration x wear"]   # a made-up law
    another = 2000 + 250 * signs["duration"] - 25 * signs["wear"]                                    # a different year
    table = wi.effects(pd.DataFrame([one_year, another], index=[2025, 2026]), wi.design())
    assert table.loc["duration"].to_dict() == pytest.approx({"effect": 400, "in_2025": 300, "in_2026": 500})
    assert table.loc["duration x wear"].to_dict() == pytest.approx({"effect": 10, "in_2025": 20, "in_2026": 0})
    assert table.loc["wear", "effect"] == pytest.approx(-50)
    assert table.drop(index=["duration", "wear", "duration x wear"]).abs().max().max() < 1e-9      # nothing else moves it

def test_weeks_are_seven_days_and_stay_inside_their_year():
    days = pd.date_range("2025-12-20", "2026-01-10", freq="D").delete(3)    # 23 December is missing
    blocks = wi.week_blocks(days)
    assert list(blocks) == [2025, 2026]
    assert [len(week) for week in blocks[2025]] == [3, 7, 1] and [len(week) for week in blocks[2026]] == [7, 3]
    assert sorted(np.concatenate(blocks[2025] + blocks[2026])) == list(range(len(days)))   # every day is in exactly one week
    assert days[blocks[2026][0]][0] == pd.Timestamp("2026-01-01")


def test_a_steady_effect_has_no_spread_and_a_shaky_one_does():
    plan = make_plan()
    steady = wi.intervals(make_daily(), plan, TWO, draws=50)
    assert steady.loc["forecast"].to_dict() == pytest.approx({"low": 730, "high": 730})   # every week says the same thing

    shaky = make_daily(shake=5.0)
    wide = wi.intervals(shaky, plan, TWO, draws=200)
    narrow = wi.intervals(shaky, plan, TWO, draws=200, coverage=0.5)
    assert (wide["low"] < wide["high"]).all() and list(wide.index) == ["duration", "forecast", "duration x forecast"]
    assert (wide["low"] < narrow["low"]).all() and (narrow["high"] < wide["high"]).all()   # 90% of the draws need more room than 50%
    assert wide.equals(wi.intervals(shaky, plan, TWO, draws=200))                         # the same draws every time
    assert not wide.equals(wi.intervals(shaky, plan, TWO, draws=200, seed=1))

def make_effects(forecast=2000.0):
    """A made-up table of effects in round numbers, with an interval of 500 either side."""
    effect = pd.Series({"duration": 30000.0, "efficiency": 6000.0, "wear": -8000.0, "cycles": 1000.0, "forecast": forecast})
    return pd.DataFrame({"effect": effect, "low": effect - 500, "high": effect + 500})


def test_the_two_strongest_dials_are_picked_whatever_their_sign():
    table = make_effects()
    assert wi.strongest(table) == ["duration", "wear"]                  # wear counts, though its effect is negative
    assert wi.strongest(table, n=1) == ["duration"]
    table.loc["cycles", "effect"] = 99999.0
    assert wi.strongest(table) == ["duration", "wear"]                  # cycles cannot take in-between values, so it is never mapped


def test_the_map_tries_every_pair_of_values(tmp_path):
    points = wi.grid("duration", "wear", "gbm")
    assert len(points) == 25
    assert sorted({point["duration"] for point in points}) == [1, 1.75, 2.5, 3.25, 4]
    assert sorted({point["wear"] for point in points}) == [0, 5, 10, 15, 20]
    assert all(point["efficiency"] == wi.BASE["efficiency"] and point["cycles"] == 1 and point["forecast"] == "gbm" for point in points)

    area = wi.surface(make_forecasts(), "duration", "wear", "price", tmp_path, steps=2)   # four points, seen by the crystal ball
    assert list(area.columns) == ["duration", "wear", "per_year"] and len(area) == 4
    square = area.pivot(index="duration", columns="wear", values="per_year")
    assert (square.loc[4.0] >= square.loc[1.0]).all()                   # seeing the real prices, a bigger battery never earns less
    assert (square[0.0] >= square[20.0]).all()                          # and wear never helps

def test_the_sentence_names_the_numbers():
    text = wi.sentence(make_effects())
    assert "most sensitive to duration: going from 1 to 4 hours of storage changes it by EUR +30,000." in text
    assert "naive forecast to the gbm forecast adds EUR 2,000 a year (90% interval EUR 1,500 to 2,500)" in text
    assert text.endswith("as much as 4.0 more points of efficiency.")   # 6,000 for 12 points is 500 a point
    assert "more than raising efficiency from 80% to 92%" in wi.sentence(make_effects(forecast=9000.0))
    worse = wi.sentence(make_effects(forecast=-700.0))
    assert "costs EUR 700 a year" in worse and "points of efficiency" not in worse
    top_is_wear = make_effects()
    top_is_wear.loc["duration", "effect"] = 100.0
    assert "most sensitive to wear: going from EUR 0 to 20 of wear per MWh moved changes it by EUR -8,000." in wi.sentence(top_is_wear)