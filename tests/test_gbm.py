import numpy as np
import pandas as pd
import pytest

import src.backtest.replay as rp
import src.models.gbm as gbm

DAY = pd.Timestamp("2024-12-10")   # the delivery day most tests forecast
FAST = {"n_estimators": 30, "num_leaves": 7, "objective": "l2"}   # a small model, so the tests stay quick


def make_features(start="2024-08-01", end="2025-01-07", seed=0):
    """A stand-in copy of the features table: 24 rows a day, the price, and a few clues from each of the four groups."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", name="delivery_time")
    n = len(hours)
    table = pd.DataFrame(index=hours)
    table["load_fc_nl"] = 12000 + 2500 * np.sin((hours.hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 300, n)
    table["load_fc_de"] = 4.6 * table["load_fc_nl"] + rng.normal(0, 900, n)
    table["sea_ijmuiden_wind_speed_100m"] = rng.gamma(2.0, 4.0, n)
    table["de_bilt_shortwave_radiation"] = np.clip(np.sin((hours.hour - 6) / 12 * np.pi), 0, None) * 600
    table.insert(0, "price", 60 + 0.01 * (table["load_fc_nl"] - 12000) - 2 * table["sea_ijmuiden_wind_speed_100m"]
                 - 0.03 * table["de_bilt_shortwave_radiation"] + rng.normal(0, 4, n))
    table["price_lag_24"] = table["price"].shift(24)
    table["price_lag_168"] = table["price"].shift(168)
    table["hour"] = hours.hour
    table["day_of_week"] = hours.day_of_week
    table["month"] = hours.month
    table["is_holiday"] = 0
    table["wind_index"] = table["sea_ijmuiden_wind_speed_100m"]
    table["sun_index"] = table["de_bilt_shortwave_radiation"]
    return table.dropna()                                     # as in Step 4: the first week has no seven-day lag

def test_the_model_learns_only_from_priced_hours_inside_the_window():
    known = rp.known_before(make_features(), DAY)
    rows = gbm.learning_rows(known, DAY, window=30)
    assert rows.index.max() == pd.Timestamp("2024-12-09 23:00")         # nothing from the day itself
    assert rows.index.min() == pd.Timestamp("2024-11-10 00:00")         # nothing older than the window
    assert len(rows) == 30 * 24 and rows["price"].notna().all()         # and every example has its price


def test_the_model_finds_a_pattern_that_last_week_cannot():
    table = make_features()
    days = rp.days_from(table, DAY)[:5]
    actual = table.loc[table.index.normalize().isin(days), "price"]
    forecast = rp.replay(table, days, gbm.make_gbm(FAST))
    naive = rp.replay(table, days, rp.naive_day)
    assert forecast.index.equals(actual.index)                          # 24 answers for each day, like any forecaster
    assert rp.mae(actual, forecast) < 0.7 * rp.mae(actual, naive)       # it has learned how the clues move the price


def test_the_same_settings_give_the_same_forecast():
    known = rp.known_before(make_features(), DAY)
    first = gbm.make_gbm(FAST)(known, DAY)
    again = gbm.make_gbm(FAST)(known, DAY)
    assert np.array_equal(first, again)


def test_the_model_only_reads_the_columns_it_is_given():
    known = rp.known_before(make_features(), DAY)
    changed = known.copy()
    for wind in ["sea_ijmuiden_wind_speed_100m", "wind_index"]:
        changed[wind] = changed[wind].to_numpy()[::-1]                  # scramble the wind clues
    short = ["load_fc_nl", "hour"]                                      # a list that leaves the wind out
    assert np.array_equal(gbm.make_gbm(FAST, columns=short)(known, DAY), gbm.make_gbm(FAST, columns=short)(changed, DAY))
    assert not np.array_equal(gbm.make_gbm(FAST)(known, DAY), gbm.make_gbm(FAST)(changed, DAY))

def test_there_are_twelve_candidates_with_the_simplest_first():
    tried = gbm.candidates()
    assert len(tried) == 12
    assert tried[0] == {"n_estimators": 300, "num_leaves": 15, "objective": "l2"}
    assert {"n_estimators": 300, "num_leaves": 31, "objective": "l2"} in tried      # the tool's own defaults
    assert len({gbm.name_of(settings) for settings in tried}) == 12                 # every name is different
    assert gbm.name_of(tried[0]) == "trees300_leaves15_l2"


def test_the_practice_exam_stops_before_the_real_one():
    days = gbm.practice_days(make_features())
    assert days[0] == pd.Timestamp("2024-09-01") and days[-1] == pd.Timestamp("2024-12-31")
    assert len(days) == 122
    assert (days < pd.Timestamp(gbm.TEST_START)).all()

def test_every_entrant_is_replayed_on_the_same_days():
    table = make_features()
    days = rp.days_from(table, DAY)[:3]
    forecasts = gbm.try_out(table, days, {"zeros": lambda known, day: np.zeros(24),
                                          "hours": lambda known, day: np.arange(24)})
    assert list(forecasts.columns) == ["price", "naive", "zeros", "hours"]
    assert forecasts.index.equals(table.index[table.index.normalize().isin(days)])
    assert forecasts["price"].equals(table.loc[forecasts.index, "price"])
    assert (forecasts["hours"].to_numpy() == np.tile(np.arange(24), 3)).all()


def test_the_lowest_error_wins_and_a_tie_goes_to_the_first():
    scores = pd.DataFrame({"mae": [1.0, 5.0, 3.0, 3.0]}, index=["naive", "plain", "middle", "fancy"])
    assert gbm.choose(scores) == "middle"                               # never the baseline, and not the later of two equals

def test_every_clue_lands_in_one_group_and_the_ladder_only_grows():
    columns = ["price", "load_fc_nl", "load_fc_de", "de_bilt_temperature_2m", "bavaria_shortwave_radiation",
               "price_lag_24", "price_lag_168", "hour", "day_of_week", "month", "is_holiday", "wind_index", "sun_index"]
    groups = gbm.clue_groups(columns)
    assert list(groups) == ["lags", "calendar", "load", "weather"]
    assert groups["weather"] == ["de_bilt_temperature_2m", "bavaria_shortwave_radiation", "wind_index", "sun_index"]
    assert sorted(sum(groups.values(), [])) == sorted(columns[1:])      # every clue once, and never the price

    rungs = gbm.ladder(groups)
    assert list(rungs) == ["lags", "+calendar", "+load", "+weather"]
    assert [len(members) for members in rungs.values()] == [2, 6, 8, 12]
    assert rungs["+calendar"][:2] == rungs["lags"]                      # each rung keeps everything below it


def test_the_first_run_picks_a_winner_and_keeps_every_forecast(tmp_path):
    table = make_features()
    days = rp.days_from(table, DAY)[:3]
    small = {"n_estimators": (20,), "num_leaves": (4, 8), "objective": ("l2",)}
    winner, settings, tuning, ablation = gbm.practice(table, days, grid=small, folder=tmp_path)
    assert list(tuning.index) == ["naive", "trees20_leaves4_l2", "trees20_leaves8_l2"]
    assert winner == tuning.drop(index="naive")["mae"].idxmin() and gbm.name_of(settings) == winner
    assert list(ablation.index) == ["naive", "lags", "+calendar", "+load", "+weather"]
    assert ablation.loc["+weather", "mae"] == tuning.loc[winner, "mae"]             # the top rung is the winner itself
    assert len(list(tmp_path.glob("*.csv"))) == 5                                   # two candidates and three lower rungs

def test_the_settings_file_comes_back_as_written_and_guards_the_window(tmp_path, monkeypatch):
    path = tmp_path / "config" / "model.json"
    days = pd.date_range("2024-09-01", "2024-12-31", freq="D")
    written = gbm.write_settings(path, "trees30_leaves7_l2", FAST, days, 12.3456)
    record = gbm.read_settings(path)
    assert record == written
    assert record["settings"] == FAST and record["practice_mae"] == 12.346
    assert record["chosen_on"] == "2024-09-01 to 2024-12-31" and record["practice_days"] == 122

    monkeypatch.setattr(gbm, "WINDOW", 100)                             # as if the code changed after the freeze
    with pytest.raises(AssertionError, match="window of 364 days"):
        gbm.read_settings(path)
    
def make_forecasts():
    """Two days of made-up prices with three forecasts whose errors are known exactly."""
    hours = pd.date_range("2025-06-01", periods=48, freq="h", name="delivery_time")
    price = pd.Series(50.0, index=hours)
    price[hours.hour == 13] = -5.0                                      # one negative hour a day
    table = pd.DataFrame({"price": price})
    table["naive"] = price + 20
    table["lear"] = price + np.where(hours.day == 1, 4, 8)              # worse on the second day
    table["gbm"] = price + np.where(hours.hour.isin(gbm.EVENING), 10, 2)   # worse in the evening peak
    return table


def test_errors_are_cut_by_hour_month_and_regime():
    slices = gbm.error_slices(make_forecasts())
    assert list(slices.columns) == ["cut", "slice", "hours", "naive", "lear", "gbm"]
    assert slices["cut"].value_counts().to_dict() == {"hour": 24, "regime": 3, "all": 1, "month": 1}

    whole = slices[slices["cut"] == "all"].iloc[0]
    assert whole["hours"] == 48 and whole["naive"] == 20 and whole["lear"] == 6
    regime = slices[slices["cut"] == "regime"].set_index("slice")
    assert regime["hours"].to_dict() == {"evening peak": 8, "negative price": 2, "other hours": 38}
    assert regime.loc["evening peak", "gbm"] == 10 and regime.loc["other hours", "gbm"] == 2
    hour = slices[slices["cut"] == "hour"].set_index("slice")
    assert hour.loc[18, "gbm"] == 10 and hour.loc[3, "gbm"] == 2


def test_the_worst_days_come_first():
    worst = gbm.worst_days(make_forecasts(), name="lear", n=1)
    assert list(worst.index) == [pd.Timestamp("2025-06-02")]            # the day lear missed by 8, not the day it missed by 4
    assert worst.index.name == "delivery_date"
    assert worst.loc["2025-06-02", "lear"] == 8 and worst.loc["2025-06-02", "lowest_price"] == -5
    both = gbm.worst_days(make_forecasts(), name="lear")
    assert list(both["lear"]) == [8, 4]

def test_the_significance_test_is_handed_whole_days_in_the_right_order():
    forecasts = make_forecasts()
    handed = []

    def fake_test(real, first, second):
        handed.append((real, first, second))
        return 0.01 if len(handed) == 1 else 0.99

    challenger_p, reference_p = gbm.edge(forecasts, test=fake_test)
    assert (challenger_p, reference_p) == (0.01, 0.99)
    assert all(array.shape == (2, 24) for call in handed for array in call)         # one row per day
    lear, own = gbm.as_days(forecasts["lear"]), gbm.as_days(forecasts["gbm"])
    assert (handed[0][1] == lear).all() and (handed[0][2] == own).all()             # first: is gbm better than lear?
    assert (handed[1][1] == own).all() and (handed[1][2] == lear).all()             # then: is lear better than gbm?

    assert gbm.verdict(0.01, 0.99).startswith("gbm is significantly more accurate than lear")
    assert gbm.verdict(0.99, 0.01).startswith("lear is significantly more accurate than gbm")
    assert gbm.verdict(0.40, 0.60).startswith("no significant difference")