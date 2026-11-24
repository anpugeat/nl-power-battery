import numpy as np
import pandas as pd

import src.features.build_features as bf

TZ = "Europe/Amsterdam"
SPRING = "2025-03-30"   # the clocks go forward: 02:00 does not exist
AUTUMN = "2025-10-26"   # the clocks go back: 02:00 happens twice


def make_inputs(start="2025-03-01", end="2025-11-30", seed=0):
    """Stand-in copies of prices_utc, load_fc, weather_fc and the weather points."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ) + pd.Timedelta(hours=23),
                          freq="h").tz_convert("UTC").rename("time_utc")
    n = len(hours)
    prices = pd.DataFrame({"price": rng.normal(60, 30, n)}, index=hours)
    load = pd.DataFrame({"load_fc_nl": rng.normal(12000, 1500, n),
                         "load_fc_de": rng.normal(55000, 6000, n)}, index=hours)
    points = pd.DataFrame({"name": ["sea_ijmuiden", "groningen", "de_bilt"],
                           "role": ["wind", "wind", "sun"]})
    weather = pd.DataFrame(index=hours)
    for name in points["name"]:
        weather[f"{name}_wind_speed_100m"] = rng.gamma(2.0, 4.0, n)
        weather[f"{name}_shortwave_radiation"] = rng.uniform(0, 800, n)
        weather[f"{name}_temperature_2m"] = rng.normal(11, 6, n)
    return prices, load, weather, points

def test_grid_gives_every_day_24_rows_even_when_the_clocks_change():
    prices, _, _, _ = make_inputs()
    grid = bf.to_grid(prices)
    assert (grid.groupby(grid.index.normalize()).size() == 24).all()
    assert grid.notna().all().all()

    # an ordinary hour is carried over unchanged: 13:00 in Amsterdam on 1 June is 11:00 UTC
    assert grid.loc["2025-06-01 13:00", "price"] == prices.loc["2025-06-01 11:00+00:00", "price"]

    # autumn: the grid's 02:00 is the average of the two real 02:00 hours (00:00 and 01:00 UTC)
    both = prices.loc[f"{AUTUMN} 00:00+00:00":f"{AUTUMN} 01:00+00:00", "price"]
    assert np.isclose(grid.loc[f"{AUTUMN} 02:00", "price"], both.mean())

    # spring: the grid's 02:00 is filled from 01:00 and 03:00
    around = grid.loc[[f"{SPRING} 01:00", f"{SPRING} 03:00"], "price"]
    assert np.isclose(grid.loc[f"{SPRING} 02:00", "price"], around.mean())


def test_grid_leaves_a_real_gap_blank():
    prices, _, _, _ = make_inputs()
    holed = prices.drop(pd.Timestamp("2025-06-01 11:00", tz="UTC"))
    grid = bf.to_grid(holed)
    assert np.isnan(grid.loc["2025-06-01 13:00", "price"])
    assert grid["price"].isna().sum() == 1

def test_table_has_the_expected_clues():
    prices, load, weather, points = make_inputs()
    table = bf.build_features(prices, load, weather, points)
    assert table.index.name == "delivery_time"
    assert table.columns[0] == bf.TARGET

    # past prices come from the same hour on earlier days
    assert table.loc["2025-06-10 17:00", "price_lag_24"] == table.loc["2025-06-09 17:00", bf.TARGET]
    assert table.loc["2025-06-10 17:00", "price_lag_168"] == table.loc["2025-06-03 17:00", bf.TARGET]

    # calendar
    assert table.loc["2025-06-10 17:00", "hour"] == 17
    assert table.loc["2025-06-10 17:00", "day_of_week"] == 1                    # a Tuesday
    assert (table.loc["2025-04-21", "is_holiday"] == 1).all()                   # Easter Monday
    assert (table.loc["2025-04-22", "is_holiday"] == 0).all()

    # the wind index averages the wind points, the sun index the sun points
    row = table.loc["2025-06-10 17:00"]
    assert np.isclose(row["wind_index"], (row["sea_ijmuiden_wind_speed_100m"] + row["groningen_wind_speed_100m"]) / 2)
    assert row["sun_index"] == row["de_bilt_shortwave_radiation"]

def test_cutoff_keeps_only_what_had_been_published():
    prices, load, weather, _ = make_inputs()
    cut_prices, cut_load, cut_weather = bf.known_at_cutoff(prices, load, weather, "2025-06-10")
    assert cut_prices.index.max() == pd.Timestamp("2025-06-09 23:00", tz=TZ)    # up to the day before
    assert cut_load.index.max() == pd.Timestamp("2025-06-10 23:00", tz=TZ)      # the delivery day itself
    assert cut_weather.index.max() == pd.Timestamp("2025-06-11 11:00", tz=TZ)   # issued by 11:00 on 9 June

def test_removing_the_future_changes_nothing():
    prices, load, weather, points = make_inputs()
    days = pd.to_datetime(["2025-04-15", SPRING, "2025-03-31", "2025-06-10", AUTUMN, "2025-10-27", "2025-11-20"])
    assert bf.leaking_days(bf.build_features, prices, load, weather, points, days) == []


def test_a_planted_leak_is_caught():
    def leaky(prices, load, weather, points):
        table = bf.build_features(prices, load, weather, points)
        table["price_12h_ago"] = table[bf.TARGET].shift(12)   # for the afternoon, this is the same day's price
        return table

    prices, load, weather, points = make_inputs()
    days = pd.to_datetime(["2025-04-15", "2025-06-10"])
    assert bf.leaking_days(leaky, prices, load, weather, points, days) == list(days)

def test_too_good_flags_a_copy_of_the_price():
    prices, load, weather, points = make_inputs()
    table = bf.build_features(prices, load, weather, points)
    assert bf.too_good(table) == []
    table["sneaky"] = table[bf.TARGET] * 2 + 1
    assert bf.too_good(table) == ["sneaky"]

def test_only_complete_days_are_kept():
    prices, load, weather, points = make_inputs()
    table = bf.build_features(prices, load, weather, points)
    table.loc["2025-06-10 17:00", "load_fc_nl"] = np.nan      # one blank value in the middle
    kept, holes = bf.complete_days(table)

    assert kept.notna().all().all()
    assert (kept.groupby(kept.index.normalize()).size() == 24).all()
    assert kept.index.min() == pd.Timestamp("2025-03-08 00:00")   # the first week has no seven-day lag
    assert holes == [pd.Timestamp("2025-06-10").date()]           # the whole day goes, and is reported