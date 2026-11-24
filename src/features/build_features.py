"""One table holding the price and every clue, 24 rows for every Amsterdam day.

Run with:  python -m src.features.build_features
"""
import os

import holidays
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

TZ = "Europe/Amsterdam"
CUTOFF_HOUR = 11            # the forecast is made at 11:00 on the day before delivery
TARGET = "price"            # the column the models will predict
LAGS = (24, 48, 72, 168)    # hours back: one, two, three and seven days
CHECK_DAYS = 20             # delivery days used for the leak check on the real data

def to_grid(table):
    """Put an hourly UTC table on the auction's grid: 24 rows for every Amsterdam day."""
    local = table.tz_convert(TZ)
    day = local.index.normalize().tz_localize(None)
    grid = local.groupby([day, local.index.hour]).mean()      # averages the repeated autumn hour
    grid.index = grid.index.get_level_values(0) + pd.to_timedelta(grid.index.get_level_values(1), unit="h")

    hours = pd.date_range(grid.index.min().normalize(),
                          grid.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    grid = grid.reindex(hours)                                 # every day now has 24 rows

    real = hours.tz_localize(TZ, nonexistent="NaT", ambiguous=np.ones(len(hours), dtype=bool))
    skipped = real.isna()                                      # the spring hour that never happens
    neighbours = (grid.shift(1) + grid.shift(-1)) / 2
    grid.loc[skipped] = neighbours.loc[skipped]                # fill it from the hours either side
    grid.index.name = "delivery_time"
    return grid

def build_features(prices, load, weather, points):
    """One row per delivery hour: the price, and every clue known at 11:00 the day before."""
    grids = [to_grid(prices), to_grid(load), to_grid(weather)]
    hours = pd.date_range(min(g.index.min() for g in grids), max(g.index.max() for g in grids),
                          freq="h", name="delivery_time")
    table = pd.concat([g.reindex(hours) for g in grids], axis=1)

    for lag in LAGS:                                           # the same hour on earlier days
        table[f"price_lag_{lag}"] = table[TARGET].shift(lag)

    table["hour"] = hours.hour
    table["day_of_week"] = hours.day_of_week
    table["month"] = hours.month
    dutch_holidays = holidays.country_holidays("NL")
    table["is_holiday"] = [int(day in dutch_holidays) for day in hours.date]

    wind = [f"{name}_wind_speed_100m" for name in points.loc[points["role"] == "wind", "name"]]
    sun = [f"{name}_shortwave_radiation" for name in points.loc[points["role"] == "sun", "name"]]
    table["wind_index"] = table[wind].mean(axis=1, skipna=False)
    table["sun_index"] = table[sun].mean(axis=1, skipna=False)
    return table

def known_at_cutoff(prices, load, weather, delivery_day):
    """Cut each input down to what the availability table says was known at the cut-off."""
    day = pd.Timestamp(delivery_day)
    day_start = day.tz_localize(TZ)
    next_day_start = (day + pd.Timedelta(days=1)).tz_localize(TZ)
    cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_HOUR)).tz_localize(TZ)
    return (prices[prices.index < day_start],                           # prices up to the day before
            load[load.index < next_day_start],                          # load forecasts up to the delivery day
            weather[weather.index <= cutoff + pd.Timedelta(hours=48)])  # weather issued by the cut-off

def leaking_days(build, prices, load, weather, points, days):
    """Delivery days whose clues change when everything after the cut-off is taken away."""
    full = build(prices, load, weather, points).drop(columns=TARGET)
    leaks = []
    for day in days:
        cut = build(*known_at_cutoff(prices, load, weather, day), points).drop(columns=TARGET)
        rows = pd.date_range(day, periods=24, freq="h")
        if not cut.reindex(rows).equals(full.reindex(rows)):
            leaks.append(day)
    return leaks

def too_good(table, limit=0.99):
    """Clues that follow the price almost perfectly, which usually means the answer slipped in."""
    match = table.drop(columns=TARGET).corrwith(table[TARGET]).abs()
    return sorted(match[match > limit].index)

def complete_days(table):
    """Keep the days on which all 24 rows are complete, and list the days dropped in between."""
    day = table.index.normalize()
    complete = table.notna().all(axis=1).groupby(day).transform("all")
    kept = table[complete.to_numpy()]
    span = pd.date_range(kept.index.min().normalize(), kept.index.max().normalize(), freq="D")
    holes = span.difference(kept.index.normalize())
    return kept, [hole.date() for hole in holes]

def read_table(engine, name):
    """Read one of the hourly tables saved by Steps 2 and 3, with a UTC time index."""
    table = pd.read_sql_table(name, engine, index_col="time_utc")
    table.index = pd.to_datetime(table.index, utc=True)
    return table.sort_index()


def main():
    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])
    prices = read_table(engine, "prices_utc")
    load = read_table(engine, "load_fc")
    weather = read_table(engine, "weather_fc")
    points = pd.read_csv("config/weather_points.csv")
    assert list(prices.columns) == [TARGET], f"unexpected price columns: {list(prices.columns)}"
    assert list(load.columns) == ["load_fc_nl", "load_fc_de"], f"unexpected load columns: {list(load.columns)}"
    assert len(weather.columns) == 3 * len(points), f"expected 3 weather columns per point, found {len(weather.columns)}"

    table = build_features(prices, load, weather, points)
    features, holes = complete_days(table)

    assert features.notna().all().all(), "blank values are left in the table"
    assert (features.groupby(features.index.normalize()).size() == 24).all(), "a day does not have 24 rows"

    days = pd.Series(features.index.normalize().unique()).sample(CHECK_DAYS, random_state=0)
    leaks = leaking_days(build_features, prices, load, weather, points, sorted(days))
    assert not leaks, f"clues change when the future is removed, on: {[day.date() for day in leaks]}"

    suspects = too_good(features)
    assert not suspects, f"clues that follow the price too closely: {suspects}"

    print(f"note: features run from {features.index.min().date()} to {features.index.max().date()}")
    if holes:
        print(f"note: {len(holes)} days inside that range were dropped for blanks: {', '.join(map(str, holes))}")
    features.to_sql("features", engine, if_exists="replace", index_label="delivery_time")
    print(f"saved {len(features)} rows and {features.shape[1]} columns for {len(features) // 24} days")


if __name__ == "__main__":
    main()