import pandas as pd

from src.ingest.prices import to_daily, to_hourly, trim_to_complete_days


def make_raw():
    """Stand-in for the download: hourly, then quarter-hourly, across both clock changes."""
    a = pd.date_range("2025-01-01", "2025-09-30 23:00", freq="h", tz="Europe/Amsterdam")
    b = pd.date_range("2025-10-01", "2025-12-31 23:45", freq="15min", tz="Europe/Amsterdam")
    index = a.append(b)
    return pd.Series(range(len(index)), index=index, dtype=float)


def test_hourly_has_no_gaps():
    hourly = to_hourly(make_raw())
    assert hourly.isna().sum() == 0
    assert len(hourly) == 365 * 24


def test_every_day_has_24_prices():
    table = to_daily(to_hourly(make_raw()))
    assert table.shape == (365, 24)
    assert table.isna().sum().sum() == 0


def test_complete_data_is_left_alone():
    hourly = to_hourly(make_raw())
    assert trim_to_complete_days(hourly).equals(hourly)


def test_hole_near_the_end_is_trimmed_to_a_full_day():
    raw = make_raw()
    holed = raw.drop(raw["2025-12-29":"2025-12-30"].index)  # two days missing, 31 December present
    table = to_daily(trim_to_complete_days(to_hourly(holed)))
    assert table.index[-1] == pd.Timestamp("2025-12-28")
    assert table.isna().sum().sum() == 0