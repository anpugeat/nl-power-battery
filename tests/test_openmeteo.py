import pandas as pd
import requests

from src.ingest.entsoe_inputs import pull_load, read_load
from src.ingest.openmeteo import LEAD, check_hourly, chunks, combine, to_frame


def make_payload(start, end):
    """Stand-in for one Open-Meteo answer."""
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    hourly = {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours]}
    for name in ("wind_speed_100m", "shortwave_radiation", "temperature_2m"):
        hourly[name + LEAD] = [1.0] * len(hours)
    return {"hourly": hourly}


def test_weather_answer_becomes_an_hourly_utc_table():
    frame = to_frame(make_payload("2024-06-01", "2024-06-07"))
    assert len(frame) == 7 * 24
    assert str(frame.index.tz) == "UTC"
    assert list(frame.columns) == ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]


def test_chunks_cover_every_day_once():
    days = pd.DatetimeIndex([])
    for start, end in chunks("2024-01-01", "2026-11-21"):
        days = days.append(pd.date_range(start, end, freq="D"))
    assert days.equals(pd.date_range("2024-01-01", "2026-11-21", freq="D"))


def test_combine_joins_the_files_and_the_check_finds_gaps(tmp_path):
    points = pd.DataFrame({"name": ["a", "b"], "lat": [52.0, 53.0], "lon": [4.0, 5.0]})
    for name in points["name"]:
        for start, end in chunks("2024-01-01", "2024-06-30"):
            to_frame(make_payload(start, end)).to_csv(tmp_path / f"{name}_{start}_{end}.csv")
    weather = combine(points, folder=tmp_path)
    assert weather.shape[1] == 6
    missing, blanks = check_hourly(weather, "2024-01-01", "2024-06-30")
    assert len(missing) == 0 and len(blanks) == 0
    missing, _ = check_hourly(weather.drop(weather.index[100:103]), "2024-01-01", "2024-06-30")
    assert len(missing) == 3


class FlakyClient:
    """Stand-in for ENTSO-E that fails once for March, as the real service sometimes does."""

    def __init__(self):
        self.failed = False

    def query_load_forecast(self, zone, start, end):
        if start.month == 3 and not self.failed:
            self.failed = True
            raise requests.ConnectionError("dropped")
        index = pd.date_range(start, end, freq="15min")
        return pd.DataFrame({"Forecasted Load": 10000.0}, index=index)


def test_load_pull_resumes_after_a_failure(tmp_path):
    client = FlakyClient()
    pull_load(client, "NL", "2024-01-01", "2024-06-01", folder=tmp_path, pause=0)
    assert len(list(tmp_path.glob("NL_*.csv"))) == 5  # March failed
    pull_load(client, "NL", "2024-01-01", "2024-06-01", folder=tmp_path, pause=0)
    assert len(list(tmp_path.glob("NL_*.csv"))) == 6  # the second run filled it
    load = read_load("NL", folder=tmp_path)
    assert load.isna().sum() == 0
    assert str(load.index.tz) == "UTC"