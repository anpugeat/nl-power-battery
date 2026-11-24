import requests
import pandas as pd
import time
from pathlib import Path

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["wind_speed_100m", "shortwave_radiation", "temperature_2m"]
LEAD = "_previous_day2"

def to_frame(payload):
    """Turn one Open-Meteo answer into a table indexed by hour, in UTC."""
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time")
    frame.columns = [c.replace(LEAD, "") for c in frame.columns]  # drop the suffix from the names
    return frame

def past_forecast(lat, lon, start, end):
    """Ask Open-Meteo what the forecast said 48 hours before each hour."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(v + LEAD for v in VARIABLES),
        "start_date": start, "end_date": end,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    return to_frame(response.json())

def chunks(first_day, last_day, months=3):
    """Split a long period into pieces of a few months each."""
    starts = pd.date_range(first_day, last_day, freq=f"{months}MS")
    for start in starts:
        end = min(start + pd.DateOffset(months=months) - pd.Timedelta(days=1), pd.Timestamp(last_day))
        yield start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def pull_all(points, first_day, last_day, folder="data/raw/weather", pause=1.0):
    """Download every point piece by piece, skipping pieces that are already saved."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for point in points.itertuples():
        for start, end in chunks(first_day, last_day):
            file = folder / f"{point.name}_{start}_{end}.csv"
            if file.exists():
                continue
            past_forecast(point.lat, point.lon, start, end).to_csv(file)
            time.sleep(pause)

def combine(points, folder="data/raw/weather"):
    """One table: a row per hour, a column per point and variable."""
    tables = []
    for point in points.itertuples():
        files = sorted(Path(folder).glob(f"{point.name}_*.csv"))
        one = pd.concat(pd.read_csv(f, index_col="time", parse_dates=True) for f in files)
        one = one[~one.index.duplicated()].sort_index()
        tables.append(one.add_prefix(f"{point.name}_"))
    return pd.concat(tables, axis=1)

def check_hourly(table, first_day, last_day):
    """Report missing hours and blank values."""
    expected = pd.date_range(first_day, pd.Timestamp(last_day) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    missing_hours = expected.difference(table.index)
    blanks = table.reindex(expected).isna().sum()
    return missing_hours, blanks[blanks > 0]

