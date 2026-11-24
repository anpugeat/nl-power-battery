"""Functions for downloading the grid operators' load forecasts from ENTSO-E."""
import time
from pathlib import Path

import pandas as pd
import requests


def hourly_utc(table):
    """Average to one value per hour, on the UTC clock."""
    return table.tz_convert("UTC").resample("h").mean()


def pull_load(client, zone, first_month, last_month, folder="data/raw/load", pause=1.0):
    """Fetch the load forecast one month at a time, saving each month and skipping finished ones."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="Europe/Amsterdam")
    for month in pd.date_range(first_month, last_month, freq="MS", tz="Europe/Amsterdam"):
        file = folder / f"{zone}_{month:%Y-%m}.csv"
        month_end = month + pd.DateOffset(months=1)
        if file.exists() and month_end <= now:  # a finished month is never fetched twice
            continue
        try:
            client.query_load_forecast(zone, start=month, end=month_end).to_csv(file)
        except requests.RequestException as error:
            code = getattr(error.response, "status_code", type(error).__name__)
            print(zone, f"{month:%Y-%m}", "failed:", code)
        time.sleep(pause)


def read_load(zone, folder="data/raw/load"):
    """Join the saved months into one hourly series in UTC."""
    files = sorted(Path(folder).glob(f"{zone}_*.csv"))
    table = pd.concat(pd.read_csv(f, index_col=0) for f in files)
    table.index = pd.to_datetime(table.index, utc=True)
    table = table[~table.index.duplicated()].sort_index()
    return table["Forecasted Load"].resample("h").mean()