"""Download the weather and load inputs and save them to the database.
Run from the project folder:  python -m src.ingest.weather_and_load"""
import os

import pandas as pd
import requests

from src.ingest.entsoe_inputs import hourly_utc, pull_load, read_load
from src.ingest.openmeteo import check_hourly, combine, pull_all

FIRST_DAY = "2024-01-01"

AVAILABILITY = [
    {"input": "prices", "source": "ENTSO-E", "use_as_feature": True,
     "known_when": "Prices for a day are published the day before it, after the auction"},
    {"input": "load_fc_nl, load_fc_de", "source": "ENTSO-E", "use_as_feature": True,
     "known_when": "By 10:00 on the day before delivery"},
    {"input": "weather (_previous_day2)", "source": "Open-Meteo", "use_as_feature": True,
     "known_when": "48 hours before each hour"},
    {"input": "tso_wind_solar_fc", "source": "ENTSO-E", "use_as_feature": False,
     "known_when": "Only guaranteed by late afternoon on the day before delivery"},
]


def main():
    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient
    from sqlalchemy import create_engine

    load_dotenv()
    today = pd.Timestamp.now(tz="Europe/Amsterdam").normalize()
    last_day = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yesterday

    # weather
    points = pd.read_csv("config/weather_points.csv")
    pull_all(points, FIRST_DAY, last_day)
    weather = combine(points)
    first_full = max(weather[c].first_valid_index() for c in weather.columns)  # the last column to start
    weather_start = first_full.ceil("D").strftime("%Y-%m-%d")                   # the next full day
    weather = weather.loc[weather_start:]
    missing_hours, blanks = check_hourly(weather, weather_start, last_day)
    assert len(missing_hours) == 0, f"{len(missing_hours)} weather hours are missing"
    print(f"note: weather history starts on {weather_start}")
    if len(blanks) > 0:
        print("note: blank weather values after that date:", {k: int(v) for k, v in blanks.items()})

    # load forecasts
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"], retry_count=1)
    for zone in ("NL", "DE_LU"):
        pull_load(client, zone, FIRST_DAY, last_day)
    loads = pd.concat([read_load("NL").rename("load_fc_nl"),
                       read_load("DE_LU").rename("load_fc_de")], axis=1)
    if loads.isna().sum().sum() > 0:
        print("note: blank load hours:", dict(loads.isna().sum()), "Run again to fill failed months.")

    engine = create_engine(os.environ["DATABASE_URL"])
    weather.to_sql("weather_fc", engine, if_exists="replace", index_label="time_utc")
    loads.to_sql("load_fc", engine, if_exists="replace", index_label="time_utc")
    pd.DataFrame(AVAILABILITY).to_sql("availability", engine, if_exists="replace", index=False)
    print(f"saved {len(weather)} weather hours and {len(loads)} load hours")

    # official wind and solar forecasts: stored, not used as features
    try:
        start = pd.Timestamp(FIRST_DAY, tz="Europe/Amsterdam")
        tso_nl = hourly_utc(client.query_wind_and_solar_forecast("NL", start=start, end=today))
        tso_nl.columns = ["tso_" + c.lower().replace(" ", "_") + "_nl" for c in tso_nl.columns]
        tso_nl.to_sql("tso_wind_solar_fc", engine, if_exists="replace", index_label="time_utc")
        print(f"saved {len(tso_nl)} wind and solar hours")
    except requests.RequestException as error:
        print("note: official wind and solar forecasts not updated:", type(error).__name__)


if __name__ == "__main__":
    main()