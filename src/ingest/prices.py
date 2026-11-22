"""Download Dutch day-ahead prices and build the two price tables."""
import os
from pathlib import Path

import pandas as pd


def download(start="2024-01-01"):
    """Fetch the raw prices from ENTSO-E (EUR per MWh, Amsterdam time)."""
    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient

    load_dotenv()
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    begin = pd.Timestamp(start, tz="Europe/Amsterdam")
    end = pd.Timestamp.now(tz="Europe/Amsterdam").normalize() - pd.Timedelta(minutes=15)
    return client.query_day_ahead_prices("NL", start=begin, end=end)


def to_hourly(raw):
    """One price per hour, on the UTC clock."""
    hourly = raw.tz_convert("UTC").resample("h").mean()
    hourly.name = "price"
    return hourly


def trim_to_complete_days(hourly):
    """Keep the unbroken run of prices from the start, ending on a full Amsterdam day."""
    gaps = hourly[hourly.isna()]
    if len(gaps) > 0:
        hourly = hourly[hourly.index < gaps.index[0]]
    last = hourly.index[-1].tz_convert("Europe/Amsterdam")
    if last.hour != 23:  # the final day is cut short
        hourly = hourly[hourly.index < last.normalize()]
    return hourly


def to_daily(hourly):
    """One row per Amsterdam day, one column per hour of that day."""
    local = hourly.tz_convert("Europe/Amsterdam")
    table = (local.groupby([local.index.date, local.index.hour]).mean()  # averages the repeated autumn hour
                  .unstack()                                             # rows: days, columns: hours 0 to 23
                  .interpolate(axis=1))                                  # fills the skipped spring hour
    table.index = pd.to_datetime(table.index)
    table.index.name = "delivery_date"
    table.columns = [f"h{h:02d}" for h in table.columns]
    return table


def main():
    from sqlalchemy import create_engine

    raw = download()
    Path("data").mkdir(exist_ok=True)
    raw.tz_convert("UTC").to_csv("data/raw_prices_nl.csv", header=["price"])

    hourly = trim_to_complete_days(to_hourly(raw))
    table = to_daily(hourly)
    assert table.shape[1] == 24 and table.isna().sum().sum() == 0, "a day is incomplete"

    yesterday = pd.Timestamp.now(tz="Europe/Amsterdam").normalize().tz_localize(None) - pd.Timedelta(days=1)
    days_short = (yesterday - table.index[-1]).days
    assert days_short <= 7, f"usable prices stop at {table.index[-1].date()}, {days_short} days short: see step 2.5"
    if days_short > 0:
        print(f"note: no unbroken prices after {table.index[-1].date()} ({days_short} days short). Rerun later.")

    engine = create_engine(os.environ["DATABASE_URL"])
    hourly.to_frame().to_sql("prices_utc", engine, if_exists="replace", index_label="time_utc")
    table.to_sql("prices_daily", engine, if_exists="replace")
    print(f"saved {len(hourly)} hours and {len(table)} days")


if __name__ == "__main__":
    main()