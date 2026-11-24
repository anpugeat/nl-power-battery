"""replay history one day at a time, and score two reference forecasts.

Run with:  python -m src.backtest.replay
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

TARGET = "price"                              # the column being forecast
TEST_START = "2025-01-01"                     # first delivery day that is forecast and scored
LEAR_WINDOW = 364                             # the most days of history the textbook method may use
LEAR_EXTRAS = ("load_fc_nl", "wind_index")    # its two extra inputs, besides past prices
SAVED = "data/forecasts"                      # slow forecasts are kept here as they are made
RANK_HOURS = 4                                # hours counted at each end of the day for the rank hit rate

def known_before(table, day):
    """What a forecaster may see for one delivery day: the past, and that day's clues with its prices hidden."""
    day = pd.Timestamp(day)
    known = table[table.index < day + pd.Timedelta(days=1)].copy()
    known.loc[known.index >= day, TARGET] = np.nan
    return known


def days_from(table, start):
    """The delivery days in the table on or after a date, in order."""
    days = table.index.normalize().unique().sort_values()
    return days[days >= pd.Timestamp(start)]


def replay(table, days, forecast_day, every=0):
    """Forecast each day in turn, handing the forecaster only what was known the morning before."""
    pieces = []
    for count, day in enumerate(pd.DatetimeIndex(days), start=1):
        known = known_before(table, day)
        rows = known.index[known.index >= day]
        assert len(rows) == 24, f"{day.date()} is not a complete day in the table"
        values = np.asarray(forecast_day(known, day), dtype=float).reshape(-1)
        assert len(values) == 24 and np.isfinite(values).all(), f"{day.date()}: the forecast is not 24 numbers"
        pieces.append(pd.Series(values, index=rows))
        if every and count % every == 0:
            print(f"  {count} of {len(days)} days")
    return pd.concat(pieces).rename_axis("delivery_time")

def naive_day(known, day):
    """Same as last week: every hour gets the price of the same hour seven days earlier."""
    return known.loc[known.index >= day, "price_lag_168"].to_numpy()

def lear_table(known, day, extras=LEAR_EXTRAS):
    """The table the textbook method expects: every hour present, its column names, the day's prices blank."""
    day = pd.Timestamp(day)
    hours = pd.date_range(known.index.min(), day + pd.Timedelta(hours=23), freq="h")
    table = known[[TARGET, *extras]].reindex(hours)
    table.columns = ["Price"] + [f"Exogenous {number}" for number in range(1, len(extras) + 1)]
    for _ in range(4):                                     # a missing day borrows the same hours of the week before
        table = table.fillna(table.shift(168))
    table = table.fillna(table.shift(-168))                # or of the week after, when there is no week before
    table.loc[table.index >= day, "Price"] = np.nan        # borrowing must not fill in the day being forecast
    assert table.loc[table.index < day].notna().all().all(), f"{day.date()}: a gap in the history is too long to fill"
    return table

def make_lear():
    """The textbook method from the epftoolbox library, with its last step repaired for current NumPy."""
    from epftoolbox.models import LEAR                     # imported here because it takes several seconds

    class Lear(LEAR):
        def predict(self, X):
            X[:, :-7] = self.scalerX.transform(X[:, :-7])
            scaled = np.array([self.models[hour].predict(X)[0] for hour in range(24)])
            return self.scalerY.inverse_transform(scaled.reshape(1, -1))

    return Lear(calibration_window=LEAR_WINDOW)


def lear_day(known, day, model=None):
    """The textbook method's forecast for one day, fitted afresh on the days before it."""
    model = model or make_lear()
    table = lear_table(known, day)
    forecast = model.recalibrate_and_forecast_next_day(df=table, next_day_date=pd.Timestamp(day),
                                                       calibration_window=LEAR_WINDOW)
    return np.asarray(forecast, dtype=float).reshape(-1)

def kept_on_disk(forecast_day, path):
    """Wrap a slow forecaster so each day's answer is saved as it is made, and never worked out twice."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = pd.Series(dtype=float)
    if path.exists():
        on_file = pd.read_csv(path, index_col="delivery_time", parse_dates=True)["forecast"]
        saved = on_file[~on_file.index.duplicated(keep="last")]
        saved = saved[saved.groupby(saved.index.normalize()).transform("size") == 24]
        if len(saved) < len(on_file):                      # a day cut short by a crash is thrown away
            saved.to_frame().to_csv(path)

    def forecast(known, day):
        rows = pd.date_range(day, periods=24, freq="h", name="delivery_time")
        if rows[0] in saved.index:
            return saved.reindex(rows).to_numpy()
        values = np.asarray(forecast_day(known, day), dtype=float).reshape(-1)
        pd.DataFrame({"forecast": values}, index=rows).to_csv(path, mode="a", header=not path.exists())
        return values

    return forecast

def mae(actual, forecast):
    """Average size of the error, in euros per MWh."""
    return float((actual - forecast).abs().mean())


def rank_hit_rate(actual, forecast, k=RANK_HOURS):
    """Share of each day's k cheapest and k dearest hours that the forecast also picks."""
    hits = []
    for _, real in actual.groupby(actual.index.normalize()):
        guess = forecast.loc[real.index]
        cheap = len(set(real.nsmallest(k).index) & set(guess.nsmallest(k).index))
        dear = len(set(real.nlargest(k).index) & set(guess.nlargest(k).index))
        hits.append((cheap + dear) / (2 * k))
    return float(np.mean(hits))


def score_table(forecasts, baseline="naive"):
    """One row per forecast: average error, error relative to the baseline, and rank hit rate."""
    actual = forecasts[TARGET]
    rows = {}
    for name in forecasts.columns.drop(TARGET):
        rows[name] = {"mae": mae(actual, forecasts[name]),
                      "rmae": mae(actual, forecasts[name]) / mae(actual, forecasts[baseline]),
                      "rank_hit_rate": rank_hit_rate(actual, forecasts[name])}
    return pd.DataFrame(rows).T

def read_features(engine):
    """Read the table saved by Step 4, with delivery_time as its index."""
    table = pd.read_sql_table("features", engine, index_col="delivery_time")
    table.index = pd.to_datetime(table.index)
    return table.sort_index()


def main():
    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])
    features = read_features(engine)
    needed = [TARGET, "price_lag_168", *LEAR_EXTRAS]
    assert set(needed) <= set(features.columns), f"features has no column called: {sorted(set(needed) - set(features.columns))}"
    assert features[needed].notna().all().all(), "blank values in features"
    assert (features.groupby(features.index.normalize()).size() == 24).all(), "a day in features does not have 24 rows"

    days = days_from(features, TEST_START)
    assert len(days) > 0, f"features has no days on or after {TEST_START}"
    first = features.index.min().normalize()
    history = (days[0] - first).days
    inputs = 96 + 7 + 72 * len(LEAR_EXTRAS)                # the numbers the textbook method weighs for each hour
    assert history >= inputs + 9, f"the textbook method needs {inputs + 9} days before {TEST_START}, features has {history}"

    print(f"note: replaying {len(days)} days, from {days[0].date()} to {days[-1].date()}")
    if history < LEAR_WINDOW:
        full = (first + pd.Timedelta(days=LEAR_WINDOW)).date()
        print(f"note: the textbook method starts with {history} days of history and has its full {LEAR_WINDOW} from {full}")
    holes = pd.date_range(first, days[-1], freq="D").difference(features.index.normalize())
    if len(holes):
        print(f"note: {len(holes)} days are missing from features. They are not forecast, and the textbook "
              f"method borrows the week before in their place: {', '.join(str(hole.date()) for hole in holes)}")

    forecasts = pd.DataFrame({TARGET: features.loc[features.index >= days[0], TARGET]})
    forecasts["naive"] = replay(features, days, naive_day)
    saved = f"{SAVED}/lear_{LEAR_WINDOW}_{'_'.join(LEAR_EXTRAS)}.csv"
    print(f"textbook method: a few seconds a day the first time, then reused from {saved}")
    forecasts["lear"] = replay(features, days, kept_on_disk(lear_day, saved), every=25)

    assert forecasts.notna().all().all(), "a forecast is blank"
    assert (forecasts.groupby(forecasts.index.normalize()).size() == 24).all(), "a forecast day does not have 24 rows"
    scores = score_table(forecasts)
    assert scores.loc["naive", "rmae"] == 1, "the baseline does not score exactly 1 against itself"

    print(scores.round(3).to_string())
    if scores.loc["lear", "rmae"] >= 1:
        print("note: the textbook method did not beat same-as-last-week. Look into this before Step 6")
    forecasts.to_sql("baseline_forecasts", engine, if_exists="replace", index_label="delivery_time")
    Path("reports").mkdir(exist_ok=True)
    scores.to_csv("reports/baseline_scores.csv", index_label="forecast")
    print(f"saved {len(forecasts)} rows for {len(days)} days")


if __name__ == "__main__":
    main()