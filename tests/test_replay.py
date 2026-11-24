import numpy as np
import pandas as pd
import pytest

import src.backtest.replay as rp

DAY = pd.Timestamp("2025-04-10")   # the delivery day most tests forecast


def make_features(start="2025-03-01", end="2025-05-31", seed=0, missing=()):
    """A stand-in copy of the features table: 24 rows a day, the price and the clues this step reads."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", name="delivery_time")
    n = len(hours)
    table = pd.DataFrame(index=hours)
    table["load_fc_nl"] = 12000 + 2500 * np.sin((hours.hour - 8) / 24 * 2 * np.pi) + rng.normal(0, 300, n)
    table["wind_index"] = rng.gamma(2.0, 4.0, n)
    table.insert(0, "price", 60 + 0.01 * (table["load_fc_nl"] - 12000) - 2 * table["wind_index"] + rng.normal(0, 8, n))
    table["price_lag_168"] = table["price"].shift(168)
    table = table.dropna()                                    # as in Step 4: the first week has no seven-day lag
    return table[~table.index.normalize().isin(pd.to_datetime(list(missing)))]

class FakeLear:
    """Stands in for the library's model: remembers what it was handed and answers with 24 numbers."""

    def recalibrate_and_forecast_next_day(self, df, next_day_date, calibration_window):
        self.df, self.day, self.window = df, next_day_date, calibration_window
        return np.arange(24.0).reshape(1, 24)                 # the library answers in this shape

def test_the_forecaster_is_shown_the_past_and_not_the_answer():
    table = make_features()
    known = rp.known_before(table, DAY)
    assert known.index.max() == pd.Timestamp("2025-04-10 23:00")        # nothing after the delivery day
    assert known.loc["2025-04-10", "price"].isna().all()                # that day's prices are hidden
    assert known.loc["2025-04-10", "load_fc_nl"].notna().all()          # its clues are not
    assert known.loc[:"2025-04-09"].equals(table.loc[:"2025-04-09"])    # the past is untouched
    assert table.loc["2025-04-10", "price"].notna().all()               # and so is the table it came from


def test_replay_gives_24_forecasts_for_each_day_in_order():
    table = make_features(missing=["2025-04-12"])
    days = rp.days_from(table, "2025-04-10")[:4]
    assert list(days) == list(pd.to_datetime(["2025-04-10", "2025-04-11", "2025-04-13", "2025-04-14"]))

    asked = []

    def hour_numbers(known, day):
        asked.append(day)
        return np.arange(24)

    forecast = rp.replay(table, days, hour_numbers)
    assert asked == list(days)                                          # one call per day, in order
    assert forecast.index.name == "delivery_time"
    assert forecast.index.equals(table.index[table.index.normalize().isin(days)])
    assert (forecast.to_numpy() == np.tile(np.arange(24), 4)).all()


def test_a_forecaster_that_tries_to_peek_is_stopped():
    def peek(known, day):
        return known.loc[known.index >= day, "price"]                   # the answers, if they were visible

    with pytest.raises(AssertionError, match="not 24 numbers"):
        rp.replay(make_features(), [DAY], peek)

def test_same_as_last_week_is_the_price_seven_days_earlier():
    table = make_features()
    forecast = rp.replay(table, [DAY], rp.naive_day)
    assert (forecast.to_numpy() == table.loc["2025-04-03", "price"].to_numpy()).all()
    
def test_the_textbook_table_is_unbroken_and_hides_the_day():
    table = make_features()
    lear = rp.lear_table(rp.known_before(table, DAY), DAY)
    assert list(lear.columns) == ["Price", "Exogenous 1", "Exogenous 2"]
    assert lear.index.equals(pd.date_range("2025-03-08 00:00", "2025-04-10 23:00", freq="h"))
    assert lear.loc["2025-04-10", "Price"].isna().all()                 # the day's prices are blank
    assert lear.loc[:"2025-04-09", "Price"].notna().all()               # every earlier price is there
    assert lear.drop(columns="Price").notna().all().all()               # and every extra input, the day included
    assert (lear.loc["2025-04-10", "Exogenous 1"].to_numpy() == table.loc["2025-04-10", "load_fc_nl"].to_numpy()).all()


def test_a_missing_day_is_borrowed_from_the_week_before():
    table = make_features(missing=["2025-03-10", "2025-04-03", "2025-04-08"])
    lear = rp.lear_table(rp.known_before(table, DAY), DAY)
    assert lear.index.equals(pd.date_range("2025-03-08 00:00", "2025-04-10 23:00", freq="h"))
    assert (lear.loc["2025-04-08"].to_numpy() == lear.loc["2025-04-01"].to_numpy()).all()
    assert (lear.loc["2025-04-03"].to_numpy() == lear.loc["2025-03-27"].to_numpy()).all()
    assert (lear.loc["2025-03-10"].to_numpy() == lear.loc["2025-03-17"].to_numpy()).all()   # no week before: the week after
    assert lear.loc["2025-04-10", "Price"].isna().all()                 # borrowing never fills in the answer

def test_the_textbook_method_is_handed_the_right_things():
    table = make_features()
    fake = FakeLear()
    forecast = rp.lear_day(rp.known_before(table, DAY), DAY, model=fake)
    assert forecast.shape == (24,)                                      # flattened to one number per hour
    assert fake.day == DAY and fake.window == rp.LEAR_WINDOW
    assert fake.df.index.max() == pd.Timestamp("2025-04-10 23:00")
    assert fake.df.loc["2025-04-10", "Price"].isna().all()

def test_saved_days_are_reused_and_a_half_saved_day_is_redone(tmp_path):
    table = make_features()
    days = rp.days_from(table, "2025-04-10")
    path = tmp_path / "forecasts" / "slow.csv"
    asked = []

    def slow(known, day):
        asked.append(day)
        return np.arange(24) + day.day

    first = rp.replay(table, days[:3], rp.kept_on_disk(slow, path))
    again = rp.replay(table, days[:4], rp.kept_on_disk(slow, path))
    assert asked == list(days[:3]) + [days[3]]                          # the second run only works out the new day
    assert again.iloc[:72].equals(first)

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-10]) + "\n")                      # as if the run died while saving the last day
    rp.replay(table, days[:4], rp.kept_on_disk(slow, path))
    assert asked[-1] == days[3] and len(asked) == 5                     # only that day is worked out again
    assert len(pd.read_csv(path)) == 96                                 # and the file holds four whole days

def test_scores_reward_a_good_forecast():
    table = make_features()
    days = rp.days_from(table, "2025-04-01")
    actual = table.loc[table.index >= days[0], "price"]
    forecasts = pd.DataFrame({"price": actual, "naive": rp.replay(table, days, rp.naive_day),
                              "perfect": actual, "upside_down": -actual})
    scores = rp.score_table(forecasts)
    assert list(scores.columns) == ["mae", "rmae", "rank_hit_rate"]
    assert scores.loc["naive", "rmae"] == 1                             # the baseline measured against itself
    assert scores.loc["perfect", "mae"] == 0 and scores.loc["perfect", "rank_hit_rate"] == 1
    assert scores.loc["upside_down", "rank_hit_rate"] == 0              # it picks the dearest hours as the cheapest
    assert 0 < scores.loc["naive", "rank_hit_rate"] < 1