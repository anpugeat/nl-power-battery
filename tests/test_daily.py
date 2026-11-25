import numpy as np
import pandas as pd
import pytest

import src.backtest.replay as rp
import src.dispatch.battery as bat
import src.features.build_features as bf
import src.ingest.openmeteo as om
import src.live.daily as live
import src.models.gbm as gbm

TZ = "Europe/Amsterdam"
DAY = pd.Timestamp("2026-03-30")   # the delivery day most tests forecast. On the morning before it, the clocks go forward
FAST = {"n_estimators": 30, "num_leaves": 7, "objective": "l2"}   # a small model, so the tests stay quick


def make_world(start="2026-01-01", end="2026-04-05", seed=0):
    """Everything the two sources will ever publish for a stretch of days: prices, load forecasts, weather forecasts, and the points."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ) + pd.Timedelta(hours=23),
                          freq="h").tz_convert("UTC").rename("time_utc")
    n, clock = len(hours), hours.tz_convert(TZ).hour
    points = pd.DataFrame({"name": ["sea_ijmuiden", "groningen", "de_bilt"], "role": ["wind", "wind", "sun"],
                           "lat": [52.6, 53.2, 52.1], "lon": [4.0, 6.6, 5.2]})
    load = pd.DataFrame({"load_fc_nl": 12000 + 2500 * np.sin((clock - 8) / 24 * 2 * np.pi) + rng.normal(0, 300, n)}, index=hours)
    load["load_fc_de"] = 4.6 * load["load_fc_nl"] + rng.normal(0, 900, n)
    weather = pd.DataFrame(index=hours)
    for name in points["name"]:
        weather[f"{name}_wind_speed_100m"] = rng.gamma(2.0, 4.0, n)
        weather[f"{name}_shortwave_radiation"] = np.clip(np.sin((clock - 6) / 12 * np.pi), 0, None) * 600
        weather[f"{name}_temperature_2m"] = rng.normal(11, 6, n)
    prices = pd.DataFrame({"price": 60 + 0.01 * (load["load_fc_nl"] - 12000) - 2 * weather["sea_ijmuiden_wind_speed_100m"]
                           - 0.03 * weather["de_bilt_shortwave_radiation"] + rng.normal(0, 4, n)}, index=hours)
    return prices, load, weather, points


def known_on(world, day):
    """What the sources hold on the morning before a delivery day: prices to the end of that morning's day, forecasts a day further."""
    prices, load, weather, points = world
    day_start = pd.Timestamp(day).tz_localize(TZ)
    day_end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    utc_day_end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)       # Open-Meteo answers in whole UTC days
    return prices[prices.index < day_start], load[load.index < day_end], weather[weather.index < utc_day_end], points


class FakeClient:
    """Stand-in for ENTSO-E: it answers from a world, and remembers what it was asked."""

    def __init__(self, world):
        self.prices, self.load, self.asked = world[0], world[1], []

    def answer(self, table, start, end):
        return table[(table.index >= start) & (table.index <= end)].tz_convert(TZ)

    def query_day_ahead_prices(self, zone, start, end):
        self.asked.append(("prices", zone, start, end))
        return self.answer(self.prices["price"], start, end)

    def query_load_forecast(self, zone, start, end):
        self.asked.append(("load", zone, start, end))
        return self.answer(self.load[[live.ZONES[zone]]].set_axis(["Forecasted Load"], axis=1), start, end)


def make_ask(world):
    """Stand-in for Open-Meteo: a function that answers like om.past_forecast, and the list of what it was asked."""
    weather, points = world[2], world[3]
    asked = []

    def ask(lat, lon, start, end):
        asked.append((lat, lon, start, end))
        name = points.loc[(points["lat"] == lat) & (points["lon"] == lon), "name"].iloc[0]
        return weather.filter(like=name).loc[start:f"{end} 23:00"].set_axis(om.VARIABLES, axis=1).rename_axis("time")

    return ask, asked


def make_page():
    """A day worked out by hand: real prices flat but for a dip and a peak, a gbm forecast that is exactly right, a flat naive one."""
    hours = pd.date_range(DAY, periods=24, freq="h", name="delivery_time")
    actual = pd.Series(50.0, index=hours)
    actual.iloc[[12, 13, 14]], actual.iloc[[18, 19]] = 10.0, 100.0
    page = pd.DataFrame({"gbm": actual, "naive": 50.0})
    for name in ["gbm", "naive"]:
        charge, discharge = bat.plan_battery(page[name].to_numpy())
        page[f"{name}_plan"] = discharge - charge
    return page, actual


def make_scores():
    """Three scored days in round numbers: two made in time, and one made late in between."""
    days = pd.DatetimeIndex(["2026-03-30", "2026-03-31", "2026-04-02"], name="delivery_date")
    return pd.DataFrame({"made_at": ["2026-03-29T08:17:00Z", "2026-03-30T10:30:00Z", "2026-04-01T08:16:00Z"],
                         "on_time": [True, False, True],
                         "mae_gbm": [6.0, 99.0, 18.0], "mae_naive": [20.0, 1.0, 28.0],
                         "hits_gbm": [0.75, 0.0, 0.75], "hits_naive": [0.5, 1.0, 0.5],
                         "profit_perfect": [100.0, 100.0, 300.0], "profit_gbm": [90.0, -50.0, 230.0],
                         "profit_naive": [70.0, 100.0, 230.0]}, index=days)

def test_the_delivery_day_is_tomorrow_on_the_amsterdam_clock():
    assert live.delivery_day(pd.Timestamp("2026-03-29 10:15", tz=TZ)) == DAY
    assert live.delivery_day(pd.Timestamp("2026-03-28 23:30", tz="UTC")) == DAY                          # already the 29th in Amsterdam
    assert live.delivery_day(pd.Timestamp("2026-07-01 21:30", tz="UTC")) == pd.Timestamp("2026-07-02")   # still the 1st there


def test_bidding_ends_at_noon_the_day_before_in_summer_and_in_winter():
    assert live.auction_close(DAY) == pd.Timestamp("2026-03-29 12:00", tz=TZ)
    assert live.auction_close(DAY) == pd.Timestamp("2026-03-29 10:00", tz="UTC")             # summer time began that morning
    assert live.auction_close("2026-01-15") == pd.Timestamp("2026-01-14 11:00", tz="UTC")    # winter time


def test_the_history_starts_on_the_first_of_a_month_and_covers_a_full_window():
    start = live.history_start(DAY)
    assert start == pd.Timestamp("2025-03-01")
    assert (DAY - start).days >= gbm.WINDOW + 8                # the learning days, plus a week for the oldest lag
    assert live.history_start("2026-12-31") == pd.Timestamp("2025-12-01")


def test_only_githubs_machine_writes_to_the_record():
    assert live.record_folder({"GITHUB_ACTIONS": "true"}) == live.RECORD
    assert live.record_folder({}) == live.REHEARSAL
    assert live.record_folder({"GITHUB_ACTIONS": "false"}) == live.REHEARSAL
    assert live.REHEARSAL.startswith("data/")                  # a folder that git ignores

def test_prices_are_asked_up_to_tonight_and_load_up_to_tomorrow_night():
    client = FakeClient(make_world())
    first = pd.Timestamp("2026-02-01")
    prices = live.fetch_prices(client, first, DAY)
    load = live.fetch_load(client, first, DAY)

    asked = {(kind, zone): (start, end) for kind, zone, start, end in client.asked}
    assert asked[("prices", "NL")] == (pd.Timestamp("2026-02-01", tz=TZ), pd.Timestamp("2026-03-29 23:45", tz=TZ))
    assert asked[("load", "NL")] == asked[("load", "DE_LU")] == (pd.Timestamp("2026-02-01", tz=TZ), pd.Timestamp("2026-03-31", tz=TZ))

    assert list(prices.columns) == ["price"] and str(prices.index.tz) == "UTC"
    assert prices.index.max() == pd.Timestamp("2026-03-29 23:00", tz=TZ)          # the last hour of the morning's own day
    assert list(load.columns) == ["load_fc_nl", "load_fc_de"] and str(load.index.tz) == "UTC"
    assert load.index.max() >= pd.Timestamp("2026-03-30 23:00", tz=TZ)            # the whole delivery day is there
    assert prices.notna().all().all() and load.notna().all().all()


def test_weather_is_asked_for_every_point_and_named_as_in_step_3():
    world = make_world()
    ask, asked = make_ask(world)
    weather = live.fetch_weather(world[3], pd.Timestamp("2025-12-01"), DAY, ask=ask, pause=0)

    assert len(asked) == 3 * 2                                                    # three points, two pieces each
    assert {(start, end) for _, _, start, end in asked} == {("2025-12-01", "2026-02-28"), ("2026-03-01", "2026-03-30")}
    assert list(weather.columns) == list(world[2].columns)                        # point by point, variable by variable
    assert weather.index.is_unique and weather.index.is_monotonic_increasing
    assert weather.index.max() == pd.Timestamp("2026-03-30 23:00", tz="UTC")      # past the end of the Amsterdam day
    pd.testing.assert_frame_equal(weather, world[2].loc[:"2026-03-30 23:00"], check_names=False, check_freq=False)

def test_the_table_ends_with_the_day_and_only_its_price_is_blank():
    table = live.live_table(*known_on(make_world(), DAY), DAY)
    ahead, past = table[table.index >= DAY], table[table.index < DAY]
    assert list(ahead.index) == list(pd.date_range(DAY, periods=24, freq="h"))
    assert ahead["price"].isna().all() and ahead.drop(columns="price").notna().all().all()
    assert past.notna().all().all() and (past.groupby(past.index.normalize()).size() == 24).all()
    assert past.index.max() == DAY - pd.Timedelta(hours=1)     # the morning's own day is in, clock change and all
    assert ahead["price_lag_24"].iloc[13] == past.loc[DAY - pd.Timedelta(hours=11), "price"]   # 13:00 tomorrow sees 13:00 today
    assert live.not_ready(table, DAY) == []


def test_a_missing_input_is_named_in_plain_words():
    prices, load, weather, points = known_on(make_world(), DAY)
    midnight = DAY.tz_localize(TZ)

    no_load = live.live_table(prices, load[load.index < midnight], weather, points, DAY)
    assert live.not_ready(no_load, DAY) == ["the load forecast for the day is not out"]

    no_weather = live.live_table(prices, load, weather[weather.index < midnight], points, DAY)
    assert live.not_ready(no_weather, DAY) == ["the weather forecast for the day is not out"]

    no_today = live.live_table(prices[prices.index < midnight - pd.Timedelta(days=1)], load, weather, points, DAY)
    assert live.not_ready(no_today, DAY) == ["a price from the past week is missing"]

    nothing = live.live_table(prices, load[load.index < midnight], weather[weather.index < midnight], points, DAY)
    assert live.not_ready(nothing, DAY) == ["the inputs stop before 2026-03-30"]


def test_too_little_history_is_a_reason_not_to_forecast():
    table = live.live_table(*known_on(make_world(start="2026-03-10"), DAY), DAY)
    assert live.not_ready(table, DAY) == ["only 13 complete days of history to learn from"]   # 17 to 29 March


def test_the_clocks_going_back_leave_no_gap():
    world = make_world(start="2025-09-01", end="2025-11-05")
    for day in ["2025-10-26", "2025-10-27"]:                   # the 25-hour day itself, and the day after it
        table = live.live_table(*known_on(world, day), day)
        assert live.not_ready(table, day) == [] and len(table[table.index >= day]) == 24
    
def test_the_page_has_24_hours_two_forecasts_and_a_plan_for_each():
    table = live.live_table(*known_on(make_world(), DAY), DAY)
    page = live.make_day(table, DAY, live.forecasters(FAST))
    assert list(page.columns) == ["gbm", "naive", "gbm_plan", "naive_plan"]
    assert list(page.index) == list(pd.date_range(DAY, periods=24, freq="h")) and page.index.name == "delivery_time"
    assert page.notna().all().all()
    assert (page[["gbm", "naive"]] == page[["gbm", "naive"]].round(2)).all().all()        # in cents, as prices are quoted
    week_ago = table.loc[page.index - pd.Timedelta(days=7), "price"].round(2)
    assert (page["naive"].to_numpy() == week_ago.to_numpy()).all()                        # same as last week


def test_the_live_forecast_is_the_one_the_replay_makes_later():
    world = make_world()
    page = live.make_day(live.live_table(*known_on(world, DAY), DAY), DAY, live.forecasters(FAST))

    prices, load, weather, points = world                      # later, with every price published: Step 4's table, Step 5's replay
    features, _ = bf.complete_days(bf.build_features(prices, load, weather, points))
    replayed = rp.replay(features, [DAY], gbm.make_gbm(FAST))
    assert (page["gbm"] == replayed.round(2)).all()


def test_each_plan_follows_its_own_forecast_and_stays_inside_the_limits():
    table = live.live_table(*known_on(make_world(), DAY), DAY)
    page = live.make_day(table, DAY, live.forecasters(FAST))
    for name in ["gbm", "naive"]:
        plan = page[f"{name}_plan"].to_numpy()
        assert bat.broken_limits(np.maximum(-plan, 0), np.maximum(plan, 0)) == []
        charge, discharge = bat.plan_battery(page[name].to_numpy())
        assert np.allclose(plan, discharge - charge)           # anyone can remake the plan from the forecast in the file

def test_a_page_comes_back_as_written_with_the_moment_it_was_made(tmp_path):
    page, _ = make_page()
    made_at = pd.Timestamp("2026-03-29 10:17:42", tz=TZ)
    path = live.write_day(page, DAY, made_at, tmp_path / "live")
    assert path == tmp_path / "live" / "2026-03-30.csv"
    assert "2026-03-29T08:17:42Z" in path.read_text()          # stamped in UTC
    back, moment = live.read_day(path)
    pd.testing.assert_frame_equal(back, page, check_freq=False)
    assert moment == made_at


def test_a_written_day_is_never_replaced_and_the_days_are_listed_in_order(tmp_path):
    page, _ = make_page()
    made_at = pd.Timestamp("2026-03-29 10:17", tz=TZ)
    live.write_day(page, DAY, made_at, tmp_path)
    with pytest.raises(AssertionError, match="never replaced"):
        live.write_day(page, DAY, made_at, tmp_path)
    live.write_day(page, DAY - pd.Timedelta(days=1), made_at, tmp_path)
    (tmp_path / live.SCORES).write_text("not a day\n")
    assert [file.name for file in live.day_files(tmp_path)] == ["2026-03-29.csv", "2026-03-30.csv"]

def test_a_day_can_be_scored_by_hand():
    page, actual = make_page()
    line = live.score_day(page, pd.Timestamp("2026-03-29 10:17", tz=TZ), actual)
    assert list(line) == ["made_at", "on_time", "mae_gbm", "mae_naive", "hits_gbm", "hits_naive",
                          "profit_perfect", "profit_gbm", "profit_naive"]
    assert line["made_at"] == "2026-03-29T08:17:00Z" and line["on_time"] is True
    assert line["mae_gbm"] == 0 and line["hits_gbm"] == 1 > line["hits_naive"]    # the gbm column is the real prices
    assert np.isclose(line["mae_naive"], (3 * 40 + 2 * 50) / 24)                  # the flat forecast misses the dip and the peak
    assert np.isclose(line["profit_perfect"], 100 * 2 * 0.94 - 10 * 2 / 0.94)     # sells what a full battery gives back, buys what fills it
    assert np.isclose(line["profit_gbm"], line["profit_perfect"]) and line["profit_naive"] == 0
    turned = live.score_day(page[["naive", "gbm", "naive_plan", "gbm_plan"]], pd.Timestamp("2026-03-29 10:17", tz=TZ), actual)
    assert turned["profit_perfect"] == line["profit_perfect"]                     # the ceiling comes from the real prices alone


def test_a_page_made_at_noon_or_later_is_marked_late():
    page, actual = make_page()
    assert live.score_day(page, pd.Timestamp("2026-03-29 11:59", tz=TZ), actual)["on_time"] is True
    assert live.score_day(page, pd.Timestamp("2026-03-29 12:00", tz=TZ), actual)["on_time"] is False


def test_a_day_is_scored_once_its_prices_are_out_and_only_once(tmp_path):
    world = make_world()
    next_day, later = DAY + pd.Timedelta(days=1), DAY + pd.Timedelta(days=2)
    for day in [DAY, next_day]:                                # two mornings, each writing its page an hour before noon
        table = live.live_table(*known_on(world, day), day)
        page = live.make_day(table, day, live.forecasters(FAST))
        live.write_day(page, day, live.auction_close(day) - pd.Timedelta(hours=1), tmp_path)

    # the morning of the 30th: the prices of the 30th are out, those of the 31st are not
    assert [file.name for file in live.unscored(tmp_path, next_day)] == ["2026-03-30.csv"]
    scores, new = live.score_new(tmp_path, known_on(world, next_day)[0], next_day)
    assert new == [DAY] and list(scores.index) == [DAY] and scores.index.name == "delivery_date"
    assert scores.loc[DAY, "on_time"] and scores.loc[DAY, "mae_gbm"] < scores.loc[DAY, "mae_naive"]
    scores.to_csv(tmp_path / live.SCORES)

    again, new = live.score_new(tmp_path, known_on(world, next_day)[0], next_day)         # a second run the same morning
    assert new == []
    pd.testing.assert_frame_equal(again, scores)

    early, new = live.score_new(tmp_path, known_on(world, next_day)[0], later)            # asked about the 31st before its prices are out
    assert new == [] and list(early.index) == [DAY]

    both, new = live.score_new(tmp_path, known_on(world, later)[0], later)                # the morning after
    assert new == [next_day] and list(both.index) == [DAY, next_day]
    pd.testing.assert_frame_equal(both.loc[[DAY]], scores)     # the first line is left as it was

def test_the_summary_counts_only_the_days_made_in_time():
    text = live.summary(make_scores())
    assert "2 days forecast before the auction closed, from 2026-03-30 to 2026-04-02, with 2 missed in between" in text
    assert "error was 50% of the naive forecast's" in text                        # (6 + 18) / (20 + 28)
    assert "captured 80% of what a perfect forecast would have earned" in text    # (90 + 230) / (100 + 300)
    assert "EUR 3,650 a year more than the naive forecast" in text                # 10 a day more, times 365
    assert "for a battery of 1 MW and 2 MWh" in text and text.endswith("1 made after the auction closed is left out.")


def test_the_summary_is_honest_about_nothing_yet_and_about_falling_behind():
    assert live.summary(live.read_scores("no/such/folder")) == "No forecast made before its auction closed has been scored yet."
    assert live.summary(make_scores().iloc[[1]]).endswith("yet. 1 made after the auction closed is left out.")
    one_day = live.summary(make_scores().iloc[[0]])
    assert "1 day forecast before the auction closed, from 2026-03-30 to 2026-03-30, with none missed in between" in one_day
    behind = make_scores().assign(profit_gbm=[60.0, 0.0, 220.0])
    assert "EUR 3,650 a year less than the naive forecast" in live.summary(behind)