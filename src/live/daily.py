"""go live. Each morning, forecast tomorrow before the auction closes, write it down once, and score the days whose prices are out.

Run with:  python -m src.live.daily
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import src.backtest.replay as rp
import src.dispatch.battery as bat
import src.features.build_features as bf
import src.ingest.openmeteo as om
import src.models.gbm as gbm
from src.ingest.entsoe_inputs import hourly_utc
from src.ingest.prices import to_hourly

TARGET = rp.TARGET                            # the real price: blank for the live day until its auction has run
TZ = bf.TZ                                    # the auction's clock
CLOSE_HOUR = 12                               # bidding for a day ends at 12:00 on the day before
LOUD_FROM = 10                                # from this hour, a morning that cannot forecast stops with an error
RECORD = "live"                               # the public record: only the scheduled job on GitHub writes here
REHEARSAL = "data/live_rehearsal"             # where the same command writes on any other machine
SCORES = "scores.csv"                         # the scoreboard, one line per scored day, kept beside the days
SUMMARY = "summary.txt"                       # the scoreboard in words
ZONES = {"NL": "load_fc_nl", "DE_LU": "load_fc_de"}    # the load forecasts to fetch, and the column each one becomes
MIN_DAYS = 28                                 # the fewest days of history the model may learn from

def delivery_day(now):
    """The day being forecast: tomorrow, by the Amsterdam clock."""
    return now.tz_convert(TZ).normalize().tz_localize(None) + pd.Timedelta(days=1)


def auction_close(day):
    """The moment bidding for a delivery day ends: 12:00 Amsterdam time on the day before."""
    return (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=CLOSE_HOUR)).tz_localize(TZ)


def history_start(day):
    """The first day to fetch: a month start, early enough for a full window of learning days and their week-old prices."""
    return (pd.Timestamp(day) - pd.Timedelta(days=gbm.WINDOW + max(bf.LAGS) // 24 + 1)).replace(day=1)


def record_folder(environment):
    """Where the days are written: the public record on GitHub's machine, a rehearsal folder anywhere else."""
    return RECORD if environment.get("GITHUB_ACTIONS") == "true" else REHEARSAL

def fetch_prices(client, first, day):
    """Hourly prices in UTC, from the first day to the end of the day before the delivery day."""
    start = pd.Timestamp(first).tz_localize(TZ)
    end = pd.Timestamp(day).tz_localize(TZ) - pd.Timedelta(minutes=15)
    return to_hourly(client.query_day_ahead_prices("NL", start=start, end=end)).to_frame()


def fetch_load(client, first, day, zones=ZONES):
    """The grid operators' load forecasts, hourly in UTC, from the first day to the end of the delivery day."""
    start = pd.Timestamp(first).tz_localize(TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    return pd.DataFrame({column: hourly_utc(client.query_load_forecast(zone, start=start, end=end))["Forecasted Load"]
                         for zone, column in zones.items()})


def fetch_weather(points, first, day, ask=om.past_forecast, pause=1.0):
    """The weather forecasts as they stood 48 hours ahead, for every point, from the first day to the end of the delivery day."""
    tables = []
    for point in points.itertuples():
        pieces = []
        for start, end in om.chunks(first, day):
            pieces.append(ask(point.lat, point.lon, start, end))
            time.sleep(pause)
        tables.append(pd.concat(pieces).add_prefix(f"{point.name}_"))
    return pd.concat(tables, axis=1)

def live_table(prices, load, weather, points, day):
    """Step 4's table for a live day: every complete day before it, then the day itself with its price still blank."""
    day = pd.Timestamp(day)
    full = bf.build_features(prices, load, weather, points)
    past, _ = bf.complete_days(full[full.index < day])
    return pd.concat([past, full[(full.index >= day) & (full.index < day + pd.Timedelta(days=1))]])


def not_ready(table, day):
    """The reasons the day cannot be forecast yet, in plain words. An empty list means everything is in."""
    day = pd.Timestamp(day)
    ahead = table[table.index >= day]
    if len(ahead) != 24:
        return [f"the inputs stop before {day.date()}"]
    waiting = {"lags": "a price from the past week is missing",
               "load": "the load forecast for the day is not out",
               "weather": "the weather forecast for the day is not out"}
    groups = gbm.clue_groups(table.columns)
    reasons = [words for group, words in waiting.items() if ahead[groups[group]].isna().any().any()]
    days = gbm.learning_rows(table, day).index.normalize().nunique()
    if days < MIN_DAYS:
        reasons.append(f"only {days} complete days of history to learn from")
    return reasons

def forecasters(settings):
    """The forecasts written down each morning: your frozen model, and same-as-last-week to measure it against."""
    return {"gbm": gbm.make_gbm(settings), "naive": rp.naive_day}


def make_day(table, day, makers, battery=bat.BATTERY):
    """The day's page: each forecast for its 24 hours, in cents, and the battery plan made from it."""
    page = pd.DataFrame({name: rp.replay(table, [day], maker).round(2) for name, maker in makers.items()})
    for name in makers:
        charge, discharge = bat.plan_battery(page[name].to_numpy(), battery)
        broken = bat.broken_limits(charge, discharge, battery)
        assert not broken, f"the plan made from {name} for {pd.Timestamp(day).date()} {' and '.join(broken)}"
        page[f"{name}_plan"] = discharge - charge
    return page

def stamp(moment):
    """A moment written the same way everywhere: UTC, to the second."""
    return pd.Timestamp(moment).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def day_file(folder, day):
    """The file a delivery day is written to, named after the day."""
    return Path(folder) / f"{pd.Timestamp(day):%Y-%m-%d}.csv"


def day_files(folder):
    """Every day written so far, oldest first."""
    return sorted(Path(folder).glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].csv"))


def write_day(page, day, made_at, folder):
    """Write a day's page to its file, stamped with the moment it was made. A day once written is never written again."""
    path = day_file(folder, day)
    assert not path.exists(), f"{path} is already written, and a forecast is never replaced"
    path.parent.mkdir(parents=True, exist_ok=True)
    page.assign(made_at=stamp(made_at)).to_csv(path)
    return path


def read_day(path):
    """Read a written day back: its page, and the moment it was made."""
    page = pd.read_csv(path, index_col="delivery_time", parse_dates=True)
    return page.drop(columns="made_at"), pd.Timestamp(page["made_at"].iloc[0])

def score_day(page, made_at, actual, battery=bat.BATTERY):
    """One line of the scoreboard: when the page was made, how far each forecast was off, and what each plan earned."""
    names = [column for column in page.columns if not column.endswith("_plan")]
    real = actual.to_numpy()
    line = {"made_at": stamp(made_at), "on_time": bool(made_at < auction_close(page.index[0].normalize()))}
    line.update({f"mae_{name}": rp.mae(actual, page[name]) for name in names})
    line.update({f"hits_{name}": rp.rank_hit_rate(actual, page[name]) for name in names})
    line["profit_perfect"] = bat.settle(real, *bat.plan_battery(real, battery), battery["wear"])
    for name in names:
        plan = page[f"{name}_plan"].to_numpy()                 # positive while selling, negative while buying
        line[f"profit_{name}"] = bat.settle(real, np.maximum(-plan, 0), np.maximum(plan, 0), battery["wear"])
    return line


def read_scores(folder):
    """The scoreboard so far, one line per scored day. It has no lines until the first day is scored."""
    path = Path(folder) / SCORES
    if not path.exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], name="delivery_date"))
    return pd.read_csv(path, index_col="delivery_date", parse_dates=True)


def unscored(folder, before):
    """The written days before a given day that have no line on the scoreboard yet."""
    scored = read_scores(folder).index
    days = {pd.Timestamp(file.stem): file for file in day_files(folder)}
    return [file for day, file in days.items() if day < pd.Timestamp(before) and day not in scored]


def score_new(folder, prices, before, battery=bat.BATTERY):
    """The scoreboard with a line added for every unscored day whose 24 real prices are out, and the days added."""
    scores = read_scores(folder)
    real = bf.to_grid(prices)[TARGET]
    lines = {}
    for file in unscored(folder, before):
        page, made_at = read_day(file)
        actual = real.reindex(page.index)
        if actual.notna().all():
            lines[pd.Timestamp(file.stem)] = score_day(page, made_at, actual, battery)
    if lines:
        scores = pd.concat([table for table in (scores, pd.DataFrame.from_dict(lines, orient="index")) if len(table)])
    return scores.sort_index().rename_axis("delivery_date"), sorted(lines)

def summary(scores, battery=bat.BATTERY, name="gbm", baseline="naive"):
    """The live record in words, counting only the days forecast before their auction closed."""
    kept = scores[scores["on_time"]] if len(scores) else scores
    late = len(scores) - len(kept)
    aside = f" {late} made after the auction closed {'is' if late == 1 else 'are'} left out." if late else ""
    if kept.empty:
        return "No forecast made before its auction closed has been scored yet." + aside
    rmae = kept[f"mae_{name}"].sum() / kept[f"mae_{baseline}"].sum()
    capture = kept[f"profit_{name}"].sum() / kept["profit_perfect"].sum()
    lead = (kept[f"profit_{name}"] - kept[f"profit_{baseline}"]).mean() * 365
    gaps = (kept.index[-1] - kept.index[0]).days + 1 - len(kept)
    return (f"Live record: {len(kept)} {'day' if len(kept) == 1 else 'days'} forecast before the auction closed, from "
            f"{kept.index[0].date()} to {kept.index[-1].date()}, with {gaps or 'none'} missed in between. "
            f"The {name} forecast's error was {rmae:.0%} of the {baseline} forecast's, and its plans captured {capture:.0%} "
            f"of what a perfect forecast would have earned. At that rate it earns EUR {abs(lead):,.0f} a year "
            f"{'more' if lead >= 0 else 'less'} than the {baseline} forecast, for a battery of {battery['power']:g} MW "
            f"and {battery['capacity']:g} MWh." + aside)

def main():
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError

    load_dotenv()
    now = pd.Timestamp.now(tz=TZ)
    day = delivery_day(now)
    folder = record_folder(os.environ)
    target = day_file(folder, day)
    print(f"note: {now:%Y-%m-%d %H:%M} in Amsterdam, delivery day {day.date()}, writing to {folder}"
          + ("" if folder == RECORD else " (a rehearsal)"))

    wanted = not target.exists() and not (folder == RECORD and now >= auction_close(day))
    if target.exists():
        print(f"note: {target} is already written")
    elif not wanted:
        print(f"note: the auction for {day.date()} closed at {CLOSE_HOUR}:00, so nothing is written to the record for it")
    if not wanted and not unscored(folder, day):
        print("note: no day is waiting for a score either. Nothing to do")
        return

    points = pd.read_csv("config/weather_points.csv")
    record = gbm.read_settings(gbm.SETTINGS)
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    first = history_start(day)
    prices, table, reasons = None, None, []
    try:
        prices = fetch_prices(client, first, day)
        if wanted:
            print(f"note: fetching load and weather forecasts from {first.date()}. This takes a minute or two")
            table = live_table(prices, fetch_load(client, first, day), fetch_weather(points, first, day), points, day)
            reasons = not_ready(table, day)
    except (requests.RequestException, NoMatchingDataError) as error:
        reasons = [f"a source did not answer ({type(error).__name__})"]

    if wanted and not reasons:                                 # first the job with a deadline
        page = make_day(table, day, forecasters(record["settings"]))
        assert len(page) == 24 and page.notna().all().all(), "the page is not 24 complete rows"
        made_at = pd.Timestamp.now(tz="UTC")
        learned = gbm.learning_rows(table, day).index.normalize().nunique()
        print(page.to_string())
        print(f"saved {write_day(page, day, made_at, folder)}, made at {made_at.tz_convert(TZ):%H:%M} Amsterdam time "
              f"by {record['name']} from {learned} days of history")
        if made_at >= auction_close(day):
            print("note: that is after the auction closed, so the day will be scored but left out of the summary")

    if prices is not None:                                     # then the marking, which can wait
        scores, new = score_new(folder, prices, day)
        if new:
            assert scores.notna().all().all(), "a score is blank"
            text = summary(scores)
            print(scores.loc[new].drop(columns="made_at").round(2).to_string())
            print(text)
            scores.to_csv(Path(folder) / SCORES)
            (Path(folder) / SUMMARY).write_text(text + "\n")
            print(f"saved {len(new)} new {'line' if len(new) == 1 else 'lines'} in {Path(folder) / SCORES}, {len(scores)} in all")

    if wanted and reasons:
        print(f"note: no forecast for {day.date()} yet: {'; '.join(reasons)}")
        if now.hour >= LOUD_FROM:
            raise SystemExit(f"{now:%H:%M} in Amsterdam and still no forecast for {day.date()}. See the note above")


if __name__ == "__main__":
    main()