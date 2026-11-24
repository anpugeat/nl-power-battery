"""Step 8: the what-if experiment. Replay Step 7's game with other batteries, and see which assumptions move the money.

Run with:  python -m src.doe.whatif
"""
import hashlib
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

import src.dispatch.battery as bat

TARGET = bat.TARGET                           # the real price: every plan is paid at it
LEVELS = {"duration": (1.0, 4.0),             # hours the battery can run at full power: its capacity divided by its power
          "efficiency": (0.80, 0.92),         # share of the energy bought that comes back out
          "wear": (0.0, 20.0),                # EUR charged for every MWh moved in or out
          "cycles": (1.0, 2.0),               # full charges allowed in a day
          "forecast": ("naive", "gbm")}       # the column of forecasts the plans are made from
DIALS = ("duration", "efficiency", "wear")    # the factors that can also take values in between: the map uses two of them
BASE = {"duration": bat.BATTERY["capacity"] / bat.BATTERY["power"],    # Step 7's battery, described the way LEVELS is
        "efficiency": bat.BATTERY["eta_c"] * bat.BATTERY["eta_d"],
        "wear": bat.BATTERY["wear"],
        "cycles": bat.BATTERY["max_cycles"]}
WORDS = {"duration": "{:g} to {:g} hours of storage",                  # each factor's two levels, in words
         "efficiency": "{:.0%} to {:.0%} efficiency",
         "wear": "EUR {:g} to {:g} of wear per MWh moved",
         "cycles": "{:g} to {:g} cycles a day",
         "forecast": "the {} forecast to the {} forecast"}
STEPS = 5                                     # values of each dial on the map, so it has STEPS times STEPS points
DRAWS = 200                                   # how many times the weeks are drawn again for the intervals
COVERAGE = 0.90                               # the share of those draws an interval holds
MIN_DAYS = 180                                # a calendar year needs this many days to stand for a year
SAVED = "data/whatif"                         # every run's daily profits are kept here as they are made

def battery_for(setting, power=bat.BATTERY["power"]):
    """The battery a setting describes, as the bundle of six numbers Step 7's planner takes."""
    each_way = float(setting["efficiency"]) ** 0.5                      # the same loss on the way in as on the way out
    return {"power": power, "capacity": float(setting["duration"]) * power, "eta_c": each_way, "eta_d": each_way,
            "wear": float(setting["wear"]), "max_cycles": float(setting["cycles"])}


def name_of(setting):
    """A name such as gbm_hours4_eff92_wear20_cycles2. The forecast comes first when the setting has one."""
    name = (f"hours{setting['duration']:g}_eff{setting['efficiency'] * 100:g}_"
            f"wear{setting['wear']:g}_cycles{setting['cycles']:g}")
    return f"{setting['forecast']}_{name}" if "forecast" in setting else name


def design(levels=LEVELS):
    """Every combination of the levels: one named row per run, one column per factor."""
    rows = [dict(zip(levels, combination)) for combination in itertools.product(*levels.values())]
    return pd.DataFrame(rows, index=pd.Index([name_of(row) for row in rows], name="run"))

def fingerprint(table):
    """A short code that changes whenever any value or any hour in the table changes."""
    return hashlib.sha1(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()).hexdigest()[:8]


def saved_play(forecasts, setting, folder=SAVED):
    """One run's daily profits: read from disk if this exact game was played before, otherwise played and saved."""
    seen = setting["forecast"]
    days = forecasts.index.normalize().unique().rename("delivery_date")
    path = Path(folder) / f"{name_of(setting)}_{fingerprint(forecasts[[TARGET, seen]])}.csv"
    if path.exists():
        saved = pd.read_csv(path, index_col="delivery_date", parse_dates=True, converters={"profit": float})["profit"]
        if len(saved) == len(days) and (saved.index == days).all() and saved.notna().all():
            return pd.Series(saved.to_numpy(), index=days, name="profit")
    profit = bat.play(forecasts, seen, battery_for(setting))[1]
    profit = pd.Series(profit.to_numpy(), index=days, name="profit")
    path.parent.mkdir(parents=True, exist_ok=True)
    profit.to_csv(path)
    return profit


def play_design(forecasts, plan, folder=SAVED):
    """Every run of the plan on the same days. Returns the daily profits, one column per run."""
    daily = {}
    for number, (name, setting) in enumerate(plan.iterrows(), start=1):
        print(f"  {number} of {len(plan)}: {name}")
        daily[name] = saved_play(forecasts, setting, folder)
    return pd.DataFrame(daily)

def long_years(days, min_days=MIN_DAYS):
    """The calendar years with enough days to stand for a year, and the years without."""
    counts = pd.Series(1, index=days).groupby(days.year).sum()
    return [int(year) for year in counts.index[counts >= min_days]], [int(year) for year in counts.index[counts < min_days]]


def yearly(daily):
    """Each run's profit per year (EUR), one row per calendar year: that year's average day, times 365."""
    return (daily.groupby(daily.index.year).mean() * 365).rename_axis("year")


def run_table(plan, daily):
    """One row per run and year: the run's settings, the days played, the money made and the profit per year."""
    parts = []
    for year, days in daily.groupby(daily.index.year):
        part = plan.copy()
        part["year"], part["days"], part["profit"], part["per_year"] = int(year), len(days), days.sum(), days.mean() * 365
        parts.append(part)
    return pd.concat(parts).set_index("year", append=True)

def contrasts(plan, levels=LEVELS):
    """Plus one where a factor is at its high level and minus one where it is low, for every factor and every pair."""
    for factor in levels:
        assert plan[factor].isin(levels[factor]).all(), f"the plan has a value of {factor} that is not one of its two levels"
    signs = pd.DataFrame({factor: np.where(plan[factor] == levels[factor][1], 1.0, -1.0) for factor in levels}, index=plan.index)
    for first, second in itertools.combinations(levels, 2):
        signs[f"{first} x {second}"] = signs[first] * signs[second]
    return signs


def effects(per_year, plan, levels=LEVELS):
    """What each factor and each pair does to the yearly profit: the average of its high runs minus that of its low runs."""
    signs = contrasts(plan, levels)
    by_year = per_year[signs.index] @ signs / (len(signs) / 2)          # one row per year, one column per factor or pair
    table = by_year.T.add_prefix("in_")
    table.insert(0, "effect", by_year.mean())                           # every year counts the same
    return table.rename_axis("term")

def week_blocks(days):
    """The row numbers of each week, year by year. A week is seven days counted from 1 January."""
    rows = pd.Series(np.arange(len(days)), index=days)
    return {int(year): [week.to_numpy() for _, week in in_year.groupby((in_year.index.day_of_year - 1) // 7)]
            for year, in_year in rows.groupby(days.year)}


def intervals(daily, plan, levels=LEVELS, draws=DRAWS, coverage=COVERAGE, seed=0):
    """How far each effect moves when the weeks are drawn again at random: the range holding `coverage` of the draws."""
    rng = np.random.default_rng(seed)
    blocks = week_blocks(daily.index)
    found = []
    for _ in range(draws):
        rows = np.concatenate([np.concatenate([weeks[pick] for pick in rng.integers(len(weeks), size=len(weeks))])
                               for weeks in blocks.values()])
        found.append(effects(yearly(daily.iloc[rows]), plan, levels)["effect"])
    found = pd.DataFrame(found)
    return pd.DataFrame({"low": found.quantile((1 - coverage) / 2), "high": found.quantile((1 + coverage) / 2)}).rename_axis("term")

def strongest(table, among=DIALS, n=2):
    """The n dials whose effect is largest, whichever way it points, listed in the order of `among`."""
    size = table.loc[list(among), "effect"].abs().sort_values(ascending=False, kind="stable")
    return [dial for dial in among if dial in size.index[:n]]


def grid(first, second, seen, steps=STEPS, levels=LEVELS, base=BASE):
    """The settings for the map: evenly spaced values of two dials, with the other numbers as in Step 7's battery."""
    values = {dial: np.round(np.linspace(*levels[dial], steps), 6) for dial in (first, second)}
    return [{**base, first: float(one), second: float(other), "forecast": seen}
            for one in values[first] for other in values[second]]


def surface(forecasts, first, second, seen, folder=SAVED, **choices):
    """The profit per year at every point of the map, one row per point."""
    points = grid(first, second, seen, **choices)
    rows = []
    for number, setting in enumerate(points, start=1):
        print(f"  {number} of {len(points)}: {name_of(setting)}")
        profit = saved_play(forecasts, setting, folder)
        rows.append({first: setting[first], second: setting[second], "per_year": yearly(profit.to_frame())["profit"].mean()})
    return pd.DataFrame(rows)

def sentence(table, levels=LEVELS, coverage=COVERAGE, power=bat.BATTERY["power"]):
    """The step's findings in words: the factor that moves the yearly profit most, and what the better forecast is worth."""
    top = table.loc[list(levels), "effect"].abs().idxmax()
    gain, low, high = table.loc["forecast", ["effect", "low", "high"]]
    worse, better = levels["forecast"]
    text = (f"For a battery of {power:g} MW, the profit per year is most sensitive to {top}: going from "
            f"{WORDS[top].format(*levels[top])} changes it by EUR {table.loc[top, 'effect']:+,.0f}. "
            f"Going from the {worse} forecast to the {better} forecast {'adds' if gain >= 0 else 'costs'} "
            f"EUR {abs(gain):,.0f} a year ({coverage:.0%} interval EUR {low:,.0f} to {high:,.0f})")
    least, most = levels["efficiency"]
    per_point = table.loc["efficiency", "effect"] / ((most - least) * 100)   # EUR a year for one more point of efficiency
    if gain > 0 and per_point > 0:
        points = gain / per_point
        if points <= (most - least) * 100:
            text += f", as much as {points:.1f} more points of efficiency"
        else:
            text += f", more than raising efficiency from {least:.0%} to {most:.0%}"
    return text + "."

def main():
    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])
    forecasts = bat.read_forecasts(engine)
    needed = [TARGET, *LEVELS["forecast"]]
    assert set(needed) <= set(forecasts.columns), f"forecasts has no column called: {sorted(set(needed) - set(forecasts.columns))}"
    assert forecasts[needed].notna().all().all(), "blank values in forecasts"
    assert (forecasts.groupby(forecasts.index.normalize()).size() == 24).all(), "a day in forecasts does not have 24 rows"

    years, short = long_years(forecasts.index.normalize().unique())
    assert years, f"no calendar year in forecasts has {MIN_DAYS} days"
    forecasts = forecasts[forecasts.index.year.isin(years)]
    days = forecasts.index.normalize().unique()
    print(f"note: {len(days)} days: {', '.join(f'{(days.year == year).sum()} in {year}' for year in years)}")
    if short:
        print(f"note: left out for having fewer than {MIN_DAYS} days: {', '.join(str(year) for year in short)}")
    plan = design()
    print(f"note: {len(plan)} combinations of {', '.join(LEVELS)}. Each is played once and kept in {SAVED}")
    daily = play_design(forecasts, plan)

    assert daily.notna().all().all(), "a daily profit is blank"
    assert daily.index.equals(days) and list(daily.columns) == list(plan.index), "a day or a combination was not played"
    runs = run_table(plan, daily)
    found = effects(yearly(daily), plan)
    table = pd.concat([found[["effect"]], intervals(daily, plan), found.drop(columns="effect")], axis=1)
    table = table.sort_values("effect", key=abs, ascending=False)
    assert table.notna().all().all() and (table["low"] <= table["high"]).all(), "an effect or its interval is blank"
    first, second = strongest(table)
    seen = LEVELS["forecast"][1]
    print(f"note: the two strongest dials are {first} and {second}. Mapping them for the {seen} forecast")
    area = surface(forecasts, first, second, seen)
    assert area["per_year"].notna().all(), "a point of the map is blank"
    text = sentence(table)

    print(table.round(0).to_string())
    unsure = list(table.index[(table["low"] < 0) & (table["high"] > 0)])
    if unsure:
        print(f"note: an interval that includes zero means the weeks do not agree on the direction. That holds for: {', '.join(unsure)}")
    print(area.pivot(index=first, columns=second, values="per_year").round(0).to_string())
    print(text)
    daily.to_sql("whatif_daily", engine, if_exists="replace", index_label="delivery_date")
    Path("reports").mkdir(exist_ok=True)
    runs.to_csv("reports/whatif_runs.csv")
    table.to_csv("reports/whatif_effects.csv")
    area.to_csv("reports/whatif_surface.csv", index=False)
    Path("reports/whatif_sentence.txt").write_text(text + "\n")
    print(f"saved {len(daily)} days for {len(plan)} combinations, and {len(area)} points of the map")


if __name__ == "__main__":
    main()