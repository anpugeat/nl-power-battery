"""our own forecast. Choose its settings on a practice exam, freeze them, then replay the test period.

Run with:  python -m src.models.gbm
"""
import itertools
import json
import os
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

import src.backtest.replay as rp

TARGET = rp.TARGET                            # the column being forecast
TEST_START = rp.TEST_START                    # the real exam starts here; nothing from it may shape the settings
VALID_START = "2024-09-01"                    # the practice exam runs from here to the day before TEST_START
WINDOW = 364                                  # the most days of history the model learns from, as for the textbook method
SETTINGS = "config/model.json"                # the chosen settings; the test period stays shut until this file exists
FIXED = {"learning_rate": 0.05, "random_state": 0, "deterministic": True, "force_row_wise": True, "verbose": -1}
GRID = {"n_estimators": (300, 600), "num_leaves": (15, 31, 63), "objective": ("l2", "l1")}
CALENDAR = ("hour", "day_of_week", "month", "is_holiday")
EVENING = (17, 18, 19, 20)                    # the evening peak: the hours starting 17:00 to 20:00
WORST = 10                                    # how many of the worst days to list
ALPHA = 0.05                                  # a lead counts as real when luck would explain it less often than this

def learning_rows(known, day, window=WINDOW):
    """The examples the model may learn from: priced hours before the day, no further back than the window."""
    day = pd.Timestamp(day)
    past = known[(known.index < day) & (known.index >= day - pd.Timedelta(days=window))]
    return past[past[TARGET].notna()]


def make_gbm(settings, columns=None, window=WINDOW):
    """A forecaster of the same shape as naive_day: it learns from the days before, then answers for the day."""

    def forecast(known, day):
        day = pd.Timestamp(day)
        clues = list(columns) if columns is not None else [name for name in known.columns if name != TARGET]
        past = learning_rows(known, day, window)
        model = lgb.LGBMRegressor(**FIXED, **settings).fit(past[clues], past[TARGET])
        return model.predict(known.loc[known.index >= day, clues])

    return forecast

def name_of(settings):
    """A short name for one set of settings, used in tables and file names."""
    return f"trees{settings['n_estimators']}_leaves{settings['num_leaves']}_{settings['objective']}"


def candidates(grid=GRID):
    """Every combination in the grid, simplest first."""
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def practice_days(table):
    """The delivery days of the practice exam: from VALID_START up to, and never including, TEST_START."""
    days = rp.days_from(table, VALID_START)
    return days[days < pd.Timestamp(TEST_START)]

def saved_gbm(settings, columns=None, tag="all", folder=rp.SAVED):
    """The model as a forecaster whose answers are kept on disk, in a file named after everything that shapes them."""
    return rp.kept_on_disk(make_gbm(settings, columns), f"{folder}/gbm_{WINDOW}_{name_of(settings)}_{tag}.csv")


def try_out(table, days, entrants):
    """Replay every entrant over the same days, beside the real price and same-as-last-week."""
    forecasts = pd.DataFrame({TARGET: table.loc[table.index.normalize().isin(days), TARGET]})
    forecasts["naive"] = rp.replay(table, days, rp.naive_day)
    for count, (name, forecaster) in enumerate(entrants.items(), start=1):
        print(f"  {count} of {len(entrants)}: {name}")
        forecasts[name] = rp.replay(table, days, forecaster)
    return forecasts


def choose(scores):
    """The entrant with the lowest average error. On a tie, the one listed first, which is the simpler."""
    return scores.drop(index="naive")["mae"].idxmin()

def clue_groups(columns):
    """Sort every clue into one of four groups, in the order the ladder adds them."""
    clues = [name for name in columns if name != TARGET]
    groups = {"lags": [name for name in clues if name.startswith("price_lag_")],
              "calendar": [name for name in clues if name in CALENDAR],
              "load": [name for name in clues if name.startswith("load_fc_")]}
    placed = [name for group in groups.values() for name in group]
    groups["weather"] = [name for name in clues if name not in placed]
    return groups


def ladder(groups):
    """Rungs that each add one group of clues to the rung below: lags, +calendar, +load, +weather."""
    rungs, so_far = {}, []
    for name, members in groups.items():
        so_far = so_far + members
        rungs[name if not rungs else f"+{name}"] = list(so_far)
    return rungs


def practice(table, days, grid=GRID, folder=rp.SAVED):
    """The first run: try every candidate on the practice days, pick one, and climb the ladder with it."""
    tried = {name_of(settings): settings for settings in candidates(grid)}
    tuning = rp.score_table(try_out(table, days, {name: saved_gbm(settings, folder=folder)
                                                  for name, settings in tried.items()}))
    winner = choose(tuning)
    rungs = ladder(clue_groups(table.columns))
    top = list(rungs)[-1]                                  # the top rung is the winner itself, so its file is reused
    ablation = rp.score_table(try_out(table, days, {
        rung: saved_gbm(tried[winner], folder=folder) if rung == top
        else saved_gbm(tried[winner], columns, f"upto_{rung.strip('+')}", folder)
        for rung, columns in rungs.items()}))
    return winner, tried[winner], tuning, ablation

def write_settings(path, name, settings, days, practice_mae):
    """Write the chosen settings down, with a note of when and on what they were chosen."""
    record = {"name": name, "settings": settings, "window": WINDOW,
              "chosen_on": f"{days[0].date()} to {days[-1].date()}", "practice_days": len(days),
              "practice_mae": round(float(practice_mae), 3)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
    return record


def read_settings(path):
    """Read the frozen settings back, and refuse them if the code has changed underneath them."""
    record = json.loads(Path(path).read_text())
    assert record["window"] == WINDOW, f"{path} was chosen with a window of {record['window']} days; WINDOW is now {WINDOW}"
    assert record["name"] == name_of(record["settings"]), f"{path}: the name does not match the settings"
    return record

def error_slices(forecasts):
    """Average error of every forecast, cut four ways: overall, by hour of day, by month and by price regime."""
    actual = forecasts[TARGET]
    errors = forecasts.drop(columns=TARGET).sub(actual, axis=0).abs()
    hour = errors.index.hour
    cuts = {"all": pd.Series("all hours", index=errors.index),
            "hour": pd.Series(hour, index=errors.index),
            "month": pd.Series(errors.index.strftime("%Y-%m"), index=errors.index),
            "regime": pd.Series("other hours", index=errors.index).mask(hour.isin(EVENING), "evening peak")
                                                                  .mask(actual < 0, "negative price")}
    pieces = []
    for cut, labels in cuts.items():
        piece = errors.groupby(labels).mean()
        piece.insert(0, "hours", labels.value_counts())
        pieces.append(piece.rename_axis("slice").reset_index().assign(cut=cut))
    table = pd.concat(pieces, ignore_index=True)
    return table[["cut", "slice", "hours", *errors.columns]]


def worst_days(forecasts, name="gbm", n=WORST):
    """The n days on which one forecast missed by most, with the other forecasts and the day's prices beside it."""
    day = forecasts.index.normalize().rename("delivery_date")
    actual = forecasts[TARGET]
    table = forecasts.drop(columns=TARGET).sub(actual, axis=0).abs().groupby(day).mean()
    table["mean_price"] = actual.groupby(day).mean()
    table["lowest_price"] = actual.groupby(day).min()
    return table.sort_values(name, ascending=False).head(n)

def as_days(series):
    """One row per day and 24 columns: the shape the significance test expects."""
    values = series.to_numpy(dtype=float)
    assert len(values) % 24 == 0, "the forecasts are not whole days"
    return values.reshape(-1, 24)


def edge(forecasts, challenger="gbm", reference="lear", test=None):
    """How likely luck is to explain a lead: first the challenger's over the reference, then the other way round."""
    if test is None:
        from epftoolbox.evaluation import GW               # imported here because it takes several seconds

        def test(real, first, second):
            return GW(p_real=real, p_pred_1=first, p_pred_2=second, norm=1, version="multivariate")

    real, new, old = (as_days(forecasts[name]) for name in (TARGET, challenger, reference))
    return float(test(real, old, new)), float(test(real, new, old))


def verdict(challenger_p, reference_p, challenger="gbm", reference="lear", alpha=ALPHA):
    """The two p-values as one plain sentence."""
    if challenger_p < alpha:
        return f"{challenger} is significantly more accurate than {reference} (p = {challenger_p:.3f})"
    if reference_p < alpha:
        return f"{reference} is significantly more accurate than {challenger} (p = {reference_p:.3f})"
    return f"no significant difference between {challenger} and {reference} (p = {challenger_p:.3f} and {reference_p:.3f})"

def read_baseline(engine):
    """Read the forecasts saved by Step 5, with delivery_time as the index."""
    table = pd.read_sql_table("baseline_forecasts", engine, index_col="delivery_time")
    table.index = pd.to_datetime(table.index)
    return table.sort_index()


def main():
    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])
    features = rp.read_features(engine)
    assert features.notna().all().all(), "blank values in features"
    assert (features.groupby(features.index.normalize()).size() == 24).all(), "a day in features does not have 24 rows"
    Path("reports").mkdir(exist_ok=True)

    if not Path(SETTINGS).exists():                        # first run: choose the settings on the practice exam
        days = practice_days(features)
        assert len(days) > 0, f"features has no days from {VALID_START} up to {TEST_START}"
        history = (days[0] - features.index.min().normalize()).days
        assert history >= 28, f"the model needs at least four weeks before {VALID_START}, features has {history} days"
        print(f"note: practice exam of {len(days)} days, from {days[0].date()} to {days[-1].date()}. "
              f"The model starts it with {history} days of history")
        print(f"candidates: about a second a day each the first time, then reused from {rp.SAVED}")
        winner, settings, tuning, ablation = practice(features, days)
        assert tuning.notna().all().all() and ablation.notna().all().all(), "a practice score is blank"
        assert tuning.loc["naive", "rmae"] == 1, "the baseline does not score exactly 1 against itself"

        print(tuning.round(3).to_string())
        print(f"chosen: {winner}")
        print(ablation.round(3).to_string())
        if tuning.loc[winner, "rmae"] >= 1:
            print("note: no candidate beat same-as-last-week on the practice exam. Look into this before freezing")
        tuning.to_csv("reports/tuning_scores.csv", index_label="candidate")
        ablation.to_csv("reports/ablation_scores.csv", index_label="clues")
        write_settings(SETTINGS, winner, settings, days, tuning.loc[winner, "mae"])
        print(f"saved {SETTINGS} and two tables in reports")
        print("note: the test period has not been touched. Commit and tag the settings, then run this command again")
        return

    record = read_settings(SETTINGS)                       # second run: the settings are fixed, so take the exam
    baseline = read_baseline(engine)
    days = rp.days_from(features, TEST_START)
    assert len(days) > 0, f"features has no days on or after {TEST_START}"
    actual = features.loc[features.index >= days[0], TARGET]
    assert baseline.index.equals(actual.index), "baseline_forecasts does not cover the same hours as features. Rerun Step 5's command"
    assert (baseline[TARGET] - actual).abs().max() < 1e-9, "baseline_forecasts holds different prices from features. Rerun Step 5's command"
    assert baseline.notna().all().all(), "blank values in baseline_forecasts"

    print(f"note: settings frozen in {SETTINGS}: {record['name']}, chosen on {record['chosen_on']}")
    print(f"note: replaying {len(days)} days, from {days[0].date()} to {days[-1].date()}")
    forecasts = baseline.copy()
    forecasts["gbm"] = rp.replay(features, days, saved_gbm(record["settings"]), every=50)
    assert forecasts.notna().all().all(), "a forecast is blank"
    assert (forecasts.groupby(forecasts.index.normalize()).size() == 24).all(), "a forecast day does not have 24 rows"
    scores = rp.score_table(forecasts)
    assert scores.loc["naive", "rmae"] == 1, "the baseline does not score exactly 1 against itself"
    slices = error_slices(forecasts)
    worst = worst_days(forecasts)
    challenger_p, reference_p = edge(forecasts)

    print(scores.round(3).to_string())
    print(slices[slices["cut"] == "regime"].drop(columns="cut").round(3).to_string(index=False))
    print(worst.round(1).to_string())
    print(f"significance: {verdict(challenger_p, reference_p)}")
    forecasts.to_sql("forecasts", engine, if_exists="replace", index_label="delivery_time")
    scores.to_csv("reports/model_scores.csv", index_label="forecast")
    slices.to_csv("reports/model_errors.csv", index=False)
    worst.to_csv("reports/model_worst_days.csv")
    pd.DataFrame({"claim": ["gbm beats lear", "lear beats gbm"], "p_value": [challenger_p, reference_p]}
                 ).to_csv("reports/model_significance.csv", index=False)
    print(f"saved {len(forecasts)} rows for {len(days)} days")


if __name__ == "__main__":
    main()