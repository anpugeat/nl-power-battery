"""the battery. Plan each day from a forecast, settle the plan at the real prices, and count the money.

Run with:  python -m src.dispatch.battery
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import Bounds, LinearConstraint, milp
from sqlalchemy import create_engine

import src.backtest.replay as rp

TARGET = rp.TARGET                            # the real price: every plan is paid at it, whatever it was planned from
BATTERY = {"power": 1.0,                      # MW: the most it can charge or discharge in an hour
           "capacity": 2.0,                   # MWh: the most it can hold
           "eta_c": 0.94,                     # share of the energy bought that ends up stored
           "eta_d": 0.94,                     # share of the stored energy that comes back out
           "wear": 0.0,                       # EUR charged for every MWh moved in or out
           "max_cycles": 1.0}                 # full charges allowed in a day
PLAYERS = {"perfect": TARGET, "gbm": "gbm", "lear": "lear", "naive": "naive"}   # each player, and the column it plans from
WORST = 5                                     # how many of the costliest days to list
SLACK = 1e-6                                  # rounding room allowed when a limit or a profit is checked

def plan_battery(prices, battery=BATTERY):
    """The most profitable charge and discharge (MW in each hour) for one day of prices, inside the battery's limits."""
    price = np.asarray(prices, dtype=float)
    assert np.isfinite(price).all(), "a price to plan from is blank"
    power, capacity, eta_c, eta_d = battery["power"], battery["capacity"], battery["eta_c"], battery["eta_d"]
    n = len(price)
    each, none = np.eye(n), np.zeros((n, n))
    so_far = np.tril(np.ones((n, n)))                                # adds up everything up to and including each hour

    # the unknowns, in order: charge in each hour, discharge in each hour, and a switch per hour that is 1 while charging
    cost = np.concatenate([price + battery["wear"], battery["wear"] - price, np.zeros(n)])   # buying costs, selling earns
    stored_after = np.hstack([eta_c * so_far, -so_far / eta_d, none])                        # energy held after each hour
    limits = [LinearConstraint(stored_after, 0, capacity),                               # never below empty or above full
              LinearConstraint(stored_after[-1], 0, 0),                                  # end the day empty, as it began
              LinearConstraint(np.hstack([each, none, -power * each]), -np.inf, 0),      # charge only while the switch is on
              LinearConstraint(np.hstack([none, each, power * each]), -np.inf, power),   # discharge only while it is off
              LinearConstraint(np.concatenate([eta_c * np.ones(n), np.zeros(2 * n)]), 0, battery["max_cycles"] * capacity)]
    answer = milp(cost, constraints=limits, integrality=np.concatenate([np.zeros(2 * n), np.ones(n)]),
                  bounds=Bounds(0, np.concatenate([np.full(2 * n, power), np.ones(n)])), options={"mip_rel_gap": 0})
    assert answer.success, f"the solver found no plan: {answer.message}"
    amounts = np.round(answer.x[:2 * n], 9) + 0.0                    # tidy the solver's last decimals
    return amounts[:n], amounts[n:]

def stored(charge, discharge, battery=BATTERY):
    """Energy in the battery after each hour (MWh), for a day that starts empty."""
    return np.cumsum(battery["eta_c"] * np.asarray(charge, dtype=float) - np.asarray(discharge, dtype=float) / battery["eta_d"])


def broken_limits(charge, discharge, battery=BATTERY):
    """The limits a plan breaks, each as a few plain words. An empty list means the battery could really carry it out."""
    charge, discharge = np.asarray(charge, dtype=float), np.asarray(discharge, dtype=float)
    level = stored(charge, discharge, battery)
    checks = {"moves a negative amount": min(charge.min(), discharge.min()) < -SLACK,
              "moves more than its power": max(charge.max(), discharge.max()) > battery["power"] + SLACK,
              "charges and discharges in the same hour": np.minimum(charge, discharge).max() > SLACK,
              "goes below empty": level.min() < -SLACK,
              "goes above full": level.max() > battery["capacity"] + SLACK,
              "does not end the day empty": abs(level[-1]) > SLACK,
              "uses more than its cycles": battery["eta_c"] * charge.sum() > battery["max_cycles"] * battery["capacity"] + SLACK}
    return [name for name, broken in checks.items() if broken]

def settle(prices, charge, discharge, wear=0.0):
    """Money made (EUR) when a plan meets a day's real prices: sales, minus purchases, minus wear."""
    price = np.asarray(prices, dtype=float)
    charge, discharge = np.asarray(charge, dtype=float), np.asarray(discharge, dtype=float)
    return float(np.sum(price * (discharge - charge) - wear * (charge + discharge)))

def play(forecasts, seen, battery=BATTERY):
    """One player's whole game: plan every day from the column `seen`, then settle each plan at the real prices."""
    plans, profits = [], {}
    for day, hours in forecasts.groupby(forecasts.index.normalize()):
        assert len(hours) == 24, f"{day.date()} does not have 24 rows"
        charge, discharge = plan_battery(hours[seen].to_numpy(), battery)
        broken = broken_limits(charge, discharge, battery)
        assert not broken, f"the plan made from {seen} for {day.date()} {' and '.join(broken)}"
        plans.append(pd.Series(discharge - charge, index=hours.index))
        profits[day] = settle(hours[TARGET].to_numpy(), charge, discharge, battery["wear"])
    return pd.concat(plans), pd.Series(profits).rename_axis("delivery_date")


def play_all(forecasts, players=PLAYERS, battery=BATTERY):
    """Every player's game on the same days. Returns the hourly plans and the daily profits, one column per player."""
    plans, profits = {TARGET: forecasts[TARGET]}, {}
    for number, (name, seen) in enumerate(players.items(), start=1):
        print(f"  {number} of {len(players)}: {name}")
        plans[name], profits[name] = play(forecasts, seen, battery)
    return pd.DataFrame(plans), pd.DataFrame(profits)

def money_table(profits, capacity, ceiling="perfect"):
    """One row per player: total profit, profit per MWh of capacity per year, share of the ceiling's profit, and losing days."""
    total = profits.sum()
    return pd.DataFrame({"profit": total,
                         "per_mwh_year": total / len(profits) * 365 / capacity,
                         "capture_rate": total / total[ceiling],
                         "losing_days": (profits < -SLACK).sum()})


def headline(scores, days, battery=BATTERY, name="gbm", baseline="naive"):
    """The project's headline sentence, filled in from the scoreboard."""
    lead = (scores.loc[name, "per_mwh_year"] - scores.loc[baseline, "per_mwh_year"]) * battery["capacity"]
    return (f"Over {len(days)} days from {days[0].date()} to {days[-1].date()}, the {name} forecast captured "
            f"{scores.loc[name, 'capture_rate']:.0%} of what a perfect forecast would have earned, which is "
            f"EUR {abs(lead):,.0f} a year {'more' if lead >= 0 else 'less'} than the {baseline} forecast, "
            f"for a battery of {battery['power']:g} MW and {battery['capacity']:g} MWh.")
        
def worst_days(profits, name="gbm", n=WORST, ceiling="perfect"):
    """The n days on which one player fell furthest short of the ceiling, with every player's profit beside it."""
    table = profits.copy()
    table["shortfall"] = profits[ceiling] - profits[name]
    return table.sort_values("shortfall", ascending=False).head(n)

def read_forecasts(engine):
    """Read the table saved by Step 6's second run, with delivery_time as the index."""
    table = pd.read_sql_table("forecasts", engine, index_col="delivery_time")
    table.index = pd.to_datetime(table.index)
    return table.sort_index()


def main():
    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])
    forecasts = read_forecasts(engine)
    needed = [TARGET, *[column for column in PLAYERS.values() if column != TARGET]]
    assert set(needed) <= set(forecasts.columns), f"forecasts has no column called: {sorted(set(needed) - set(forecasts.columns))}"
    assert forecasts[needed].notna().all().all(), "blank values in forecasts"
    assert (forecasts.groupby(forecasts.index.normalize()).size() == 24).all(), "a day in forecasts does not have 24 rows"

    days = forecasts.index.normalize().unique()
    print(f"note: planning {len(days)} days, from {days[0].date()} to {days[-1].date()}, "
          f"for a battery of {BATTERY['power']:g} MW and {BATTERY['capacity']:g} MWh")
    holes = pd.date_range(days[0], days[-1], freq="D").difference(days)
    if len(holes):
        print(f"note: {len(holes)} days are missing from forecasts and are skipped: {', '.join(str(hole.date()) for hole in holes)}")
    print("players: a few seconds each")
    plans, profits = play_all(forecasts)

    assert plans.notna().all().all() and profits.notna().all().all(), "a plan or a profit is blank"
    assert plans.index.equals(forecasts.index) and len(profits) == len(days), "a day was not planned"
    behind = profits.drop(columns="perfect").sub(profits["perfect"], axis=0)
    assert (behind <= SLACK).all().all(), f"a player earned more than the crystal ball on {behind.max(axis=1).idxmax().date()}"
    assert (profits["perfect"] >= -SLACK).all(), f"the crystal ball lost money on {profits['perfect'].idxmin().date()}"
    seen = pd.DataFrame({TARGET: forecasts[TARGET], **{name: forecasts[column] for name, column in PLAYERS.items()}})
    scores = rp.score_table(seen)[["rmae", "rank_hit_rate"]].join(money_table(profits, BATTERY["capacity"]))
    assert scores.loc["perfect", "capture_rate"] == 1, "the crystal ball does not capture exactly all of its own profit"
    worst = worst_days(profits)
    sentence = headline(scores, days)

    print(scores.round({"profit": 0, "per_mwh_year": 0}).round(3).to_string())
    forecasters = scores.drop(index="perfect")
    by_accuracy = list(forecasters["rmae"].sort_values().index)
    by_money = list(forecasters["profit"].sort_values(ascending=False).index)
    print(f"by accuracy: {', '.join(by_accuracy)} | by money: {', '.join(by_money)}")
    if by_accuracy != by_money:
        print("note: the order by accuracy is not the order by money. That is a finding for the write-up")
    if scores.loc["gbm", "profit"] <= scores.loc["naive", "profit"]:
        print("note: your forecast did not earn more than same-as-last-week. Look into this before Step 8")
    print(worst.round(0).to_string())
    print(sentence)
    plans.to_sql("battery_plans", engine, if_exists="replace", index_label="delivery_time")
    profits.to_sql("battery_profits", engine, if_exists="replace", index_label="delivery_date")
    Path("reports").mkdir(exist_ok=True)
    scores.to_csv("reports/battery_scores.csv", index_label="player")
    worst.to_csv("reports/battery_worst_days.csv")
    Path("reports/battery_headline.txt").write_text(sentence + "\n")
    print(f"saved {len(plans)} rows for {len(days)} days")


if __name__ == "__main__":
    main()