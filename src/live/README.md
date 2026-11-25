# Live forecasts

Every data file in this folder was written by a scheduled job on GitHub (`.github/workflows/daily-forecast.yml`), not by hand.

- `YYYY-MM-DD.csv` is the forecast for that delivery day. It was made on the morning before, before the day-ahead auction closed at 12:00 Amsterdam time. `gbm` is the model's forecast and `naive` is the price of the same hour a week earlier, both in EUR per MWh. `gbm_plan` and `naive_plan` are the battery's planned net output in MW for that hour: positive while selling, negative while buying. `made_at` is the moment the file was made, in UTC.
- `scores.csv` has one line per delivery day, added once that day's real prices were published.
- `summary.txt` is the scoreboard in words.

A file is never changed after it is written. The commit history of this folder shows when each one arrived.