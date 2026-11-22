-- Raw inputs, one row per value. Every timestamp is stored in UTC.

CREATE TABLE IF NOT EXISTS prices_utc (
    zone            text             NOT NULL,   -- 'NL'
    ts              timestamptz      NOT NULL,   -- start of the delivery hour
    price_eur_mwh   double precision NOT NULL,
    PRIMARY KEY (zone, ts)
);

CREATE TABLE IF NOT EXISTS load_forecast (
    zone      text             NOT NULL,         -- 'NL' or 'DE_LU'
    ts        timestamptz      NOT NULL,
    load_mw   double precision NOT NULL,
    PRIMARY KEY (zone, ts)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
    point       text             NOT NULL,       -- name of the weather point in your config
    ts          timestamptz      NOT NULL,       -- the hour the forecast is for
    variable    text             NOT NULL,       -- e.g. 'shortwave_radiation'
    lead_days   smallint         NOT NULL,       -- 2 for _previous_day2
    value       double precision,
    PRIMARY KEY (point, ts, variable, lead_days)
);

CREATE TABLE IF NOT EXISTS availability (
    input        text PRIMARY KEY,
    source       text NOT NULL,
    known_when   text NOT NULL
);