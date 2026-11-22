import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()  # reads .env from the project folder


def get_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
    )
    return create_engine(url)


def upsert(frame, table, key_columns, engine):
    """Insert rows. A row whose key already exists is overwritten, so re-running a pull is safe."""
    columns = list(frame.columns)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in key_columns)
    sql = text(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)}) "
        f"ON CONFLICT ({', '.join(key_columns)}) DO UPDATE SET {updates}"
    )
    with engine.begin() as conn:
        conn.execute(sql, frame.to_dict("records"))