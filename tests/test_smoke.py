import pytest


def test_libraries_import():
    import entsoe, lightgbm, pandas, scipy, sklearn, statsmodels  # noqa: F401


def test_database_answers():
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError
    from src.db import get_engine

    try:
        conn = get_engine().connect()
    except (KeyError, OperationalError) as error:
        pytest.skip(f"database not reachable: {error}")
    with conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1