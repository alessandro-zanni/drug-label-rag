import pytest

from api import db


@pytest.fixture
def conn():
    """Real Postgres connection; everything a test writes is rolled back."""
    with db.connect() as c:
        yield c
        c.rollback()
