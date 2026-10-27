import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
MIGRATIONS = Path(__file__).parent / "migrations"


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    migrate(conn)
    register_vector(conn)
    return conn


def migrate(conn: psycopg.Connection) -> None:
    # Advisory lock: several API tasks may start at once and race on the same migration.
    conn.execute("select pg_advisory_xact_lock(42)")
    conn.execute(
        "create table if not exists schema_migrations"
        " (name text primary key, applied_at timestamptz not null default now())"
    )
    done = {row[0] for row in conn.execute("select name from schema_migrations")}
    for file in sorted(MIGRATIONS.glob("*.sql")):
        if file.name not in done:
            conn.execute(file.read_text())
            conn.execute("insert into schema_migrations (name) values (%s)", (file.name,))
    conn.commit()
