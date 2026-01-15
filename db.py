# db.py
import os
from contextlib import contextmanager
from datetime import date
import psycopg2
from psycopg2.extras import RealDictCursor


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing (Render env var).")
    return url


@contextmanager
def get_conn():
    conn = psycopg2.connect(_db_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                start_date DATE NULL,
                end_date DATE NULL,

                landxml_key TEXT NULL,
                top_width_m DOUBLE PRECISION NULL,
                planned_length_m DOUBLE PRECISION NULL,
                planned_area_m2 DOUBLE PRECISION NULL,
                planned_volume_m3 DOUBLE PRECISION NULL,

                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        conn.commit()


# ---------------------------
# Projects
# ---------------------------

def create_project(name: str, start_date: date | None = None, end_date: date | None = None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO projects(name, start_date, end_date)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO NOTHING;
            """,
            (name, start_date, end_date),
        )
        conn.commit()


def list_projects():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM projects ORDER BY id DESC;")
        return cur.fetchall()  # list[dict]


def get_project(project_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM projects WHERE id=%s;", (project_id,))
        return cur.fetchone()


def set_project_landxml(project_id: int, landxml_key: str,
                        planned_volume_m3: float | None,
                        planned_length_m: float | None,
                        planned_area_m2: float | None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE projects
            SET landxml_key=%s,
                planned_volume_m3=%s,
                planned_length_m=%s,
                planned_area_m2=%s
            WHERE id=%s;
            """,
            (landxml_key, planned_volume_m3, planned_length_m, planned_area_m2, project_id),
        )
        conn.commit()
