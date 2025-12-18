import os
from contextlib import contextmanager
import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


@contextmanager
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing (Render env var).")
    conn = psycopg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            start_date DATE NULL,
            end_date DATE NULL,
            landxml_key TEXT NULL,
            top_width_m DOUBLE PRECISION NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_bookings (
            id SERIAL PRIMARY KEY,
            worker_id INT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
            project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            note TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT booking_dates CHECK (end_date >= start_date)
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id SERIAL PRIMARY KEY,
            project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            sort_order INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS task_deps (
            id SERIAL PRIMARY KEY,
            task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            depends_on_task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            UNIQUE(task_id, depends_on_task_id)
        );
        """)

        conn.commit()


# -------- Projects --------
def list_projects():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, name, start_date, end_date, landxml_key, top_width_m
            FROM projects
            ORDER BY created_at DESC
        """).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "start_date": r[2],
            "end_date": r[3],
            "landxml_key": r[4],
            "top_width_m": r[5],
        }
        for r in rows
    ]


def create_project(name: str, start_date=None, end_date=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO projects (name, start_date, end_date) VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
            (name, start_date, end_date),
        )
        conn.commit()


def get_project(project_id: int):
    with get_conn() as conn:
        r = conn.execute(
            "SELECT id, name, start_date, end_date, landxml_key, top_width_m FROM projects WHERE id=%s",
            (int(project_id),),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r[0],
        "name": r[1],
        "start_date": r[2],
        "end_date": r[3],
        "landxml_key": r[4],
        "top_width_m": r[5],
    }


def set_project_top_width(project_id: int, landxml_key: str, top_width_m: float | None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE projects SET landxml_key=%s, top_width_m=%s WHERE id=%s",
            (landxml_key, top_width_m, int(project_id)),
        )
        conn.commit()


# -------- Workers (basic) --------
def list_workers():
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name FROM workers ORDER BY name ASC").fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]


def create_worker(name: str):
    with get_conn() as conn:
        conn.execute("INSERT INTO workers (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (name.strip(),))
        conn.commit()


def list_bookings():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT wb.id, w.name, p.name, wb.start_date, wb.end_date, wb.note
            FROM worker_bookings wb
            JOIN workers w ON w.id = wb.worker_id
            JOIN projects p ON p.id = wb.project_id
            ORDER BY wb.start_date DESC
        """).fetchall()
    return [
        {"id": r[0], "worker": r[1], "project": r[2], "start": r[3], "end": r[4], "note": r[5]}
        for r in rows
    ]


def is_worker_available(worker_id: int, start_date, end_date) -> bool:
    with get_conn() as conn:
        r = conn.execute("""
            SELECT COUNT(*) FROM worker_bookings
            WHERE worker_id=%s
              AND NOT (end_date < %s OR start_date > %s)
        """, (int(worker_id), start_date, end_date)).fetchone()
    return (r[0] == 0)


def create_booking(worker_id: int, project_id: int, start_date, end_date, note=None):
    if not is_worker_available(worker_id, start_date, end_date):
        raise ValueError("Worker is already booked in this time range.")
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO worker_bookings (worker_id, project_id, start_date, end_date, note)
            VALUES (%s, %s, %s, %s, %s)
        """, (int(worker_id), int(project_id), start_date, end_date, note))
        conn.commit()
