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
        # --- projects ---
        conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            start_date DATE NULL,
            end_date DATE NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # add columns if missing (safe migrations)
        conn.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS landxml_key TEXT;")
        conn.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS top_width_m DOUBLE PRECISION;")

        # --- workers ---
        conn.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # --- assignments / bookings ---
        conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_assignments (
            id SERIAL PRIMARY KEY,
            worker_id INT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
            project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            note TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT assignment_dates CHECK (end_date >= start_date)
        );
        """)

        conn.commit()


# ---------------- Projects API ----------------
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
            (name.strip(), start_date, end_date),
        )
        conn.commit()


def get_project(project_id: int):
    with get_conn() as conn:
        r = conn.execute("""
            SELECT id, name, start_date, end_date, landxml_key, top_width_m
            FROM projects
            WHERE id=%s
        """, (int(project_id),)).fetchone()
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


# ---------------- Workers API (names match your workers_view.py) ----------------
def add_worker(name: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO workers (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
            (name.strip(),),
        )
        conn.commit()


def list_workers():
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name FROM workers ORDER BY name ASC").fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]


def _worker_available(worker_id: int, start_date, end_date) -> bool:
    with get_conn() as conn:
        r = conn.execute("""
            SELECT COUNT(*) FROM worker_assignments
            WHERE worker_id=%s
              AND NOT (end_date < %s OR start_date > %s)
        """, (int(worker_id), start_date, end_date)).fetchone()
    return (r[0] == 0)


def add_assignment(worker_id: int, project_id: int, start_date, end_date, note: str | None = None):
    # prevents double booking
    if not _worker_available(worker_id, start_date, end_date):
        raise ValueError("Töötaja on juba broneeritud selles ajavahemikus.")

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO worker_assignments (worker_id, project_id, start_date, end_date, note)
            VALUES (%s, %s, %s, %s, %s)
        """, (int(worker_id), int(project_id), start_date, end_date, note))
        conn.commit()


def list_assignments():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT wa.id, w.name, p.name, wa.start_date, wa.end_date, wa.note
            FROM worker_assignments wa
            JOIN workers w ON w.id = wa.worker_id
            JOIN projects p ON p.id = wa.project_id
            ORDER BY wa.start_date DESC
        """).fetchall()

    return [
        {
            "id": r[0],
            "worker": r[1],
            "project": r[2],
            "start_date": r[3],
            "end_date": r[4],
            "note": r[5],
        }
        for r in rows
    ]
