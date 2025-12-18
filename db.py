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
        # ---- projects ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            start_date DATE NULL,
            end_date DATE NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)
        conn.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS landxml_key TEXT;")
        conn.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS top_width_m DOUBLE PRECISION;")

        # ---- workers ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            role TEXT NULL,
            hourly_rate DOUBLE PRECISION NOT NULL DEFAULT 0,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # ---- worker assignments ----
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

        # ---- tasks ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id SERIAL PRIMARY KEY,
            project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            start_date DATE NULL,
            end_date DATE NULL,
            status TEXT NOT NULL DEFAULT 'todo',  -- todo | in_progress | done
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # ---- task dependencies ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS task_deps (
            task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            depends_on_task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            PRIMARY KEY (task_id, depends_on_task_id)
        );
        """)

        conn.commit()


# =========================
# Projects
# =========================
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
            "INSERT INTO projects (name, start_date, end_date) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO NOTHING",
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


# =========================
# Workers
# =========================
def add_worker(name: str, role: str = "", hourly: float = 0.0):
    name = (name or "").strip()
    role = (role or "").strip()
    hourly = float(hourly or 0.0)
    if not name:
        raise ValueError("Sisesta nimi.")

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO workers (name, role, hourly_rate, active)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT (name) DO UPDATE
            SET role=EXCLUDED.role,
                hourly_rate=EXCLUDED.hourly_rate,
                active=TRUE
        """, (name, role, hourly))
        conn.commit()


def list_workers(active_only: bool = True):
    with get_conn() as conn:
        if active_only:
            rows = conn.execute("""
                SELECT id, name, role, hourly_rate, active
                FROM workers
                WHERE active=TRUE
                ORDER BY name ASC
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, name, role, hourly_rate, active
                FROM workers
                ORDER BY name ASC
            """).fetchall()

    return [
        {"id": r[0], "name": r[1], "role": r[2], "hourly_rate": r[3], "active": r[4]}
        for r in rows
    ]


def _worker_available(worker_id: int, start_date, end_date) -> bool:
    with get_conn() as conn:
        r = conn.execute("""
            SELECT COUNT(*)
            FROM worker_assignments
            WHERE worker_id=%s
              AND NOT (end_date < %s OR start_date > %s)
        """, (int(worker_id), start_date, end_date)).fetchone()
    return (r[0] == 0)


def add_assignment(worker_id: int, project_id: int, start_date, end_date, note: str | None = None):
    if end_date < start_date:
        raise ValueError("Lõpp peab olema >= algus.")

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
            SELECT
                wa.id,
                w.name AS worker_name,
                p.name AS project_name,
                wa.start_date,
                wa.end_date,
                wa.note
            FROM worker_assignments wa
            JOIN workers w ON w.id = wa.worker_id
            JOIN projects p ON p.id = wa.project_id
            ORDER BY wa.start_date DESC, wa.id DESC
        """).fetchall()

    return [
        {
            "id": r[0],
            "worker_name": r[1],
            "project_name": r[2],
            "start_date": r[3],
            "end_date": r[4],
            "note": r[5],
        }
        for r in rows
    ]


# =========================
# Tasks + Dependencies
# =========================
def add_task(project_id: int, name: str, start_date=None, end_date=None):
    name = (name or "").strip()
    if not name:
        raise ValueError("Sisesta töö nimetus.")

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO tasks (project_id, name, start_date, end_date, status)
            VALUES (%s, %s, %s, %s, 'todo')
        """, (int(project_id), name, start_date, end_date))
        conn.commit()


def set_task_deps(task_id: int, dep_ids: list[int]):
    dep_ids = [int(x) for x in dep_ids or []]
    task_id = int(task_id)

    with get_conn() as conn:
        # wipe old deps
        conn.execute("DELETE FROM task_deps WHERE task_id=%s", (task_id,))
        # insert new deps
        for d in dep_ids:
            if d == task_id:
                continue
            conn.execute(
                "INSERT INTO task_deps (task_id, depends_on_task_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (task_id, int(d)),
            )
        conn.commit()


def _deps_done(task_id: int) -> bool:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT t.status
            FROM task_deps d
            JOIN tasks t ON t.id = d.depends_on_task_id
            WHERE d.task_id=%s
        """, (int(task_id),)).fetchall()

    # if no deps -> ok
    if not rows:
        return True
    # all must be done
    return all((r[0] == "done") for r in rows)


def set_task_status(task_id: int, status: str):
    status = (status or "").strip()
    if status not in ("todo", "in_progress", "done"):
        raise ValueError("Invalid status.")

    # block if deps not done
    if status in ("in_progress", "done") and not _deps_done(task_id):
        raise ValueError("Ei saa alustada/lõpetada — eeldustööd on tegemata (peavad olema DONE).")

    with get_conn() as conn:
        conn.execute("UPDATE tasks SET status=%s WHERE id=%s", (status, int(task_id)))
        conn.commit()


def list_tasks(project_id: int):
    project_id = int(project_id)

    with get_conn() as conn:
        tasks = conn.execute("""
            SELECT id, name, start_date, end_date, status
            FROM tasks
            WHERE project_id=%s
            ORDER BY created_at ASC, id ASC
        """, (project_id,)).fetchall()

        # deps per task
        deps = conn.execute("""
            SELECT task_id, depends_on_task_id
            FROM task_deps
            WHERE task_id IN (SELECT id FROM tasks WHERE project_id=%s)
        """, (project_id,)).fetchall()

        deps_map = {}
        for t_id, dep_id in deps:
            deps_map.setdefault(t_id, []).append(dep_id)

        # status for dep tasks (for blocked check)
        status_map = {}
        if tasks:
            ids = tuple([t[0] for t in tasks])
            if len(ids) == 1:
                dep_rows = conn.execute("SELECT id, status FROM tasks WHERE id=%s", (ids[0],)).fetchall()
            else:
                dep_rows = conn.execute("SELECT id, status FROM tasks WHERE id IN %s", (ids,)).fetchall()
            status_map = {r[0]: r[1] for r in dep_rows}

    out = []
    for (tid, name, start, end, status) in tasks:
        dep_ids = deps_map.get(tid, [])
        blocked = False
        if dep_ids:
            blocked = any(status_map.get(did) != "done" for did in dep_ids)

        out.append({
            "id": tid,
            "name": name,
            "start_date": start,
            "end_date": end,
            "status": status,
            "deps": dep_ids,
            "blocked": blocked,
        })
    return out
