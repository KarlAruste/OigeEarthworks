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
        # Create table if it does not exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            role TEXT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # ✅ MIGRATIONS for existing DBs (Render / production)
        # Ensure required columns exist even if table was created earlier
        conn.execute("ALTER TABLE workers ADD COLUMN IF NOT EXISTS role TEXT;")
        conn.execute("ALTER TABLE workers ADD COLUMN IF NOT EXISTS hourly_rate DOUBLE PRECISION NOT NULL DEFAULT 0;")
        conn.execute("ALTER TABLE workers ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;")
        conn.execute("ALTER TABLE workers ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();")

        # Optional: if you previously used another column name (hourly), copy it over once.
        # Safe: does nothing if hourly column doesn't exist.
        conn.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='workers' AND column_name='hourly'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='workers' AND column_name='hourly_rate'
            ) THEN
                ALTER TABLE workers ADD COLUMN hourly_rate DOUBLE PRECISION NOT NULL DEFAULT 0;
                EXECUTE 'UPDATE workers SET hourly_rate = COALESCE(hourly, 0)';
            END IF;
        END $$;
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

        # ---- cost items ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cost_items (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            unit TEXT NOT NULL,
            unit_price DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # ---- production (costs by work done) ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS production (
            id SERIAL PRIMARY KEY,
            project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            cost_item_id INT NOT NULL REFERENCES cost_items(id) ON DELETE RESTRICT,
            qty DOUBLE PRECISION NOT NULL DEFAULT 0,
            work_date DATE NOT NULL,
            note TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """)

        # ---- revenue ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS revenue (
            id SERIAL PRIMARY KEY,
            project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            amount DOUBLE PRECISION NOT NULL DEFAULT 0,
            rev_date DATE NOT NULL,
            note TEXT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
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
        conn.execute("DELETE FROM task_deps WHERE task_id=%s", (task_id,))
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

    if not rows:
        return True
    return all((r[0] == "done") for r in rows)


def set_task_status(task_id: int, status: str):
    status = (status or "").strip()
    if status not in ("todo", "in_progress", "done"):
        raise ValueError("Invalid status.")

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

        deps = conn.execute("""
            SELECT task_id, depends_on_task_id
            FROM task_deps
            WHERE task_id IN (SELECT id FROM tasks WHERE project_id=%s)
        """, (project_id,)).fetchall()

        deps_map = {}
        for t_id, dep_id in deps:
            deps_map.setdefault(t_id, []).append(dep_id)

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


# =========================
# Reports / Cost & Revenue
# =========================
def list_cost_items():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, name, unit, unit_price
            FROM cost_items
            ORDER BY name ASC
        """).fetchall()
    return [{"id": r[0], "name": r[1], "unit": r[2], "unit_price": r[3]} for r in rows]


def add_cost_item(name: str, unit: str, unit_price: float):
    name = (name or "").strip()
    unit = (unit or "").strip()
    unit_price = float(unit_price or 0.0)
    if not name or not unit:
        raise ValueError("Täida nimetus ja ühik.")

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO cost_items (name, unit, unit_price)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET unit=EXCLUDED.unit,
                unit_price=EXCLUDED.unit_price
        """, (name, unit, unit_price))
        conn.commit()


def add_production(project_id: int, cost_item_id: int, qty: float, work_date, note: str | None = None):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO production (project_id, cost_item_id, qty, work_date, note)
            VALUES (%s, %s, %s, %s, %s)
        """, (int(project_id), int(cost_item_id), float(qty or 0.0), work_date, note))
        conn.commit()


def add_revenue(project_id: int, amount: float, rev_date, note: str | None = None):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO revenue (project_id, amount, rev_date, note)
            VALUES (%s, %s, %s, %s)
        """, (int(project_id), float(amount or 0.0), rev_date, note))
        conn.commit()


def daily_profit_series(project_id: int):
    project_id = int(project_id)

    with get_conn() as conn:
        rows = conn.execute("""
            WITH d AS (
                SELECT work_date::date AS dt
                FROM production
                WHERE project_id=%s
                UNION
                SELECT rev_date::date AS dt
                FROM revenue
                WHERE project_id=%s
            ),
            rev AS (
                SELECT rev_date::date AS dt, SUM(amount) AS revenue
                FROM revenue
                WHERE project_id=%s
                GROUP BY rev_date::date
            ),
            cost AS (
                SELECT p.work_date::date AS dt, SUM(p.qty * ci.unit_price) AS cost
                FROM production p
                JOIN cost_items ci ON ci.id = p.cost_item_id
                WHERE p.project_id=%s
                GROUP BY p.work_date::date
            )
            SELECT
                d.dt AS date,
                COALESCE(rev.revenue, 0) AS revenue,
                COALESCE(cost.cost, 0) AS cost,
                COALESCE(rev.revenue, 0) - COALESCE(cost.cost, 0) AS profit
            FROM d
            LEFT JOIN rev ON rev.dt = d.dt
            LEFT JOIN cost ON cost.dt = d.dt
            ORDER BY d.dt ASC
        """, (project_id, project_id, project_id, project_id)).fetchall()

    return [{"date": r[0], "revenue": float(r[1]), "cost": float(r[2]), "profit": float(r[3])} for r in rows]
