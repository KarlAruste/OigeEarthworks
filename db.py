import os
from contextlib import contextmanager
from datetime import date
import psycopg2
from psycopg2.extras import RealDictCursor


# -----------------------------
# Connection
# -----------------------------
def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing (Render env var).")
    return url


@contextmanager
def get_conn():
    conn = psycopg2.connect(_db_url(), sslmode="require")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """
    Creates required tables if they don't exist.
    Call this once at app startup.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Projects
            cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                start_date DATE NULL,
                end_date DATE NULL,

                -- LandXML / geometry fields (optional)
                landxml_key TEXT NULL,
                top_width_m DOUBLE PRECISION NULL,
                planned_length_m DOUBLE PRECISION NULL,
                planned_area_m2 DOUBLE PRECISION NULL,
                planned_volume_m3 DOUBLE PRECISION NULL
            );
            """)

            # Workers
            cur.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NULL,
                hourly DOUBLE PRECISION NOT NULL DEFAULT 0,
                active BOOLEAN NOT NULL DEFAULT TRUE
            );
            """)

            # Worker assignments to projects
            cur.execute("""
            CREATE TABLE IF NOT EXISTS assignments (
                id SERIAL PRIMARY KEY,
                worker_id INT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
                project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                note TEXT NULL
            );
            """)

            # Tasks per project
            cur.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'todo'  -- todo / in_progress / done
            );
            """)

            # Task dependencies
            cur.execute("""
            CREATE TABLE IF NOT EXISTS task_deps (
                task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                dep_task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                PRIMARY KEY (task_id, dep_task_id)
            );
            """)

            # Cost items (global price list)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cost_items (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                unit TEXT NOT NULL,
                unit_price DOUBLE PRECISION NOT NULL DEFAULT 0
            );
            """)

            # Production / cost (per project)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS production (
                id SERIAL PRIMARY KEY,
                project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                cost_item_id INT NOT NULL REFERENCES cost_items(id) ON DELETE RESTRICT,
                qty DOUBLE PRECISION NOT NULL DEFAULT 0,
                work_date DATE NOT NULL,
                note TEXT NULL
            );
            """)

            # Revenue (per project)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS revenue (
                id SERIAL PRIMARY KEY,
                project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                amount DOUBLE PRECISION NOT NULL DEFAULT 0,
                rev_date DATE NOT NULL,
                note TEXT NULL
            );
            """)


# -----------------------------
# Projects
# -----------------------------
def list_projects():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, start_date, end_date,
                       landxml_key, top_width_m,
                       planned_length_m, planned_area_m2, planned_volume_m3
                FROM projects
                ORDER BY id DESC
            """)
            return cur.fetchall()


def create_project(name: str, start_date=None, end_date=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO projects(name, start_date, end_date) VALUES(%s, %s, %s)",
                (name, start_date, end_date),
            )


def get_project(project_id: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, start_date, end_date,
                       landxml_key, top_width_m,
                       planned_length_m, planned_area_m2, planned_volume_m3
                FROM projects
                WHERE id=%s
            """, (project_id,))
            return cur.fetchone()


def set_project_landxml(project_id: int, landxml_key: str, planned_volume_m3=None, planned_length_m=None, planned_area_m2=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE projects
                SET landxml_key=%s,
                    planned_volume_m3=%s,
                    planned_length_m=%s,
                    planned_area_m2=%s
                WHERE id=%s
            """, (landxml_key, planned_volume_m3, planned_length_m, planned_area_m2, project_id))


def set_project_top_width(project_id: int, landxml_key: str, top_width_m=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE projects
                SET landxml_key=%s,
                    top_width_m=%s
                WHERE id=%s
            """, (landxml_key, top_width_m, project_id))


# -----------------------------
# Workers
# -----------------------------
def add_worker(name: str, role: str, hourly: float):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO workers(name, role, hourly, active) VALUES(%s, %s, %s, TRUE)",
                (name, role or None, float(hourly)),
            )


def list_workers(active_only=True):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if active_only:
                cur.execute("SELECT id, name, role, hourly, active FROM workers WHERE active=TRUE ORDER BY name")
            else:
                cur.execute("SELECT id, name, role, hourly, active FROM workers ORDER BY name")
            return cur.fetchall()


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end < b_start or b_end < a_start)


def add_assignment(worker_id: int, project_id: int, start_date: date, end_date: date, note: str = ""):
    if end_date < start_date:
        raise Exception("Lõppkuupäev peab olema >= alguskuupäev.")

    # Disallow overlapping assignments for same worker (any project)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, project_id, start_date, end_date
                FROM assignments
                WHERE worker_id=%s
            """, (worker_id,))
            rows = cur.fetchall()

            for r in rows:
                if _overlaps(start_date, end_date, r["start_date"], r["end_date"]):
                    raise Exception("Töötaja on juba broneeritud sellel ajavahemikul (topeltbroneering keelatud).")

        with conn.cursor() as cur2:
            cur2.execute("""
                INSERT INTO assignments(worker_id, project_id, start_date, end_date, note)
                VALUES(%s, %s, %s, %s, %s)
            """, (worker_id, project_id, start_date, end_date, note or None))


def list_assignments():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.id, a.worker_id, w.name AS worker_name,
                       a.project_id, p.name AS project_name,
                       a.start_date, a.end_date, a.note
                FROM assignments a
                JOIN workers w ON w.id=a.worker_id
                JOIN projects p ON p.id=a.project_id
                ORDER BY a.start_date DESC, a.id DESC
            """)
            return cur.fetchall()


def list_assignments_by_project(project_id: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.id, a.worker_id, w.name AS worker_name,
                       a.project_id, a.start_date, a.end_date, a.note
                FROM assignments a
                JOIN workers w ON w.id=a.worker_id
                WHERE a.project_id=%s
                ORDER BY a.start_date DESC, a.id DESC
            """, (project_id,))
            return cur.fetchall()


# -----------------------------
# Tasks + Dependencies
# -----------------------------
def add_task(project_id: int, name: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tasks(project_id, name, status) VALUES(%s, %s, 'todo')", (project_id, name))


def set_task_deps(task_id: int, dep_ids):
    dep_ids = dep_ids or []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM task_deps WHERE task_id=%s", (task_id,))
            for dep in dep_ids:
                cur.execute("INSERT INTO task_deps(task_id, dep_task_id) VALUES(%s, %s)", (task_id, dep))


def _deps_done(conn, task_id: int) -> bool:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT d.dep_task_id, t.status
            FROM task_deps d
            JOIN tasks t ON t.id=d.dep_task_id
            WHERE d.task_id=%s
        """, (task_id,))
        rows = cur.fetchall()
        return all(r["status"] == "done" for r in rows)


def set_task_status(task_id: int, status: str):
    if status not in ("todo", "in_progress", "done"):
        raise Exception("Vigane status.")

    with get_conn() as conn:
        if status in ("in_progress", "done"):
            if not _deps_done(conn, task_id):
                raise Exception("Ei saa alustada/lõpetada: eeldustööd pole DONE.")

        with conn.cursor() as cur:
            cur.execute("UPDATE tasks SET status=%s WHERE id=%s", (status, task_id))


def list_tasks(project_id: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, project_id, name, status
                FROM tasks
                WHERE project_id=%s
                ORDER BY id ASC
            """, (project_id,))
            tasks = cur.fetchall()

        # compute "blocked" for each task
        with conn.cursor(cursor_factory=RealDictCursor) as cur2:
            for t in tasks:
                cur2.execute("""
                    SELECT COUNT(*) AS missing
                    FROM task_deps d
                    JOIN tasks dep ON dep.id=d.dep_task_id
                    WHERE d.task_id=%s AND dep.status <> 'done'
                """, (t["id"],))
                missing = int(cur2.fetchone()["missing"])
                t["blocked"] = (missing > 0)
        return tasks


# -----------------------------
# Cost items (global)
# -----------------------------
def add_cost_item(name: str, unit: str, unit_price: float):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cost_items(name, unit, unit_price) VALUES(%s, %s, %s)",
                (name, unit, float(unit_price)),
            )


def list_cost_items():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, name, unit, unit_price FROM cost_items ORDER BY id DESC")
            return cur.fetchall()


# -----------------------------
# Production (cost) + Revenue
# -----------------------------
def add_production(project_id: int, cost_item_id: int, qty: float, work_date: date, note: str = ""):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO production(project_id, cost_item_id, qty, work_date, note)
                VALUES(%s, %s, %s, %s, %s)
            """, (project_id, cost_item_id, float(qty), work_date, note or None))


def add_revenue(project_id: int, amount: float, rev_date: date, note: str = ""):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO revenue(project_id, amount, rev_date, note)
                VALUES(%s, %s, %s, %s)
            """, (project_id, float(amount), rev_date, note or None))


def daily_profit_series(project_id: int):
    """
    Returns rows like:
    [{"date": <date>, "revenue": <float>, "cost": <float>, "profit": <float>}]
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # cost by day
            cur.execute("""
                SELECT pr.work_date AS day,
                       COALESCE(SUM(pr.qty * ci.unit_price), 0) AS cost
                FROM production pr
                JOIN cost_items ci ON ci.id=pr.cost_item_id
                WHERE pr.project_id=%s
                GROUP BY pr.work_date
            """, (project_id,))
            costs = {r["day"]: float(r["cost"]) for r in cur.fetchall()}

            # revenue by day
            cur.execute("""
                SELECT r.rev_date AS day,
                       COALESCE(SUM(r.amount), 0) AS revenue
                FROM revenue r
                WHERE r.project_id=%s
                GROUP BY r.rev_date
            """, (project_id,))
            revs = {r["day"]: float(r["revenue"]) for r in cur.fetchall()}

    all_days = sorted(set(costs.keys()) | set(revs.keys()))
    out = []
    for d in all_days:
        revenue = revs.get(d, 0.0)
        cost = costs.get(d, 0.0)
        out.append({
            "date": d,
            "revenue": revenue,
            "cost": cost,
            "profit": revenue - cost
        })
    return out
