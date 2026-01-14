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
    """
    Create all tables if they don't exist.
    Safe to run on every app start.
    """
    with get_conn() as conn:
        cur = conn.cursor()

        # Projects
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                start_date DATE NULL,
                end_date DATE NULL,

                -- LandXML / geometry
                landxml_key TEXT NULL,
                top_width_m DOUBLE PRECISION NULL,
                planned_length_m DOUBLE PRECISION NULL,
                planned_area_m2 DOUBLE PRECISION NULL,
                planned_volume_m3 DOUBLE PRECISION NULL,

                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        # Workers
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workers (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NULL,
                hourly_eur DOUBLE PRECISION NOT NULL DEFAULT 0,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        # Assignments (worker booking)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                id SERIAL PRIMARY KEY,
                worker_id INTEGER NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                note TEXT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        # Tasks
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'planned', -- planned | in_progress | done
                start_date DATE NULL,
                end_date DATE NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        # Task dependencies (VERY IMPORTANT columns: task_id, dep_task_id)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_deps (
                task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                dep_task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                PRIMARY KEY (task_id, dep_task_id)
            );
            """
        )

        # Reports / cost items
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_items (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                unit TEXT NOT NULL,
                unit_price DOUBLE PRECISION NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        # Production rows (costs based on qty * unit_price)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS productions (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                cost_item_id INTEGER NOT NULL REFERENCES cost_items(id) ON DELETE RESTRICT,
                qty DOUBLE PRECISION NOT NULL DEFAULT 0,
                work_date DATE NOT NULL,
                note TEXT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        # Revenue rows
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS revenues (
                id SERIAL PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                amount DOUBLE PRECISION NOT NULL DEFAULT 0,
                rev_date DATE NOT NULL,
                note TEXT NULL,
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
        return cur.fetchall()


def get_project(project_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM projects WHERE id=%s;", (project_id,))
        return cur.fetchone()


def set_project_top_width(project_id: int, landxml_key: str, top_width_m: float | None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE projects
            SET landxml_key=%s, top_width_m=%s
            WHERE id=%s;
            """,
            (landxml_key, top_width_m, project_id),
        )
        conn.commit()


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


# ---------------------------
# Workers
# ---------------------------

def add_worker(name: str, role: str | None, hourly_eur: float):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO workers(name, role, hourly_eur)
            VALUES (%s, %s, %s);
            """,
            (name, role or None, float(hourly_eur)),
        )
        conn.commit()


def list_workers(active_only: bool = True):
    with get_conn() as conn:
        cur = conn.cursor()
        if active_only:
            cur.execute("SELECT * FROM workers WHERE active=TRUE ORDER BY name;")
        else:
            cur.execute("SELECT * FROM workers ORDER BY name;")
        return cur.fetchall()


def add_assignment(worker_id: int, project_id: int, start: date, end: date, note: str | None):
    if end < start:
        raise Exception("Lõppkuupäev ei tohi olla enne algust.")

    with get_conn() as conn:
        cur = conn.cursor()

        # conflict check: same worker overlapping date range
        cur.execute(
            """
            SELECT COUNT(*) AS c
            FROM assignments
            WHERE worker_id=%s
              AND NOT (end_date < %s OR start_date > %s);
            """,
            (worker_id, start, end),
        )
        c = cur.fetchone()["c"]
        if c and int(c) > 0:
            raise Exception("Töötaja on sellel perioodil juba broneeritud teisele objektile.")

        cur.execute(
            """
            INSERT INTO assignments(worker_id, project_id, start_date, end_date, note)
            VALUES (%s,%s,%s,%s,%s);
            """,
            (worker_id, project_id, start, end, note or None),
        )
        conn.commit()


def list_assignments():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT a.*,
                   w.name AS worker_name,
                   p.name AS project_name
            FROM assignments a
            JOIN workers w ON w.id=a.worker_id
            JOIN projects p ON p.id=a.project_id
            ORDER BY a.start_date DESC, a.id DESC;
            """
        )
        return cur.fetchall()


def list_assignments_by_project(project_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT a.*,
                   w.name AS worker_name
            FROM assignments a
            JOIN workers w ON w.id=a.worker_id
            WHERE a.project_id=%s
            ORDER BY a.start_date ASC;
            """,
            (project_id,),
        )
        return cur.fetchall()


def delete_assignment(assignment_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM assignments WHERE id=%s;", (assignment_id,))
        conn.commit()


# ---------------------------
# Tasks + dependencies
# ---------------------------

def add_task(project_id: int, name: str, start_date: date | None = None, end_date: date | None = None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks(project_id, name, start_date, end_date)
            VALUES (%s,%s,%s,%s);
            """,
            (project_id, name, start_date, end_date),
        )
        conn.commit()


def delete_task(task_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        # remove dependency links too (both directions)
        cur.execute("DELETE FROM task_deps WHERE task_id=%s OR dep_task_id=%s;", (task_id, task_id))
        cur.execute("DELETE FROM tasks WHERE id=%s;", (task_id,))
        conn.commit()


def set_task_deps(task_id: int, dep_ids: list[int]):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM task_deps WHERE task_id=%s;", (task_id,))
        for dep_id in dep_ids:
            cur.execute(
                """
                INSERT INTO task_deps(task_id, dep_task_id)
                VALUES (%s,%s)
                ON CONFLICT DO NOTHING;
                """,
                (task_id, dep_id),
            )
        conn.commit()


def _missing_deps_count(cur, task_id: int) -> int:
    cur.execute(
        """
        SELECT COUNT(*) AS missing
        FROM task_deps d
        JOIN tasks td ON td.id = d.dep_task_id
        WHERE d.task_id = %s
          AND (td.status IS NULL OR td.status <> 'done');
        """,
        (task_id,),
    )
    return int(cur.fetchone()["missing"])


def set_task_status(task_id: int, status: str):
    if status not in ("planned", "in_progress", "done"):
        raise Exception("Vale staatus.")

    with get_conn() as conn:
        cur = conn.cursor()

        missing = _missing_deps_count(cur, task_id)
        if missing > 0 and status in ("in_progress", "done"):
            raise Exception("Ei saa alustada/lõpetada: eeldustööd on tegemata.")

        cur.execute("UPDATE tasks SET status=%s WHERE id=%s;", (status, task_id))
        conn.commit()


def list_tasks(project_id: int):
    """
    Returns tasks with:
      blocked=True if deps missing and task not done
      missing_deps=int
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT t.*
            FROM tasks t
            WHERE t.project_id=%s
            ORDER BY t.id ASC;
            """,
            (project_id,),
        )
        tasks = cur.fetchall()

        out = []
        for t in tasks:
            missing = _missing_deps_count(cur, t["id"])
            status = t["status"] or "planned"
            blocked = (missing > 0) and (status != "done")
            out.append({
                "id": t["id"],
                "project_id": t["project_id"],
                "name": t["name"],
                "status": status,
                "start_date": t["start_date"],
                "end_date": t["end_date"],
                "missing_deps": missing,
                "blocked": blocked,
            })
        return out


# ---------------------------
# Reports: cost + revenue
# ---------------------------

def add_cost_item(name: str, unit: str, unit_price: float):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO cost_items(name, unit, unit_price)
            VALUES (%s,%s,%s);
            """,
            (name, unit, float(unit_price)),
        )
        conn.commit()


def list_cost_items():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM cost_items ORDER BY id DESC;")
        return cur.fetchall()


def add_production(project_id: int, cost_item_id: int, qty: float, work_date: date, note: str | None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO productions(project_id, cost_item_id, qty, work_date, note)
            VALUES (%s,%s,%s,%s,%s);
            """,
            (project_id, cost_item_id, float(qty), work_date, note or None),
        )
        conn.commit()


def add_revenue(project_id: int, amount: float, rev_date: date, note: str | None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO revenues(project_id, amount, rev_date, note)
            VALUES (%s,%s,%s,%s);
            """,
            (project_id, float(amount), rev_date, note or None),
        )
        conn.commit()


def daily_profit_series(project_id: int):
    """
    Returns rows:
      date, revenue, cost, profit
    """
    with get_conn() as conn:
        cur = conn.cursor()

        # costs = sum(qty * unit_price) per day
        cur.execute(
            """
            WITH cost_by_day AS (
              SELECT pr.work_date AS d,
                     SUM(pr.qty * ci.unit_price) AS cost
              FROM productions pr
              JOIN cost_items ci ON ci.id = pr.cost_item_id
              WHERE pr.project_id=%s
              GROUP BY pr.work_date
            ),
            rev_by_day AS (
              SELECT r.rev_date AS d,
                     SUM(r.amount) AS revenue
              FROM revenues r
              WHERE r.project_id=%s
              GROUP BY r.rev_date
            ),
            days AS (
              SELECT d FROM cost_by_day
              UNION
              SELECT d FROM rev_by_day
            )
            SELECT
              days.d AS date,
              COALESCE(rev_by_day.revenue, 0) AS revenue,
              COALESCE(cost_by_day.cost, 0) AS cost,
              COALESCE(rev_by_day.revenue, 0) - COALESCE(cost_by_day.cost, 0) AS profit
            FROM days
            LEFT JOIN cost_by_day ON cost_by_day.d = days.d
            LEFT JOIN rev_by_day ON rev_by_day.d = days.d
            ORDER BY days.d ASC;
            """,
            (project_id, project_id),
        )

        return cur.fetchall()
