# db.py
import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import date

DATABASE_URL = os.getenv("DATABASE_URL")

# ---------- connection helpers ----------

@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # projects
        cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            start_date DATE,
            end_date DATE,
            landxml_key TEXT,
            top_width_m DOUBLE PRECISION
        );
        """)

        # workers
        cur.execute("""
        CREATE TABLE IF NOT EXISTS workers (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT,
            hourly_rate DOUBLE PRECISION,
            active BOOLEAN DEFAULT TRUE
        );
        """)

        # assignments (worker ↔ project)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id SERIAL PRIMARY KEY,
            worker_id INTEGER REFERENCES workers(id) ON DELETE CASCADE,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            note TEXT
        );
        """)

        # tasks
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            start_date DATE,
            end_date DATE,
            status TEXT DEFAULT 'planned'
        );
        """)

        # task dependencies
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_deps (
            task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
            dep_task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
            PRIMARY KEY (task_id, dep_task_id)
        );
        """)

        # cost items
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cost_items (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            unit TEXT NOT NULL,
            unit_price DOUBLE PRECISION NOT NULL
        );
        """)

        # production (costs)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS production (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            cost_item_id INTEGER REFERENCES cost_items(id),
            qty DOUBLE PRECISION,
            work_date DATE,
            note TEXT
        );
        """)

        # revenue
        cur.execute("""
        CREATE TABLE IF NOT EXISTS revenue (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
            amount DOUBLE PRECISION,
            rev_date DATE,
            note TEXT
        );
        """)


# ---------- projects ----------

def list_projects():
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM projects ORDER BY name;")
        return cur.fetchall()


def create_project(name, start_date=None, end_date=None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO projects (name, start_date, end_date) VALUES (%s,%s,%s)",
            (name, start_date, end_date),
        )


def get_project(project_id):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
        return cur.fetchone()


def set_project_top_width(project_id, landxml_key, top_width):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE projects SET landxml_key=%s, top_width_m=%s WHERE id=%s",
            (landxml_key, top_width, project_id),
        )


# ---------- workers ----------

def add_worker(name, role, hourly):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO workers (name, role, hourly_rate) VALUES (%s,%s,%s)",
            (name, role, hourly),
        )


def list_workers(active_only=True):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if active_only:
            cur.execute("SELECT * FROM workers WHERE active=TRUE ORDER BY name;")
        else:
            cur.execute("SELECT * FROM workers ORDER BY name;")
        return cur.fetchall()


# ---------- assignments ----------

def add_assignment(worker_id, project_id, start, end, note):
    with get_conn() as conn:
        cur = conn.cursor()
        # prevent double booking
        cur.execute("""
        SELECT 1 FROM assignments
        WHERE worker_id=%s
          AND NOT (end_date < %s OR start_date > %s)
        """, (worker_id, start, end))
        if cur.fetchone():
            raise Exception("Töötaja on sel perioodil juba broneeritud.")

        cur.execute("""
        INSERT INTO assignments (worker_id, project_id, start_date, end_date, note)
        VALUES (%s,%s,%s,%s,%s)
        """, (worker_id, project_id, start, end, note))


def list_assignments():
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
        SELECT a.*, w.name AS worker_name, p.name AS project_name
        FROM assignments a
        JOIN workers w ON w.id=a.worker_id
        JOIN projects p ON p.id=a.project_id
        ORDER BY a.start_date DESC
        """)
        return cur.fetchall()


# ---------- tasks ----------

def add_task(project_id, name, start=None, end=None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tasks (project_id, name, start_date, end_date) VALUES (%s,%s,%s,%s)",
            (project_id, name, start, end),
        )


def delete_task(task_id):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM tasks WHERE id=%s", (task_id,))


def list_tasks(project_id):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
        SELECT t.*,
          EXISTS (
            SELECT 1 FROM task_deps d
            JOIN tasks td ON td.id=d.dep_task_id
            WHERE d.task_id=t.id AND td.status!='done'
          ) AS blocked
        FROM tasks t
        WHERE t.project_id=%s
        ORDER BY t.id
        """, (project_id,))
        return cur.fetchall()


def set_task_deps(task_id, dep_ids):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM task_deps WHERE task_id=%s", (task_id,))
        for d in dep_ids:
            cur.execute(
                "INSERT INTO task_deps (task_id, dep_task_id) VALUES (%s,%s)",
                (task_id, d),
            )


def set_task_status(task_id, status):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE tasks SET status=%s WHERE id=%s",
            (status, task_id),
        )


# ---------- cost & reports ----------

def list_cost_items():
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM cost_items ORDER BY name")
        return cur.fetchall()


def add_cost_item(name, unit, price):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO cost_items (name, unit, unit_price) VALUES (%s,%s,%s)",
            (name, unit, price),
        )


def add_production(project_id, cost_item_id, qty, work_date, note):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO production (project_id, cost_item_id, qty, work_date, note)
        VALUES (%s,%s,%s,%s,%s)
        """, (project_id, cost_item_id, qty, work_date, note))


def add_revenue(project_id, amount, rev_date, note):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO revenue (project_id, amount, rev_date, note)
        VALUES (%s,%s,%s,%s)
        """, (project_id, amount, rev_date, note))


def daily_profit_series(project_id):
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
        SELECT d::date AS date,
               COALESCE(SUM(r.amount),0) AS revenue,
               COALESCE(SUM(p.qty*c.unit_price),0) AS cost,
               COALESCE(SUM(r.amount),0) - COALESCE(SUM(p.qty*c.unit_price),0) AS profit
        FROM generate_series(
            (SELECT MIN(work_date) FROM production WHERE project_id=%s),
            CURRENT_DATE,
            INTERVAL '1 day'
        ) d
        LEFT JOIN production p ON p.work_date=d AND p.project_id=%s
        LEFT JOIN cost_items c ON c.id=p.cost_item_id
        LEFT JOIN revenue r ON r.rev_date=d AND r.project_id=%s
        GROUP BY d
        ORDER BY d
        """, (project_id, project_id, project_id))
        return cur.fetchall()
