import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

@contextmanager
def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing (Render env var).")
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    ddl = """
    CREATE TABLE IF NOT EXISTS projects (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        start_date DATE,
        end_date DATE,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS cost_items (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        unit TEXT NOT NULL,
        unit_price NUMERIC(12,2) NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS production_entries (
        id SERIAL PRIMARY KEY,
        project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        cost_item_id INT NOT NULL REFERENCES cost_items(id),
        qty NUMERIC(14,3) NOT NULL,
        work_date DATE NOT NULL,
        note TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS revenue_entries (
        id SERIAL PRIMARY KEY,
        project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        amount NUMERIC(14,2) NOT NULL,
        rev_date DATE NOT NULL,
        note TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS workers (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT,
        hourly_cost NUMERIC(12,2) NOT NULL DEFAULT 0,
        active BOOLEAN NOT NULL DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS assignments (
        id SERIAL PRIMARY KEY,
        worker_id INT NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
        project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        note TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        start_date DATE,
        end_date DATE,
        status TEXT NOT NULL DEFAULT 'planned',
        completed_at TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS task_deps (
        task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        depends_on_task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        PRIMARY KEY(task_id, depends_on_task_id)
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)

            # --- migrations (safe) ---
            cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS planned_volume_m3 NUMERIC(14,3)")
            cur.execute("ALTER TABLE projects ADD COLUMN IF NOT EXISTS landxml_key TEXT")
            cur.execute("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS sort_order INT NOT NULL DEFAULT 0")

            cur.execute("""
            CREATE TABLE IF NOT EXISTS weekly_plan (
                id SERIAL PRIMARY KEY,
                project_id INT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                task_id INT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                week_start DATE NOT NULL,
                note TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """)
        conn.commit()

# ---------- Projects ----------
def create_project(name: str, start_date=None, end_date=None):
    q = "INSERT INTO projects(name, start_date, end_date) VALUES(%s,%s,%s) RETURNING id"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (name, start_date, end_date))
            pid = cur.fetchone()[0]
        conn.commit()
    return pid

def list_projects():
    q = """
    SELECT id, name, start_date, end_date, status, planned_volume_m3, landxml_key
    FROM projects
    ORDER BY name
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q)
            return [dict(r) for r in cur.fetchall()]

def set_project_landxml(project_id: int, landxml_key: str, planned_volume_m3):
    q = "UPDATE projects SET landxml_key=%s, planned_volume_m3=%s WHERE id=%s"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (landxml_key, planned_volume_m3, project_id))
        conn.commit()

# ---------- Cost items ----------
def list_cost_items():
    q = "SELECT id, name, unit, unit_price FROM cost_items ORDER BY name"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q)
            return [dict(r) for r in cur.fetchall()]

def add_cost_item(name, unit, unit_price):
    q = "INSERT INTO cost_items(name, unit, unit_price) VALUES(%s,%s,%s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (name, unit, unit_price))
        conn.commit()

# ---------- Production / Revenue ----------
def add_production(project_id, cost_item_id, qty, work_date, note=""):
    q = """
    INSERT INTO production_entries(project_id, cost_item_id, qty, work_date, note)
    VALUES(%s,%s,%s,%s,%s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (project_id, cost_item_id, qty, work_date, note))
        conn.commit()

def add_revenue(project_id, amount, rev_date, note=""):
    q = """
    INSERT INTO revenue_entries(project_id, amount, rev_date, note)
    VALUES(%s,%s,%s,%s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (project_id, amount, rev_date, note))
        conn.commit()

# ---------- Workers ----------
def add_worker(name, role="", hourly_cost=0):
    q = "INSERT INTO workers(name, role, hourly_cost) VALUES(%s,%s,%s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (name, role, hourly_cost))
        conn.commit()

def list_workers(active_only=True):
    if active_only:
        q = "SELECT id, name, role, hourly_cost, active FROM workers WHERE active=TRUE ORDER BY name"
    else:
        q = "SELECT id, name, role, hourly_cost, active FROM workers ORDER BY name"
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q)
            return [dict(r) for r in cur.fetchall()]

# ---------- Assignments (double booking check) ----------
def add_assignment(worker_id, project_id, start_date, end_date, note=""):
    if end_date < start_date:
        raise ValueError("End date must be >= start date")

    overlap_q = """
    SELECT 1 FROM assignments
    WHERE worker_id=%s
      AND %s <= end_date
      AND %s >= start_date
    LIMIT 1
    """
    insert_q = """
    INSERT INTO assignments(worker_id, project_id, start_date, end_date, note)
    VALUES(%s,%s,%s,%s,%s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(overlap_q, (worker_id, start_date, end_date))
            if cur.fetchone():
                raise ValueError("Worker is already assigned in this date range.")
            cur.execute(insert_q, (worker_id, project_id, start_date, end_date, note))
        conn.commit()

def list_assignments():
    q = """
    SELECT a.id, w.name AS worker_name, p.name AS project_name, a.start_date, a.end_date, a.note
    FROM assignments a
    JOIN workers w ON w.id=a.worker_id
    JOIN projects p ON p.id=a.project_id
    ORDER BY a.start_date DESC, w.name
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q)
            return [dict(r) for r in cur.fetchall()]

# ---------- Tasks + dependencies ----------
def add_task(project_id, name, start_date=None, end_date=None, sort_order=0):
    q = """
    INSERT INTO tasks(project_id, name, start_date, end_date, sort_order)
    VALUES(%s,%s,%s,%s,%s) RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (project_id, name, start_date, end_date, sort_order))
            tid = cur.fetchone()[0]
        conn.commit()
    return tid

def set_task_deps(task_id, depends_on_ids):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM task_deps WHERE task_id=%s", (task_id,))
            for dep_id in depends_on_ids:
                if dep_id == task_id:
                    continue
                cur.execute(
                    "INSERT INTO task_deps(task_id, depends_on_task_id) VALUES(%s,%s)",
                    (task_id, dep_id)
                )
        conn.commit()

def task_is_blocked(task_id):
    q = """
    SELECT 1
    FROM task_deps d
    JOIN tasks t ON t.id=d.depends_on_task_id
    WHERE d.task_id=%s AND t.status <> 'done'
    LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, (task_id,))
            return cur.fetchone() is not None

def list_tasks(project_id):
    # higher sort_order => later work. We'll show DESC so top is last.
    q = """
    SELECT id, name, start_date, end_date, status, sort_order
    FROM tasks
    WHERE project_id=%s
    ORDER BY sort_order DESC, id DESC
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(q, (project_id,))
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["blocked"] = task_is_blocked(r["id"])
    return rows

def set_task_status(task_id, new_status):
    if new_status in ("in_progress", "done") and task_is_blocked(task_id):
        raise ValueError("Task is blocked by prerequisites.")
    with get_conn() as conn:
        with conn.cursor() as cur:
            if new_status == "done":
                cur.execute("UPDATE tasks SE
