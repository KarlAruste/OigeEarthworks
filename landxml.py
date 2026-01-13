import streamlit as st

from views.projects_view import render_projects_view
from views.workers_view import render_workers_view
from views.tasks_view import render_tasks_view
from views.reports_view import render_reports_view
from views.machines_view import render_machines_view

st.set_page_config(page_title="Earthworks", layout="wide")

# --- Theme CSS ---
st.markdown(
    """
<style>
.stApp { background-color:#0f1117; color:#e5e7eb; }
section[data-testid="stSidebar"] { background-color:#141821; border-right:1px solid #1f2937; }
section[data-testid="stSidebar"] * { color:#e5e7eb; }
div[role="radiogroup"] > label > div:first-child { display:none; }
div[role="radiogroup"] label { padding:10px 12px; border-radius:10px; margin-bottom:6px; }
div[role="radiogroup"] label:hover { background-color:#1b1f2a; }
div[role="radiogroup"] label:has(input:checked) { background-color:#ff8a00; color:#000; }
h1,h2,h3 { color:#f9fafb; }
.block { background:#1b1f2a; border:1px solid #243042; border-radius:14px; padding:16px; }
.small { color:#9ca3af; font-size:13px; }
.stButton>button { background-color:#ff8a00; color:#000; border-radius:10px; border:none; }
.stButton>button:hover { background-color:#ffa733; }
</style>
""",
    unsafe_allow_html=True,
)

# ---- Session state defaults ----
st.session_state.setdefault("active_project_id", None)

# ---- Sidebar ----
st.sidebar.markdown("## 🏗 Earthworks")
st.sidebar.caption("Kaevetööd • Ressursid • Aruanded")
st.sidebar.divider()

page = st.sidebar.radio(
    "Menüü",
    ["Projects", "Workers", "Tasks", "Machines", "Reports"],
    index=0,
)

# ---- Render (ära kasuta st.write(render_...()) ----
try:
    if page == "Projects":
        render_projects_view()
    elif page == "Workers":
        render_workers_view()
    elif page == "Tasks":
        render_tasks_view()
    elif page == "Machines":
        render_machines_view()
    elif page == "Reports":
        render_reports_view()
except Exception as e:
    st.error("View crashis. Vaata errorit allpool.")
    st.exception(e)
