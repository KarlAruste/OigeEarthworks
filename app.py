import streamlit as st

from db import init_db
from views.projects_view import render_projects_view
from views.workers_view import render_workers_view
from views.tasks_view import render_tasks_view
from views.reports_view import render_reports_view

st.set_page_config(page_title="Earthworks", layout="wide")

# theme (kui sul on eraldi theme fail, võid siit välja võtta)
st.markdown("""
<style>
.stApp { background-color:#0f1117; color:#e5e7eb; }
section[data-testid="stSidebar"] { background-color:#141821; border-right:1px solid #1f2937; }
section[data-testid="stSidebar"] * { color:#e5e7eb; }
h1,h2,h3 { color:#f9fafb; }
.block { background:#1b1f2a; border:1px solid #243042; border-radius:14px; padding:16px; }
.stButton>button { background-color:#ff8a00; color:#000; border-radius:10px; border:none; }
.stButton>button:hover { background-color:#ffa733; }
</style>
""", unsafe_allow_html=True)

# init DB once
init_db()

st.sidebar.markdown("## 🏗 Earthworks")
page = st.sidebar.radio(
    "Menüü",
    ["📁 Projects", "👷 Workers", "🗓 Tasks", "📊 Reports"],
)

if page == "📁 Projects":
    render_projects_view()
elif page == "👷 Workers":
    render_workers_view()
elif page == "🗓 Tasks":
    render_tasks_view()
elif page == "📊 Reports":
    render_reports_view()
