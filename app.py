import streamlit as st
from db import init_db

from views.projects_page import render_projects_page
from views.workers_page import render_workers_page
from views.tasks_page import render_tasks_page
from views.reports_page import render_reports_page

st.set_page_config(page_title="Earthworks App", layout="wide")

# Theme
st.markdown("""
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
""", unsafe_allow_html=True)

# init DB tables (+ migrations)
init_db()

st.sidebar.markdown("## 🏗 Earthworks")
st.sidebar.caption("Kaevetööd • Ressursid • Aruanded")
st.sidebar.divider()

page = st.sidebar.radio("", ["📁 Projects", "👷 Workers", "🗓 Tasks", "📊 Reports"])

if page == "📁 Projects":
    render_projects_page()
elif page == "👷 Workers":
    render_workers_page()
elif page == "🗓 Tasks":
    render_tasks_page()
elif page == "📊 Reports":
    render_reports_page()
