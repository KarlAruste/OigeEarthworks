import streamlit as st

from views.projects_view import render_projects_view
from views.workers_view import render_workers_view
from views.tasks_view import render_tasks_view
from views.reports_view import render_reports_view
from views.machines_view import render_machines_view

st.set_page_config(page_title="Earthworks", layout="wide")

st.markdown("""
<style>
.stApp { background-color:#0b0f17; color:#f3f4f6; }
section[data-testid="stSidebar"] { background-color:#0f1522; border-right:1px solid #1f2937; }
section[data-testid="stSidebar"] * { color:#f3f4f6 !important; }

h1,h2,h3,h4 { color:#ffffff; }
p, span, label, div { color:#e5e7eb; }

.block { background:#101827; border:1px solid #223047; border-radius:14px; padding:16px; }

.small, .stCaption, [data-testid="stCaptionContainer"] { color:#cbd5e1 !important; }

.stButton>button {
  background-color:#ff8a00; color:#000; border-radius:10px; border:none;
  font-weight:700;
}
.stButton>button:hover { background-color:#ffa733; }

div[role="radiogroup"] > label > div:first-child { display:none; }
div[role="radiogroup"] label { padding:10px 12px; border-radius:10px; margin-bottom:6px; }
div[role="radiogroup"] label:hover { background-color:#111c2e; }
div[role="radiogroup"] label:has(input:checked) { background-color:#ff8a00; color:#000; }

button[data-baseweb="tab"] { color:#e5e7eb !important; }
button[data-baseweb="tab"][aria-selected="true"] {
  color:#000 !important;
  background:#ff8a00 !important;
  border-radius:10px !important;
}
</style>
""", unsafe_allow_html=True)

if "active_project_id" not in st.session_state:
    st.session_state["active_project_id"] = None

st.sidebar.markdown("## 🏗 Earthworks")
st.sidebar.caption("Kaevetööd • Ressursid • Aruanded")
st.sidebar.divider()

page = st.sidebar.radio("Menüü", ["Projects", "Workers", "Tasks", "Machines", "Reports"])

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
