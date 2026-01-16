import streamlit as st

from db import init_db

from views.projects_view import render_projects_view
from views.workers_view import render_workers_view
from views.tasks_view import render_tasks_view
from views.machines_view import render_machines_view
from views.reports_view import render_reports_view


def inject_global_ui_css():
    st.markdown(
        """
        <style>
        /* =========================
           FORCE DARK APP THEME
           ========================= */

        .stApp {
            background: #060b14 !important;
            color: rgba(229,231,235,0.92) !important;
        }

        section.main > div { background: #060b14 !important; }
        header[data-testid="stHeader"] { background: #060b14 !important; }

        section[data-testid="stSidebar"] {
            background: #050a12 !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
        }
        section[data-testid="stSidebar"] > div { background: #050a12 !important; }

        /* =========================
           SIDEBAR RADIO READABILITY
           ========================= */

        /* Sidebar titles/captions */
        section[data-testid="stSidebar"] * {
            color: rgba(229,231,235,0.92) !important;
        }

        /* Radio items text (unselected) */
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            color: rgba(229,231,235,0.78) !important;
        }

        /* Make the selected item brighter + bold */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            color: rgba(229,231,235,1.0) !important;
            font-weight: 700 !important;
        }

        /* If :has is not supported (some builds), make all radio labels bright enough anyway */
        section[data-testid="stSidebar"] div[role="radiogroup"] label span {
            color: rgba(229,231,235,0.85) !important;
        }

        /* =========================
           INPUTS + SELECTS READABLE
           ========================= */

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            background-color: #0f172a !important;
            color: #e5e7eb !important;
            border-color: rgba(255,255,255,0.18) !important;
        }

        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {
            color: rgba(229,231,235,0.55) !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            color: #e5e7eb !important;
            border-color: rgba(255,255,255,0.18) !important;
        }
        div[data-baseweb="select"] span { color: #e5e7eb !important; }

        div[data-baseweb="popover"] > div {
            background-color: #0b1220 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }

        div[data-baseweb="menu"] { background-color: #0b1220 !important; }
        div[data-baseweb="menu"] li,
        div[data-baseweb="menu"] div { color: #e5e7eb !important; }

        div[data-baseweb="menu"] li:hover,
        div[data-baseweb="menu"] div:hover {
            background-color: rgba(255,255,255,0.08) !important;
        }

        div[data-baseweb="tag"] {
            background-color: rgba(255,255,255,0.10) !important;
            color: #e5e7eb !important;
            border-color: rgba(255,255,255,0.18) !important;
        }

        /* Focus outline (orange accent) */
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within {
            box-shadow: 0 0 0 2px rgba(255,165,0,0.35) !important;
            border-color: rgba(255,165,0,0.65) !important;
        }

        /* Buttons nicer */
        .stButton > button { border-radius: 10px !important; }

        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Earthworks", layout="wide")
    init_db()
    inject_global_ui_css()

    st.sidebar.title("Earthworks")
    st.sidebar.caption("Kaevetööd • Ressursid • Aruanded")

    # ---- ICONS BACK (emoji) + stable internal keys ----
    pages = {
        "Projects": "📁 Projects",
        "Workers": "👷 Workers",
        "Tasks": "✅ Tasks",
        "Machines": "🚜 Machines",
        "Reports": "📊 Reports",
    }

    page_key = st.sidebar.radio(
        "Menüü",
        options=list(pages.keys()),
        index=list(pages.keys()).index("Tasks"),
        format_func=lambda k: pages[k],
    )

    if page_key == "Projects":
        render_projects_view()
    elif page_key == "Workers":
        render_workers_view()
    elif page_key == "Tasks":
        render_tasks_view()
    elif page_key == "Machines":
        render_machines_view()
    elif page_key == "Reports":
        render_reports_view()


if __name__ == "__main__":
    main()
