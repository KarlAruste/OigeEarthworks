import streamlit as st

from db import init_db
from views.tasks_view import render_tasks_view

# Kui sul on muud view-d, impordi need ka:
# from views.projects_view import render_projects_view
# from views.workers_view import render_workers_view
# from views.machines_view import render_machines_view
# from views.reports_view import render_reports_view


def inject_global_ui_css():
    st.markdown(
        """
        <style>
        /* ========= Global readable dark UI ========= */

        /* General text */
        html, body, [class*="css"]  {
            color: rgba(229,231,235,0.92);
        }

        /* Labels */
        label, .stMarkdown, .stCaption {
          color: rgba(229,231,235,0.92) !important;
        }

        /* Inputs (text/date/number/textarea) */
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

        /* Select / Multiselect control */
        div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            color: #e5e7eb !important;
            border-color: rgba(255,255,255,0.18) !important;
        }
        div[data-baseweb="select"] span {
            color: #e5e7eb !important;
        }

        /* Dropdown menu container */
        div[data-baseweb="popover"] > div {
            background-color: #0b1220 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }

        /* Dropdown items */
        div[data-baseweb="menu"] {
            background-color: #0b1220 !important;
        }
        div[data-baseweb="menu"] li,
        div[data-baseweb="menu"] div {
            color: #e5e7eb !important;
        }
        div[data-baseweb="menu"] li:hover,
        div[data-baseweb="menu"] div:hover {
            background-color: rgba(255,255,255,0.08) !important;
        }

        /* Multiselect chips/tags */
        div[data-baseweb="tag"] {
            background-color: rgba(255,255,255,0.10) !important;
            color: #e5e7eb !important;
            border-color: rgba(255,255,255,0.18) !important;
        }

        /* Icons in inputs */
        div[data-baseweb="input"] svg {
            color: rgba(229,231,235,0.75) !important;
        }

        /* Focus outline (orange accent) */
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within {
            box-shadow: 0 0 0 2px rgba(255,165,0,0.35) !important;
            border-color: rgba(255,165,0,0.65) !important;
        }

        /* Buttons (optional: unify with your orange) */
        .stButton > button {
            border-radius: 10px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Earthworks", layout="wide")
    inject_global_ui_css()

    # DB init + migrations (important!)
    init_db()

    st.sidebar.title("Earthworks")
    st.sidebar.caption("Kaevetööd • Ressursid • Aruanded")

    page = st.sidebar.radio(
        "Menüü",
        ["Tasks"],  # lisa siia teised kui sul olemas: "Projects", "Workers", "Machines", "Reports"
        index=0,
    )

    if page == "Tasks":
        render_tasks_view()
    # elif page == "Projects":
    #     render_projects_view()
    # elif page == "Workers":
    #     render_workers_view()
    # elif page == "Machines":
    #     render_machines_view()
    # elif page == "Reports":
    #     render_reports_view()


if __name__ == "__main__":
    main()
