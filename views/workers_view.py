import streamlit as st
from datetime import date, timedelta

from db import (
    list_projects,
    add_worker,
    list_workers,
    add_assignment,
    list_assignments,
)

def render_workers_view():
    st.title("Workers & Assignments")

    projects = list_projects()
    if not projects:
        st.info("Loo enne projekt (Projects lehel).")
        return

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Lisa töötaja")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        name = st.text_input("Nimi", placeholder="nt Mari Maasikas")
    with c2:
        role = st.text_input("Roll", placeholder="nt Foreman / Worker")
    with c3:
        hourly = st.number_input("€/h", min_value=0.0, value=0.0, step=1.0)

    if st.button("Lisa töötaja", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta nimi.")
        else:
            add_worker(name.strip(), role.strip(), hourly)
            st.success("Lisatud.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    workers = list_workers(active_only=True)
    if not workers:
        st.info("Lisa vähemalt üks töötaja.")
        return

    st.subheader("📅 Broneeri töötaja projektile (topeltbroneering keelatud)")

    worker_id = st.selectbox(
        "Töötaja",
        options=[w["id"] for w in workers],
        format_func=lambda wid: next(x["name"] for x in workers if x["id"] == wid),
    )

    project_ids = [p["id"] for p in projects]
    default_pid = st.session_state.get("project_id")
    if default_pid not in project_ids:
        default_pid = project_ids[0]

    project_id = st.selectbox(
        "Projekt",
        options=project_ids,
        index=project_ids.index(default_pid),
        format_func=lambda pid: next(x["name"] for x in projects if x["id"] == pid),
    )

    d1, d2 = st.columns([1, 1])
    with d1:
        start = st.date_input("Algus", value=date.today())
    with d2:
        end = st.date_input("Lõpp", value=date.today() + timedelta(days=6))

    note = st.text_input("Märkus (valikuline)")

    if st.button("Salvesta broneering", use_container_width=True):
        try:
            add_assignment(worker_id, project_id, start, end, note)
            st.success("Broneering salvestatud.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.subheader("📌 Broneeringud (kõik projektid)")
    rows = list_assignments()
    if not rows:
        st.info("Broneeringuid pole.")
        return

    for r in rows[:300]:
        st.write(
            f"**{r['worker_name']}** → {r['project_name']} • "
            f"{r['start_date']} .. {r['end_date']} — {r.get('note','') or ''}"
        )
