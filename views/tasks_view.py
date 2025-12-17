import streamlit as st
from db import list_projects, add_task, list_tasks, set_task_deps, set_task_status

def render_tasks_view():
    st.title("Tasks + eeldustööd")

    projects = list_projects()
    if not projects:
        st.info("Loo enne projekt (Projects lehel).")
        return

    project_id = st.selectbox(
        "Vali projekt",
        options=[p["id"] for p in projects],
        format_func=lambda pid: next(x["name"] for x in projects if x["id"] == pid),
    )

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Lisa töö")
    name = st.text_input("Töö nimetus", placeholder="nt Freesimine / Asfalt / Haljastus")
    c1,c2 = st.columns([1,1])
    with c1:
        start = st.date_input("Algus (valikuline)", value=None)
    with c2:
        end = st.date_input("Lõpp (valikuline)", value=None)

    if st.button("Lisa töö", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta töö nimetus.")
        else:
            add_task(project_id, name.strip(), start, end)
            st.success("Lisatud.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    tasks = list_tasks(project_id)
    if not tasks:
        st.info("Lisa töid.")
        return

    st.subheader("🔗 Eeldustööd")
    task_id = st.selectbox(
        "Vali töö, millele määrad eeldused",
        options=[t["id"] for t in tasks],
        format_func=lambda tid: next(x["name"] for x in tasks if x["id"] == tid),
    )

    dep_options = [t for t in tasks if t["id"] != task_id]
    dep_ids = st.multiselect(
        "Vali eeldustööd (need peavad olema DONE)",
        options=[t["id"] for t in dep_options],
        format_func=lambda tid: next(x["name"] for x in dep_options if x["id"] == tid),
    )

    if st.button("Salvesta eeldused", use_container_width=True):
        set_task_deps(task_id, dep_ids)
        st.success("Eeldused salvestatud.")
        st.rerun()

    st.subheader("📋 Tööd")
    for t in tasks:
        blocked = t["blocked"] and t["status"] != "done"
        tag = "🔴 BLOKEERITUD" if blocked else "🟢 OK"
        st.write(f"**{t['name']}** • {t['status']} • {tag}")

        c1,c2 = st.columns([1,1])
        with c1:
            if st.button("Start", key=f"start_{t['id']}"):
                try:
                    set_task_status(t["id"], "in_progress")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        with c2:
            if st.button("Done", key=f"done_{t['id']}"):
                try:
                    set_task_status(t["id"], "done")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
