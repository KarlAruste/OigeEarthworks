import streamlit as st
from db import (
    list_projects,
    add_task,
    list_tasks,
    set_task_deps,
    set_task_status,
    delete_task,
    list_task_deps,
    delete_task_dep,
)

def render_tasks_view():
    st.title("Tasks + eeldustööd")

    projects = list_projects()
    if not projects:
        st.info("Loo enne projekt (Projects lehel).")
        return

    # Kui sul on aktiivne projekt session_state-s, kasuta seda
    default_pid = st.session_state.get("project_id")
    project_ids = [p["id"] for p in projects]
    if default_pid not in project_ids:
        default_pid = project_ids[0]

    project_id = st.selectbox(
        "Aktiivne projekt",
        options=project_ids,
        index=project_ids.index(default_pid),
        format_func=lambda pid: next(x["name"] for x in projects if x["id"] == pid),
    )

    st.caption(f"Aktiivne projekt: {next(x['name'] for x in projects if x['id']==project_id)}")

    # ---------------- Add task ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Lisa töö")
    name = st.text_input("Töö nimetus", placeholder="nt Freesimine / Asfalt / Haljastus")

    if st.button("Lisa töö", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta töö nimetus.")
        else:
            add_task(project_id, name.strip())
            st.success("Lisatud.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    tasks = list_tasks(project_id)
    if not tasks:
        st.info("Lisa töid.")
        return

    # ---------------- Dependencies editor ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
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

    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("Salvesta eeldused", use_container_width=True):
            set_task_deps(task_id, dep_ids)
            st.success("Eeldused salvestatud.")
            st.rerun()

    with cB:
        if st.button("Eemalda kõik eeldused", use_container_width=True):
            set_task_deps(task_id, [])
            st.success("Eeldused eemaldatud.")
            st.rerun()

    # olemasolevad eeldused + remove one-by-one
    deps = list_task_deps(task_id)
    if deps:
        st.write("**Selle töö eeldused:**")
        for d in deps:
            row1, row2 = st.columns([6, 2])
            with row1:
                st.write(f"- {d['name']} ({d['status']})")
            with row2:
                if st.button("❌ Eemalda", key=f"rmdep_{task_id}_{d['dep_task_id']}", use_container_width=True):
                    delete_task_dep(task_id, d["dep_task_id"])
                    st.rerun()
    else:
        st.caption("Eeldusi pole.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Task list ----------------
    st.subheader("📋 Tööd (ainult see projekt)")

    for t in tasks:
        blocked = t.get("blocked") and t["status"] != "done"
        tag = "🔴 BLOKEERITUD" if blocked else "🟢 OK"
        st.write(f"**{t['name']}** • {t['status']} • {tag}")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Start", key=f"start_{t['id']}", use_container_width=True):
                try:
                    set_task_status(t["id"], "in_progress")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c2:
            if st.button("Done", key=f"done_{t['id']}", use_container_width=True):
                try:
                    set_task_status(t["id"], "done")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        with c3:
            if st.button("🗑️ Kustuta", key=f"del_{t['id']}", use_container_width=True):
                delete_task(t["id"])
                st.rerun()

        st.divider()
