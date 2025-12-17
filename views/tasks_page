import streamlit as st
from db import list_projects, add_task, list_tasks, set_task_deps, set_task_status

def render_tasks_page():
    st.title("Tasks + eeldustööd")

    projects = list_projects()
    if not projects:
        st.info("Loo enne projekt (Projects lehel).")
        return

    p_map = {p["name"]: p["id"] for p in projects}
    proj_name = st.selectbox("Vali projekt", list(p_map.keys()))
    project_id = p_map[proj_name]

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Lisa task")
    name = st.text_input("Taski nimi", placeholder="nt Alus / Asfalt / Haljastus")
    c1,c2 = st.columns([1,1])
    with c1:
        start = st.date_input("Algus (valikuline)", value=None)
    with c2:
        end = st.date_input("Lõpp (valikuline)", value=None)

    if st.button("Lisa task", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta taski nimi.")
        else:
            add_task(project_id, name.strip(), start, end)
            st.success("Lisatud.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    tasks = list_tasks(project_id)
    if not tasks:
        st.info("Lisa taskid.")
        return

    st.subheader("🔗 Eeldustööd (dependencies)")
    t_labels = {f"{t['name']} (id={t['id']})": t["id"] for t in tasks}
    selected_task_label = st.selectbox("Vali task", list(t_labels.keys()))
    task_id = t_labels[selected_task_label]

    candidates = [lbl for lbl, tid in t_labels.items() if tid != task_id]
    deps = st.multiselect("Vali eeldustööd (peavad olema DONE)", candidates)

    if st.button("Salvesta eeldused", use_container_width=True):
        dep_ids = [t_labels[d] for d in deps]
        set_task_deps(task_id, dep_ids)
        st.success("Eeldused salvestatud.")
        st.rerun()

    st.subheader("📋 Taskid")
    for t in tasks:
        blocked = t["blocked"] and t["status"] != "done"
        tag = "🔴 BLOCKED" if blocked else "🟢 OK"
        st.write(f"**{t['name']}** • status={t['status']} • {tag} • end={t.get('end_date')}")
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
