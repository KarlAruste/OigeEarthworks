# views/tasks_view.py
import streamlit as st
import pandas as pd
import plotly.express as px

from db import (
    list_projects,
    add_task,
    list_tasks,
    list_task_deps_by_project,   # <-- NEW
    set_task_deps,
    set_task_status,
    delete_task,
)

def _render_tasks_gantt_with_deps(tasks: list[dict], deps_rows: list[dict]):
    st.subheader("📊 Ajakava (Gantt + eeldused)")

    if not tasks:
        st.info("Selles projektis pole veel töid.")
        return

    df = pd.DataFrame(tasks).copy()

    # parse dates
    df["start_date"] = pd.to_datetime(df.get("start_date"), errors="coerce")
    df["end_date"] = pd.to_datetime(df.get("end_date"), errors="coerce")

    # only tasks with both dates can be drawn on gantt
    have_dates = df["start_date"].notna() & df["end_date"].notna()
    dfg = df.loc[have_dates].copy()
    missing = df.loc[~have_dates].copy()

    if dfg.empty:
        st.warning("Graafiku jaoks lisa töödele algus ja lõpp kuupäev.")
        if not missing.empty:
            st.caption("Tööd, millel kuupäevad puuduvad:")
            for _, r in missing.iterrows():
                st.write(f"• {r.get('name','(nimetu)')} — start: {r.get('start_date')} end: {r.get('end_date')}")
        return

    # Validate end >= start
    bad = dfg["end_date"] < dfg["start_date"]
    if bad.any():
        st.error("Mõnel tööl on Lõpp enne Algust. Paranda kuupäevad.")
        st.dataframe(dfg.loc[bad, ["name", "start_date", "end_date"]], use_container_width=True)
        return

    # Stable order (same as current list order)
    dfg["label"] = dfg["name"].astype(str)
    category_order = list(dfg["label"].values)

    fig = px.timeline(
        dfg,
        x_start="start_date",
        x_end="end_date",
        y="label",
        color="status",
        hover_data=["status", "blocked", "missing_deps", "start_date", "end_date"],
    )
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=category_order,
    )
    fig.update_layout(
        height=max(420, 34 * min(len(dfg), 25)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Kuupäev",
        yaxis_title="Töö",
        legend_title="Staatus",
    )

    # -----------------------------
    # Draw dependency arrows
    # -----------------------------
    # Build lookup by task_id for dates + label
    by_id = {}
    for _, r in dfg.iterrows():
        by_id[int(r["id"])] = {
            "label": r["label"],
            "start": r["start_date"],
            "end": r["end_date"],
        }

    # Add arrows: dep end -> task start
    arrow_count = 0
    for row in (deps_rows or []):
        try:
            task_id = int(row["task_id"])
            dep_id = int(row["dep_task_id"])
        except Exception:
            continue

        if task_id not in by_id or dep_id not in by_id:
            # If either task has missing dates, we skip
            continue

        t = by_id[task_id]
        d = by_id[dep_id]

        # Arrow from dependency END to task START
        x0 = d["end"]
        x1 = t["start"]
        y0 = d["label"]
        y1 = t["label"]

        # If somehow x1 is before x0, still draw (it highlights schedule issue)
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            opacity=0.75,
        )
        arrow_count += 1

    if arrow_count == 0:
        st.caption("Eelduste nooli ei kuvata (kas pole eeldusi või kuupäevad puuduvad).")

    st.plotly_chart(fig, use_container_width=True)


def render_tasks_view():
    st.title("Tasks + eeldustööd")

    projects = list_projects()
    if not projects:
        st.info("Loo enne projekt (Projects lehel).")
        return

    # pick project
    project_id = st.selectbox(
        "Vali projekt",
        options=[p["id"] for p in projects],
        format_func=lambda pid: next(x["name"] for x in projects if x["id"] == pid),
    )

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("➕ Lisa töö")

    name = st.text_input("Töö nimetus", placeholder="nt Freesimine / Asfalt / Haljastus")
    c1, c2 = st.columns([1, 1])
    with c1:
        start = st.date_input("Algus (valikuline)", value=None)
    with c2:
        end = st.date_input("Lõpp (valikuline)", value=None)

    if st.button("Lisa töö", use_container_width=True):
        if not name.strip():
            st.warning("Sisesta töö nimetus.")
        else:
            if start and end and end < start:
                st.error("Lõpp ei tohi olla enne algust.")
            else:
                add_task(project_id, name.strip(), start, end)
                st.success("Lisatud.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    tasks = list_tasks(project_id)
    if not tasks:
        st.info("Selles projektis pole veel töid.")
        return

    deps_rows = list_task_deps_by_project(project_id)

    # Gantt + dependencies
    st.markdown('<div class="block">', unsafe_allow_html=True)
    _render_tasks_gantt_with_deps(tasks, deps_rows)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dependencies editor
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

    if st.button("Salvesta eeldused", use_container_width=True):
        set_task_deps(task_id, dep_ids)
        st.success("Eeldused salvestatud.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Task list
    st.subheader("📋 Tööd (ainult see projekt)")

    for t in tasks:
        blocked = t["blocked"] and t["status"] != "done"
        tag = "🔴 BLOKEERITUD" if blocked else "🟢 OK"

        extra = ""
        if blocked:
            extra = f" (puudub {t.get('missing_deps', 0)} eeldust)"

        sd = t.get("start_date")
        ed = t.get("end_date")
        dates_txt = ""
        if sd or ed:
            dates_txt = f" • {sd or '-'} → {ed or '-'}"

        st.write(f"**{t['name']}** • {t['status']}{dates_txt} • {tag}{extra}")

        c1, c2, c3 = st.columns([1, 1, 1])
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
        with c3:
            if st.button("🗑️ Kustuta", key=f"del_{t['id']}"):
                delete_task(t["id"])
                st.rerun()
