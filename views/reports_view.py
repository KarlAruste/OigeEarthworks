# views/reports_view.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from db import (
    get_project,
    list_cost_items,
    add_cost_item,
    add_production,
    add_revenue,
    daily_profit_series,
)


def render_reports_view():
    st.title("Reports: kasum/kaotus graafikud")

    project_id = st.session_state.get("active_project_id")
    if not project_id:
        st.info("Vali enne projekt (Projects lehel).")
        return

    p = get_project(project_id)
    if not p:
        st.error("Aktiivset projekti ei leitud. Vali projekt uuesti (Projects).")
        return

    st.caption(f"Aktiivne projekt: **{p['name']}** • tähtaeg: **{p.get('end_date')}**")

    # ---------------- COST ITEMS ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Hinnakiri (Cost items)")

    ci_name = st.text_input("Nimetus", placeholder="nt Kaevetöö", key="ci_name")
    c1, c2 = st.columns([1, 1])
    with c1:
        unit = st.text_input("Ühik", placeholder="m3 / m2 / h", key="ci_unit")
    with c2:
        price = st.number_input("Ühiku hind (€)", min_value=0.0, value=0.0, step=0.5, key="ci_price")

    if st.button("Lisa hinnakirja", use_container_width=True, key="ci_add_btn"):
        if ci_name.strip() and unit.strip():
            add_cost_item(ci_name.strip(), unit.strip(), float(price))
            st.success("Lisatud.")
            st.rerun()
        else:
            st.warning("Täida nimetus ja ühik.")

    cost_items = list_cost_items()
    if cost_items:
        df_ci = pd.DataFrame(cost_items)
        # proovime ilusti näidata
        cols = [c for c in ["name", "unit", "unit_price"] if c in df_ci.columns]
        st.dataframe(df_ci[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Hinnakiri on tühi. Lisa vähemalt 1 cost item.")

    st.markdown("</div>", unsafe_allow_html=True)

    # kui hinnakirja pole, ei saa kulusid lisada
    if not cost_items:
        return

    # ---------------- ADD COST (PRODUCTION) ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Lisa kulu (tehtud töö)")

    cost_item_id = st.selectbox(
        "Töö",
        options=[c["id"] for c in cost_items],
        format_func=lambda cid: next(x["name"] for x in cost_items if x["id"] == cid),
        key="prod_cost_item",
    )
    chosen = next(x for x in cost_items if x["id"] == cost_item_id)
    unit_label = chosen.get("unit", "")
    unit_price = float(chosen.get("unit_price", 0.0))
    st.caption(f"Ühik: {unit_label} • Hind: {unit_price:.2f} €/ühik")

    c3, c4 = st.columns([1, 1])
    with c3:
        qty = st.number_input(f"Kogus ({unit_label})", min_value=0.0, value=0.0, step=1.0, key="prod_qty")
    with c4:
        work_date = st.date_input("Kuupäev", value=date.today(), key="prod_date")

    note = st.text_input("Märkus", key="prod_note")

    if st.button("Salvesta kulu", use_container_width=True, key="prod_save"):
        add_production(project_id, cost_item_id, float(qty), work_date, note)
        st.success("Kulu salvestatud.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- ADD REVENUE ----------------
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Lisa tulu (arve/laekumine)")

    c5, c6 = st.columns([1, 1])
    with c5:
        amount = st.number_input("Summa €", min_value=0.0, value=0.0, step=10.0, key="rev_amount")
    with c6:
        rev_date = st.date_input("Kuupäev", value=date.today(), key="rev_date")

    rev_note = st.text_input("Märkus", key="rev_note")

    if st.button("Salvesta tulu", use_container_width=True, key="rev_save"):
        add_revenue(project_id, float(amount), rev_date, rev_note)
        st.success("Tulu salvestatud.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- SERIES + GRAPHS ----------------
    rows = daily_profit_series(project_id)
    df = pd.DataFrame(rows)

    if df.empty:
        st.info("Sellel projektil pole veel kulusid/tulu.")
        return

    # normalise columns
    for col in ["revenue", "cost", "profit"]:
        if col not in df.columns:
            df[col] = 0.0

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["cum_profit"] = df["profit"].cumsum()

    total_rev = float(df["revenue"].sum())
    total_cost = float(df["cost"].sum())
    total_profit = float(df["profit"].sum())

    a, b, c = st.columns(3)
    a.metric("Tulu kokku", f"{total_rev:.2f} €")
    b.metric("Kulu kokku", f"{total_cost:.2f} €")
    c.metric("Kasum/kaotus", f"{total_profit:.2f} €")

    st.subheader("📈 Päevane kasum/kaotus")
    fig1 = plt.figure()
    plt.plot(df["date"], df["profit"])
    plt.xlabel("Date")
    plt.ylabel("Profit (€)")
    st.pyplot(fig1, clear_figure=True)

    st.subheader("📈 Kumulatiivne kasum/kaotus")
    fig2 = plt.figure()
    plt.plot(df["date"], df["cum_profit"])
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit (€)")
    st.pyplot(fig2, clear_figure=True)

    st.subheader("📋 Päevade tabel")
    show = df[["date", "revenue", "cost", "profit", "cum_profit"]].copy()
    show["date"] = show["date"].dt.date
    st.dataframe(show, use_container_width=True, hide_index=True)
