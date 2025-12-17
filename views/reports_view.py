import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from db import list_projects, list_cost_items, add_cost_item, add_production, add_revenue, daily_profit_series

def render_reports_view():
    st.title("Reports: kasum/kaotus graafikud")

    projects = list_projects()
    if not projects:
        st.info("Loo enne projekt (Projects lehel).")
        return

    project_id = st.selectbox(
        "Vali projekt",
        options=[p["id"] for p in projects],
        format_func=lambda pid: next(x["name"] for x in projects if x["id"] == pid),
    )
    p = next(x for x in projects if x["id"] == project_id)
    st.caption(f"Projekt tähtaeg: **{p.get('end_date')}**")

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Hinnakiri (Cost items)")
    ci_name = st.text_input("Nimetus", placeholder="nt Kaevetöö")
    c1,c2 = st.columns([1,1])
    with c1:
        unit = st.text_input("Ühik", placeholder="m3 / m2 / h")
    with c2:
        price = st.number_input("Ühiku hind", min_value=0.0, value=0.0, step=0.5)

    if st.button("Lisa hinnakirja", use_container_width=True):
        if ci_name.strip() and unit.strip():
            add_cost_item(ci_name.strip(), unit.strip(), price)
            st.success("Lisatud.")
            st.rerun()
        else:
            st.warning("Täida nimetus ja ühik.")
    st.markdown('</div>', unsafe_allow_html=True)

    cost_items = list_cost_items()
    if not cost_items:
        st.info("Lisa vähemalt 1 cost item.")
        return

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Lisa kulu (tehtud töö)")
    cost_item_id = st.selectbox(
        "Töö",
        options=[c["id"] for c in cost_items],
        format_func=lambda cid: next(x["name"] for x in cost_items if x["id"] == cid),
    )
    chosen = next(x for x in cost_items if x["id"] == cost_item_id)
    st.caption(f"Ühik: {chosen['unit']} • Hind: {float(chosen['unit_price']):.2f} €/ühik")

    qty = st.number_input("Kogus", min_value=0.0, value=0.0, step=1.0)
    work_date = st.date_input("Kuupäev", value=date.today(), key="work_date")
    note = st.text_input("Märkus", key="work_note")

    if st.button("Salvesta kulu", use_container_width=True):
        add_production(project_id, cost_item_id, qty, work_date, note)
        st.success("Kulu salvestatud.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Lisa tulu (arve/laekumine)")
    amount = st.number_input("Summa €", min_value=0.0, value=0.0, step=10.0)
    rev_date = st.date_input("Kuupäev", value=date.today(), key="rev_date")
    rev_note = st.text_input("Märkus", key="rev_note")

    if st.button("Salvesta tulu", use_container_width=True):
        add_revenue(project_id, amount, rev_date, rev_note)
        st.success("Tulu salvestatud.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    rows = daily_profit_series(project_id)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Pole veel kulusid/tulu.")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["cum_profit"] = df["profit"].cumsum()

    total_rev = float(df["revenue"].sum())
    total_cost = float(df["cost"].sum())
    total_profit = float(df["profit"].sum())

    a,b,c = st.columns(3)
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
    st.dataframe(df[["date","revenue","cost","profit","cum_profit"]], use_container_width=True)
