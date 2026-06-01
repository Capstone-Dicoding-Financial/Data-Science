import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Keuangan UMKM",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.main { background-color: #0f1117; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #1e2538 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
}
.metric-card.income::before { background: linear-gradient(90deg, #43e97b, #38f9d7); }
.metric-card.expense::before { background: linear-gradient(90deg, #f77062, #fe5196); }
.metric-card.cashflow::before { background: linear-gradient(90deg, #a18cd1, #fbc2eb); }
.metric-card.invoice::before { background: linear-gradient(90deg, #fbd786, #f7797d); }

.metric-label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
    color: #8892b0; text-transform: uppercase; margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem; font-weight: 700; color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.metric-delta { font-size: 0.8rem; font-weight: 600; }
.metric-delta.pos { color: #68d391; }
.metric-delta.neg { color: #fc8181; }

/* Section headers */
.section-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    color: #4facfe; text-transform: uppercase;
    border-left: 3px solid #4facfe;
    padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid rgba(99,179,237,0.1);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] p {
    color: #8892b0 !important;
    font-size: 0.82rem;
}

/* Header */
.dashboard-header {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.2rem 1.5rem;
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid rgba(79,172,254,0.15);
    border-radius: 16px; margin-bottom: 1.5rem;
}
.header-badge {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: #0a0e1a; font-weight: 800; font-size: 0.75rem;
    padding: 0.25rem 0.75rem; border-radius: 20px;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.header-title { font-size: 1.4rem; font-weight: 800; color: #e2e8f0; }
.header-sub { font-size: 0.82rem; color: #8892b0; margin-top: 0.15rem; }

/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.82rem; font-weight: 600;
    color: #8892b0; letter-spacing: 0.04em;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #4facfe;
}

/* Insight boxes */
.insight-box {
    background: rgba(79,172,254,0.06);
    border: 1px solid rgba(79,172,254,0.2);
    border-radius: 12px; padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.insight-box.warning {
    background: rgba(247,112,98,0.06);
    border-color: rgba(247,112,98,0.25);
}
.insight-box.success {
    background: rgba(67,233,123,0.06);
    border-color: rgba(67,233,123,0.25);
}
.insight-icon { font-size: 1.1rem; margin-right: 0.5rem; }
.insight-text { font-size: 0.84rem; color: #cbd5e0; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/umkm_cashflow_final.csv", parse_dates=["tanggal"])
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["bulan_nama"] = df["tanggal"].dt.strftime("%b %Y")
    df["minggu"] = df["tanggal"].dt.to_period("W").dt.start_time
    return df

df = load_data()

CAT_COLORS = {
    "Warung Kelontong": "#4facfe",
    "Usaha Kuliner":    "#f77062",
    "Toko Pakaian":     "#a78bfa",
    "Jasa & Servis":    "#43e97b",
}
ALL_CATS = sorted(df["umkm_kategori"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2rem;'>💰</div>
        <div style='font-size:1rem; font-weight:800; color:#e2e8f0;'>UMKM Finance</div>
        <div style='font-size:0.72rem; color:#8892b0; margin-top:0.25rem;'>CC26-PSU367 · DBS Foundation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Filter Kategori**")
    selected_cats = st.multiselect(
        "Pilih Kategori UMKM",
        options=ALL_CATS,
        default=ALL_CATS,
        label_visibility="collapsed",
    )

    st.markdown("**Rentang Tanggal**")
    min_date = df["tanggal"].min().date()
    max_date = df["tanggal"].max().date()
    date_start = st.date_input("Dari", value=min_date, min_value=min_date, max_value=max_date)
    date_end   = st.date_input("Sampai", value=max_date, min_value=min_date, max_value=max_date)

    st.markdown("**Granularitas Tren**")
    granularity = st.selectbox(
        "Tampilkan tren per",
        ["Harian", "Mingguan", "Bulanan"],
        index=2,
        label_visibility="collapsed",
    )

    st.markdown("**Split Data**")
    selected_split = st.multiselect(
        "Split",
        options=["train", "validation", "test"],
        default=["train", "validation", "test"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#586070; text-align:center; line-height:1.7;'>
        Data: UCI Online Retail II + Sintetik<br>
        Periode: Des 2009 – Des 2011<br>
        Model: LSTM/GRU (TensorFlow)
    </div>
    """, unsafe_allow_html=True)

# ── Filter Data ───────────────────────────────────────────────────────────────
if not selected_cats:
    selected_cats = ALL_CATS

dff = df[
    (df["umkm_kategori"].isin(selected_cats)) &
    (df["tanggal"].dt.date >= date_start) &
    (df["tanggal"].dt.date <= date_end) &
    (df["split"].isin(selected_split if selected_split else ["train","validation","test"]))
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem;">
            <div class="header-title">Dashboard Keuangan UMKM</div>
            <div class="header-badge">Live Analytics</div>
        </div>
        <div class="header-sub">
            Sistem Informasi Manajemen Keuangan Terintegrasi · {date_start.strftime("%d %b %Y")} – {date_end.strftime("%d %b %Y")}
            · {len(selected_cats)} Kategori · {len(dff):,} Data Points
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_income   = dff["total_pemasukan"].sum()
total_expense  = dff["total_pengeluaran"].sum()
net_cf         = dff["net_cash_flow"].sum()
total_invoice  = dff["jumlah_invoice"].sum()

# Compare to full dataset for delta
full_income  = df[df["umkm_kategori"].isin(selected_cats)]["total_pemasukan"].sum()
full_expense = df[df["umkm_kategori"].isin(selected_cats)]["total_pengeluaran"].sum()

def fmt_rp(v):
    if abs(v) >= 1e9:
        return f"Rp {v/1e9:.2f}M"
    elif abs(v) >= 1e6:
        return f"Rp {v/1e6:.1f} Jt"
    else:
        return f"Rp {v:,.0f}"

c1, c2, c3, c4 = st.columns(4)
for col, cls, label, val, icon in [
    (c1, "income",   "Total Pemasukan",   total_income,  "📈"),
    (c2, "expense",  "Total Pengeluaran", total_expense, "📉"),
    (c3, "cashflow", "Net Cash Flow",     net_cf,        "💵"),
    (c4, "invoice",  "Total Invoice",     total_invoice, "🧾"),
]:
    delta_val = val / full_income * 100 if cls != "invoice" else None
    delta_html = ""
    if delta_val:
        s = "pos" if val >= 0 else "neg"
        delta_html = f'<div class="metric-delta {s}">{"▲" if val>=0 else "▼"} {abs(delta_val):.1f}% dari total</div>'
    with col:
        st.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{fmt_rp(val) if cls != 'invoice' else f'{val:,.0f}'}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Tren & Arus Kas",
    "🏪  Analisis per Kategori",
    "📅  Pola Temporal",
    "🔍  Korelasi & Distribusi",
    "📋  Data Explorer",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Tren & Arus Kas
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Tren Pemasukan vs Pengeluaran</div>', unsafe_allow_html=True)

    # Aggregate by granularity
    if granularity == "Harian":
        grp_col = "tanggal"
    elif granularity == "Mingguan":
        dff["_grp"] = dff["tanggal"].dt.to_period("W").dt.start_time
        grp_col = "_grp"
    else:
        dff["_grp"] = dff["tanggal"].dt.to_period("M").dt.start_time
        grp_col = "_grp"

    agg = dff.groupby(grp_col)[["total_pemasukan","total_pengeluaran","net_cash_flow"]].sum().reset_index()
    agg.columns = ["tanggal","pemasukan","pengeluaran","net_cf"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=agg["tanggal"], y=agg["pemasukan"]/1e6,
        name="Pemasukan", fill="tozeroy",
        line=dict(color="#43e97b", width=2),
        fillcolor="rgba(67,233,123,0.12)",
    ))
    fig1.add_trace(go.Scatter(
        x=agg["tanggal"], y=agg["pengeluaran"]/1e6,
        name="Pengeluaran", fill="tozeroy",
        line=dict(color="#f77062", width=2),
        fillcolor="rgba(247,112,98,0.12)",
    ))
    fig1.add_trace(go.Scatter(
        x=agg["tanggal"], y=agg["net_cf"]/1e6,
        name="Net Cash Flow",
        line=dict(color="#4facfe", width=2.5, dash="solid"),
    ))
    fig1.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05, x=0),
        yaxis_title="Juta Rp", xaxis_title=None,
        font=dict(family="Plus Jakarta Sans"),
        hovermode="x unified",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Net Cash Flow per kategori
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Net Cash Flow per Kategori</div>', unsafe_allow_html=True)
        agg_cat = dff.groupby(["tanggal" if granularity=="Harian" else "_grp", "umkm_kategori"])["net_cash_flow"].sum().reset_index()
        agg_cat.columns = ["tanggal","kategori","net_cf"]

        fig2 = px.line(
            agg_cat, x="tanggal", y=agg_cat["net_cf"]/1e6,
            color="kategori", color_discrete_map=CAT_COLORS,
        )
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
            yaxis_title="Juta Rp", xaxis_title=None,
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Cashflow Kumulatif</div>', unsafe_allow_html=True)
        agg_cum = dff.groupby(["tanggal","umkm_kategori"])["cashflow_kumulatif"].last().reset_index()

        fig3 = px.line(
            agg_cum, x="tanggal", y=agg_cum["cashflow_kumulatif"]/1e6,
            color="umkm_kategori", color_discrete_map=CAT_COLORS,
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="rgba(255,100,100,0.5)")
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
            yaxis_title="Kumulatif (Juta Rp)", xaxis_title=None,
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Moving Averages
    st.markdown('<div class="section-header">Moving Average — MA7 vs MA30</div>', unsafe_allow_html=True)
    cat_sel_ma = st.selectbox("Pilih kategori untuk MA:", selected_cats, key="ma_cat")
    sub_ma = dff[dff["umkm_kategori"] == cat_sel_ma].sort_values("tanggal")

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=sub_ma["tanggal"], y=sub_ma["net_cash_flow"]/1e6,
        name="Net CF Harian", line=dict(color="#586070", width=0.8), opacity=0.5,
    ))
    fig_ma.add_trace(go.Scatter(
        x=sub_ma["tanggal"], y=sub_ma["cashflow_ma7"]/1e6,
        name="MA-7", line=dict(color="#fbd786", width=1.8, dash="dash"),
    ))
    fig_ma.add_trace(go.Scatter(
        x=sub_ma["tanggal"], y=sub_ma["cashflow_ma30"]/1e6,
        name="MA-30", line=dict(color="#4facfe", width=2.5),
    ))
    fig_ma.add_hline(y=0, line_dash="dot", line_color="rgba(255,80,80,0.5)")
    fig_ma.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Juta Rp", xaxis_title=None,
        font=dict(family="Plus Jakarta Sans"),
    )
    st.plotly_chart(fig_ma, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Analisis per Kategori
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Komposisi Pemasukan</div>', unsafe_allow_html=True)
        comp = dff.groupby("umkm_kategori")[["total_pemasukan","total_pengeluaran","net_cash_flow"]].sum().reset_index()
        fig_pie = px.pie(
            comp, values="total_pemasukan", names="umkm_kategori",
            color="umkm_kategori", color_discrete_map=CAT_COLORS,
            hole=0.55,
        )
        fig_pie.update_traces(textposition="outside", textfont_size=12)
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117",
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=-0.1),
            font=dict(family="Plus Jakarta Sans"),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Pemasukan vs Pengeluaran</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        for label, col_name, color in [
            ("Pemasukan", "total_pemasukan", "#43e97b"),
            ("Pengeluaran", "total_pengeluaran", "#f77062"),
        ]:
            fig_bar.add_trace(go.Bar(
                x=comp["umkm_kategori"],
                y=comp[col_name]/1e6,
                name=label,
                marker_color=color,
                opacity=0.85,
            ))
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            barmode="group", yaxis_title="Juta Rp",
            legend=dict(orientation="h", y=1.05),
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Per-category metrics table
    st.markdown('<div class="section-header">Ringkasan Metrik per Kategori</div>', unsafe_allow_html=True)
    summary = dff.groupby("umkm_kategori").agg(
        Total_Pemasukan=("total_pemasukan", "sum"),
        Total_Pengeluaran=("total_pengeluaran", "sum"),
        Net_Cash_Flow=("net_cash_flow", "sum"),
        Rata_Invoice=("jumlah_invoice", "mean"),
        Total_Item=("total_item", "sum"),
        Hari_Aktif=("tanggal", "count"),
    ).reset_index()
    summary["Profit_Margin_%"] = (summary["Net_Cash_Flow"] / summary["Total_Pemasukan"] * 100).round(1)

    for _, row in summary.iterrows():
        cat = row["umkm_kategori"]
        color = CAT_COLORS.get(cat, "#4facfe")
        margin_cls = "success" if row["Profit_Margin_%"] > 0 else "warning"
        st.markdown(f"""
        <div class="insight-box {margin_cls}">
            <span style="color:{color}; font-weight:700;">{cat}</span>
            &nbsp;&nbsp;
            <span class="insight-text">
                Pemasukan: <b>{fmt_rp(row['Total_Pemasukan'])}</b> ·
                Pengeluaran: <b>{fmt_rp(row['Total_Pengeluaran'])}</b> ·
                Net CF: <b>{fmt_rp(row['Net_Cash_Flow'])}</b> ·
                Margin: <b>{row['Profit_Margin_%']}%</b> ·
                Hari Aktif: <b>{int(row['Hari_Aktif'])}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Scatter: Pemasukan vs Pengeluaran harian
    st.markdown('<div class="section-header">Scatter: Pemasukan vs Pengeluaran Harian</div>', unsafe_allow_html=True)
    fig_scat = px.scatter(
        dff, x="total_pemasukan", y="total_pengeluaran",
        color="umkm_kategori", color_discrete_map=CAT_COLORS,
        opacity=0.5, size_max=6,
        labels={"total_pemasukan":"Pemasukan (Rp)","total_pengeluaran":"Pengeluaran (Rp)"},
        trendline="ols",
    )
    fig_scat.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
        font=dict(family="Plus Jakarta Sans"),
    )
    st.plotly_chart(fig_scat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Pola Temporal
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    col_l, col_r = st.columns(2)

    # Day of week pattern
    with col_l:
        st.markdown('<div class="section-header">Pola Hari dalam Seminggu</div>', unsafe_allow_html=True)
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_id    = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
        day_map   = dict(zip(day_order, day_id))
        dff_day = dff.copy()
        dff_day["nama_hari_id"] = dff_day["nama_hari"].map(day_map)
        dow = dff_day.groupby(["nama_hari","umkm_kategori"])["net_cash_flow"].mean().reset_index()
        dow["hari_id"] = dow["nama_hari"].map(day_map)
        dow["day_num"] = dow["nama_hari"].map({d:i for i,d in enumerate(day_order)})
        dow = dow.sort_values("day_num")

        fig_dow = px.bar(
            dow, x="hari_id", y=dow["net_cash_flow"]/1e6,
            color="umkm_kategori", color_discrete_map=CAT_COLORS,
            barmode="group",
        )
        fig_dow.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Rata-rata Net CF (Juta Rp)", xaxis_title=None,
            legend=dict(orientation="h", y=1.1, font=dict(size=10)),
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    # Monthly pattern
    with col_r:
        st.markdown('<div class="section-header">Pola Bulanan (Rata-rata)</div>', unsafe_allow_html=True)
        monthly = dff.groupby(["bulan","umkm_kategori"])["net_cash_flow"].mean().reset_index()
        monthly.columns = ["bulan","kategori","avg_net_cf"]

        fig_mon = px.line(
            monthly, x="bulan", y=monthly["avg_net_cf"]/1e6,
            color="kategori", color_discrete_map=CAT_COLORS,
            markers=True,
        )
        fig_mon.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Rata-rata Net CF (Juta Rp)", xaxis_title="Bulan",
            legend=dict(orientation="h", y=1.1, font=dict(size=10)),
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    # Heatmap — bulan vs hari
    st.markdown('<div class="section-header">Heatmap: Rata-rata Pemasukan per Bulan × Hari</div>', unsafe_allow_html=True)
    cat_heat = st.selectbox("Kategori:", selected_cats, key="heat_cat")
    sub_heat = dff[dff["umkm_kategori"] == cat_heat].copy()
    sub_heat["hari_id"] = sub_heat["nama_hari"].map(day_map)
    pivot = sub_heat.pivot_table(
        values="total_pemasukan", index="hari_id",
        columns="bulan", aggfunc="mean"
    )
    month_order = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
    cols_sorted = [m for m in month_order if m in pivot.columns]
    rows_sorted = [d for d in day_id if d in pivot.index]
    pivot = pivot.reindex(index=rows_sorted, columns=cols_sorted)

    fig_heat = px.imshow(
        pivot/1e6, aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Juta Rp"),
    )
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117",
        height=260, margin=dict(l=0,r=0,t=10,b=0),
        font=dict(family="Plus Jakarta Sans"),
        xaxis_title="Bulan", yaxis_title="Hari",
        coloraxis_colorbar=dict(title="Jt Rp", len=0.8),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Weekend vs Weekday
    st.markdown('<div class="section-header">Weekend vs Weekday</div>', unsafe_allow_html=True)
    wknd = dff.groupby(["is_weekend","umkm_kategori"])["net_cash_flow"].mean().reset_index()
    wknd["tipe_hari"] = wknd["is_weekend"].map({0:"Weekday",1:"Weekend"})
    fig_wk = px.bar(
        wknd, x="umkm_kategori", y=wknd["net_cash_flow"]/1e6,
        color="tipe_hari",
        color_discrete_map={"Weekday":"#4facfe","Weekend":"#f77062"},
        barmode="group",
    )
    fig_wk.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_title="Rata-rata Net CF (Juta Rp)", xaxis_title=None,
        legend=dict(orientation="h", y=1.05),
        font=dict(family="Plus Jakarta Sans"),
    )
    st.plotly_chart(fig_wk, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Korelasi & Distribusi
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Distribusi Net Cash Flow</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            dff, x=dff["net_cash_flow"]/1e6, color="umkm_kategori",
            color_discrete_map=CAT_COLORS, opacity=0.7, barmode="overlay",
            nbins=50,
            labels={"x":"Net Cash Flow (Juta Rp)"},
        )
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.08, font=dict(size=10)),
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Boxplot Pemasukan per Kategori</div>', unsafe_allow_html=True)
        fig_box = px.box(
            dff, x="umkm_kategori", y=dff["total_pemasukan"]/1e6,
            color="umkm_kategori", color_discrete_map=CAT_COLORS,
            labels={"y":"Pemasukan (Juta Rp)","umkm_kategori":"Kategori"},
        )
        fig_box.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False, xaxis_title=None,
            font=dict(family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Correlation matrix
    st.markdown('<div class="section-header">Matriks Korelasi Fitur Numerik</div>', unsafe_allow_html=True)
    num_cols = ["total_pemasukan","total_pengeluaran","net_cash_flow",
                "jumlah_invoice","total_item","cashflow_ma7","cashflow_ma30"]
    corr = dff[num_cols].corr()
    labels_id = ["Pemasukan","Pengeluaran","Net CF","Invoice","Item","MA7","MA30"]
    fig_corr = px.imshow(
        corr, x=labels_id, y=labels_id,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f", aspect="auto",
    )
    fig_corr.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117",
        height=350, margin=dict(l=0,r=0,t=10,b=0),
        font=dict(family="Plus Jakarta Sans"),
        coloraxis_colorbar=dict(title="r", len=0.8),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Lag feature insights
    st.markdown('<div class="section-header">Lag Features — Pemasukan vs Lag-7 & Lag-30</div>', unsafe_allow_html=True)
    cat_lag = st.selectbox("Pilih kategori:", selected_cats, key="lag_cat")
    sub_lag = dff[dff["umkm_kategori"]==cat_lag].dropna(subset=["income_lag7","income_lag30"])

    fig_lag = make_subplots(cols=2, shared_yaxes=True,
                             subplot_titles=["Pemasukan vs Lag-7","Pemasukan vs Lag-30"])
    for col_idx, (lag_col, title) in enumerate([("income_lag7","Lag-7"),("income_lag30","Lag-30")], 1):
        fig_lag.add_trace(go.Scatter(
            x=sub_lag["total_pemasukan"]/1e6, y=sub_lag[lag_col]/1e6,
            mode="markers", marker=dict(color="#4facfe", opacity=0.4, size=5),
            name=title,
        ), row=1, col=col_idx)
    fig_lag.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1f2e",
        height=280, margin=dict(l=0,r=0,t=30,b=0),
        showlegend=False,
        font=dict(family="Plus Jakarta Sans"),
    )
    st.plotly_chart(fig_lag, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Eksplorasi Dataset</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Baris (filter aktif)", f"{len(dff):,}")
    with col_b:
        st.metric("Kolom", f"{len(dff.columns)}")
    with col_c:
        st.metric("Periode (hari)", f"{(dff['tanggal'].max()-dff['tanggal'].min()).days:,}")

    cols_show = st.multiselect(
        "Pilih kolom yang ditampilkan:",
        options=df.columns.tolist(),
        default=["tanggal","umkm_kategori","total_pemasukan","total_pengeluaran",
                 "net_cash_flow","jumlah_invoice","cashflow_ma30","split"],
    )

    n_rows = st.slider("Jumlah baris yang ditampilkan:", 10, 200, 50)

    show_df = dff[cols_show].sort_values("tanggal", ascending=False).head(n_rows).copy()
    # Format currency cols for display
    for c in ["total_pemasukan","total_pengeluaran","net_cash_flow","cashflow_ma7","cashflow_ma30",
              "cashflow_kumulatif","income_lag7","income_lag30","cashflow_lag1","cashflow_lag7"]:
        if c in show_df.columns:
            show_df[c] = show_df[c].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "-")

    st.dataframe(show_df, use_container_width=True, height=400)

    # Statistik deskriptif
    st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
    num_c = ["total_pemasukan","total_pengeluaran","net_cash_flow","jumlah_invoice","total_item"]
    desc = dff[num_c].describe().T.reset_index()
    desc.columns = ["Kolom","N","Mean","Std","Min","Q1","Median","Q3","Max"]
    for c in ["Mean","Std","Min","Q1","Median","Q3","Max"]:
        desc[c] = desc[c].apply(lambda x: f"Rp {x:,.0f}" if abs(x) >= 1000 else f"{x:.1f}")
    st.dataframe(desc, use_container_width=True)

    # Download
    st.markdown('<div class="section-header">Unduh Data</div>', unsafe_allow_html=True)
    csv_export = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download CSV (filter aktif)",
        data=csv_export,
        file_name=f"umkm_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(99,179,237,0.1); margin: 2rem 0 1rem;">
<div style="text-align:center; font-size:0.75rem; color:#586070; padding-bottom:1rem;">
    Dashboard Keuangan UMKM · Coding Camp 2026 powered by DBS Foundation · Tim CC26-PSU367<br>
    Data: UCI Online Retail II (nyata) + Sintetik UMKM Indonesia · Periode: Des 2009 – Des 2011
</div>
""", unsafe_allow_html=True)
