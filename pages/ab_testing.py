import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing · UMKM",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light Mode Blue CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.main { background-color: #f0f6ff; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

.metric-card {
    background: #ffffff; border: 1px solid #d0e4f8;
    border-radius: 14px; padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 4px rgba(26,111,219,0.07);
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: #1a6fdb;
}
.metric-card.green::before  { background: #16a34a; }
.metric-card.red::before    { background: #dc2626; }
.metric-card.purple::before { background: #7c3aed; }

.metric-label { font-size: 0.73rem; font-weight: 700; letter-spacing: 0.08em; color: #64748b; text-transform: uppercase; margin-bottom: 0.35rem; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #1a2332; }
.metric-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem; }

.section-header {
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.12em;
    color: #1a6fdb; text-transform: uppercase;
    border-left: 3px solid #1a6fdb;
    padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
}

.result-box { border-radius: 10px; padding: 0.9rem 1.1rem; font-size: 0.86rem; line-height: 1.65; }
.result-sig   { background: #f0fdf4; border: 1px solid #86efac; color: #166534; }
.result-nosig { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; }
.result-warn  { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }

.step-pill {
    display: inline-block; background: #eff6ff;
    border: 1px solid #bfdbfe; color: #1d4ed8;
    border-radius: 20px; padding: 0.18rem 0.75rem;
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.5rem;
}
.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 0.8rem 1rem;
    font-size: 0.82rem; color: #1e3a5f; line-height: 1.6;
}

[data-testid="stSidebar"] {
    background: #ffffff; border-right: 1px solid #d0e4f8;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] p { color: #4a5568 !important; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Plotly constants ──────────────────────────────────────────────────────────
PAPER = "#ffffff"
PLOT  = "#f8fbff"
FONT  = "Plus Jakarta Sans"
PTMPL = "plotly_white"
COL_A = "#1a6fdb"
COL_B = "#e05252"

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/umkm_cashflow_final.csv", parse_dates=["tanggal"])
    df["bulan_label"] = df["tanggal"].dt.strftime("%b %Y")
    df["semester"]    = df["tanggal"].dt.month.apply(lambda m: "Semester 1" if m <= 6 else "Semester 2")
    df["tahun_label"] = df["tahun"].astype(str)
    return df

df       = load_data()
ALL_CATS = sorted(df["umkm_kategori"].unique().tolist())
ALL_YEARS= sorted(df["tahun"].unique().tolist())
METRICS  = {
    "Total Pemasukan":   "total_pemasukan",
    "Total Pengeluaran": "total_pengeluaran",
    "Net Cash Flow":     "net_cash_flow",
    "Jumlah Invoice":    "jumlah_invoice",
    "Total Item":        "total_item",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_rp(v):
    if abs(v) >= 1e9: return f"Rp {v/1e9:.2f}M"
    if abs(v) >= 1e6: return f"Rp {v/1e6:.1f} Jt"
    return f"Rp {v:,.0f}"

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pool = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return (np.mean(b)-np.mean(a)) / pool if pool > 0 else 0.0

def confidence_interval(a, b, alpha):
    diff = np.mean(b) - np.mean(a)
    se   = np.sqrt(np.var(a,ddof=1)/len(a) + np.var(b,ddof=1)/len(b))
    z    = stats.norm.ppf(1-alpha/2)
    return diff, diff-z*se, diff+z*se

def interpret_d(d):
    ad = abs(d)
    if ad < 0.2: return "sangat kecil"
    if ad < 0.5: return "kecil"
    if ad < 0.8: return "sedang"
    return "besar"

def sample_size_needed(base_mean, base_std, mde_pct, alpha=0.05, power=0.8):
    delta = abs(base_mean) * mde_pct / 100
    if delta == 0 or base_std == 0: return None
    d  = delta / base_std
    za = stats.norm.ppf(1-alpha/2)
    zb = stats.norm.ppf(power)
    return int(np.ceil(((za+zb)/d)**2))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2rem;'>🧪</div>
        <div style='font-size:1rem; font-weight:800; color:#1a2332;'>A/B Testing</div>
        <div style='font-size:0.72rem; color:#64748b; margin-top:0.2rem;'>UMKM Finance · CC26-PSU367</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Mode Pengujian**")
    mode = st.selectbox("Mode", [
        "Kategori vs Kategori",
        "Periode vs Periode",
        "Weekday vs Weekend",
        "Peak Month vs Normal Month",
    ], label_visibility="collapsed")

    st.markdown("**Metrik yang Diuji**")
    metric_label = st.selectbox("Metrik", list(METRICS.keys()), label_visibility="collapsed")
    metric_col   = METRICS[metric_label]

    st.markdown("**Significance Level (α)**")
    alpha = st.select_slider("α", options=[0.01,0.05,0.10], value=0.05, label_visibility="collapsed")

    st.markdown("**Jenis Uji**")
    test_type = st.radio("Uji", ["Two-tailed","One-tailed (B > A)"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#94a3b8; line-height:1.7;'>
        <b style='color:#64748b'>Uji yang digunakan:</b><br>
        Welch's t-test (parametrik)<br>
        Mann-Whitney U (non-parametrik)<br>
        Cohen's d (effect size)
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:#ffffff; border:1px solid #d0e4f8; border-radius:14px;
padding:1.2rem 1.5rem; margin-bottom:1.5rem; box-shadow:0 1px 4px rgba(26,111,219,0.07);'>
    <div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem;'>
        <div style='font-size:1.35rem; font-weight:800; color:#1a2332;'>🧪 A/B Testing UMKM</div>
        <div style='background:#1a6fdb; color:#fff; font-weight:700; font-size:0.72rem;
        padding:0.2rem 0.7rem; border-radius:20px; letter-spacing:0.05em;'>STATISTIK</div>
    </div>
    <div style='font-size:0.82rem; color:#64748b;'>
        Mode: <b style='color:#1a2332;'>{mode}</b> ·
        Metrik: <b style='color:#1a2332;'>{metric_label}</b> ·
        α = <b style='color:#1a2332;'>{alpha}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Group Selection ───────────────────────────────────────────────────────────
if mode == "Kategori vs Kategori":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="step-pill">GRUP A — Control</div>', unsafe_allow_html=True)
        cat_a = st.selectbox("Kategori A", ALL_CATS, index=0, key="cat_a")
    with col2:
        st.markdown('<div class="step-pill">GRUP B — Treatment</div>', unsafe_allow_html=True)
        cat_b = st.selectbox("Kategori B", [c for c in ALL_CATS if c != cat_a], index=0, key="cat_b")
    data_a = df[df["umkm_kategori"]==cat_a][metric_col].dropna().values
    data_b = df[df["umkm_kategori"]==cat_b][metric_col].dropna().values
    label_a, label_b = cat_a, cat_b

elif mode == "Periode vs Periode":
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        st.markdown('<div class="step-pill">PERIODE A — Control</div>', unsafe_allow_html=True)
        years_a = st.multiselect("Tahun A", ALL_YEARS, default=[ALL_YEARS[0]], key="yr_a")
        sems_a  = st.multiselect("Semester A", ["Semester 1","Semester 2"], default=["Semester 1"], key="sem_a")
    with col2:
        st.markdown('<div class="step-pill">PERIODE B — Treatment</div>', unsafe_allow_html=True)
        years_b = st.multiselect("Tahun B", ALL_YEARS, default=[ALL_YEARS[-1]], key="yr_b")
        sems_b  = st.multiselect("Semester B", ["Semester 1","Semester 2"], default=["Semester 1"], key="sem_b")
    with col3:
        st.markdown('<div class="step-pill">FILTER</div>', unsafe_allow_html=True)
        cats_filter = st.multiselect("Kategori", ALL_CATS, default=ALL_CATS, key="cats_f")
    dff    = df[df["umkm_kategori"].isin(cats_filter)]
    data_a = dff[(dff["tahun"].isin(years_a)) & (dff["semester"].isin(sems_a))][metric_col].dropna().values
    data_b = dff[(dff["tahun"].isin(years_b)) & (dff["semester"].isin(sems_b))][metric_col].dropna().values
    label_a = f"{'/'.join(str(y) for y in years_a)} {'/'.join(sems_a)}"
    label_b = f"{'/'.join(str(y) for y in years_b)} {'/'.join(sems_b)}"

elif mode == "Weekday vs Weekend":
    col1, col2 = st.columns(2)
    with col1: cats_filter  = st.multiselect("Filter Kategori", ALL_CATS, default=ALL_CATS, key="wk_cats")
    with col2: years_filter = st.multiselect("Filter Tahun",    ALL_YEARS, default=ALL_YEARS, key="wk_years")
    dff    = df[df["umkm_kategori"].isin(cats_filter) & df["tahun"].isin(years_filter)]
    data_a = dff[dff["is_weekend"]==0][metric_col].dropna().values
    data_b = dff[dff["is_weekend"]==1][metric_col].dropna().values
    label_a, label_b = "Weekday", "Weekend"

else:
    col1, col2 = st.columns(2)
    with col1: cats_filter  = st.multiselect("Filter Kategori", ALL_CATS, default=ALL_CATS, key="pm_cats")
    with col2: years_filter = st.multiselect("Filter Tahun",    ALL_YEARS, default=ALL_YEARS, key="pm_years")
    dff    = df[df["umkm_kategori"].isin(cats_filter) & df["tahun"].isin(years_filter)]
    data_a = dff[dff["is_peak_month"]==0][metric_col].dropna().values
    data_b = dff[dff["is_peak_month"]==1][metric_col].dropna().values
    label_a, label_b = "Normal Month", "Peak Month"

# ── Validasi ──────────────────────────────────────────────────────────────────
if len(data_a) < 10 or len(data_b) < 10:
    st.markdown('<div class="result-box result-warn">⚠️ Sample terlalu kecil (&lt;10). Sesuaikan filter atau pilih grup yang lebih besar.</div>', unsafe_allow_html=True)
    st.stop()

# ── Kalkulasi Statistik ───────────────────────────────────────────────────────
mean_a, mean_b = np.mean(data_a), np.mean(data_b)
std_a,  std_b  = np.std(data_a, ddof=1), np.std(data_b, ddof=1)
med_a,  med_b  = np.median(data_a), np.median(data_b)

t_stat, p_val_2 = stats.ttest_ind(data_a, data_b, equal_var=False)
p_val = p_val_2 if test_type == "Two-tailed" else p_val_2 / 2
if test_type == "One-tailed (B > A)" and mean_b <= mean_a:
    p_val = 1 - p_val

u_stat, p_mw  = mannwhitneyu(data_a, data_b, alternative="two-sided")
d             = cohens_d(data_a, data_b)
diff, lo, hi  = confidence_interval(data_a, data_b, alpha)
lift_pct      = (mean_b - mean_a) / abs(mean_a) * 100 if mean_a != 0 else 0
sig           = p_val < alpha

se_pool   = np.sqrt(std_a**2/len(data_a) + std_b**2/len(data_b))
z_alpha   = stats.norm.ppf(1-alpha/2)
power_est = 1 - stats.norm.cdf(z_alpha - abs(diff)/se_pool) if se_pool > 0 else 0

stat_a, p_sw_a, norm_a = (lambda s,p: (s,p,p>0.05))(*stats.shapiro(data_a[:5000]))
stat_b, p_sw_b, norm_b = (lambda s,p: (s,p,p>0.05))(*stats.shapiro(data_b[:5000]))

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Ringkasan Statistik</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, cls, label, val, sub in [
    (c1, "default", f"n {label_a[:8]}", f"{len(data_a):,}", f"σ={std_a/1e6:.1f}Jt" if metric_col not in ["jumlah_invoice","total_item"] else f"σ={std_a:.1f}"),
    (c2, "default", f"n {label_b[:8]}", f"{len(data_b):,}", f"σ={std_b/1e6:.1f}Jt" if metric_col not in ["jumlah_invoice","total_item"] else f"σ={std_b:.1f}"),
    (c3, "green" if lift_pct>=0 else "red", "Lift B vs A", f"{lift_pct:+.1f}%", "Δ mean"),
    (c4, "default", "p-value", f"{p_val:.4f}" if p_val>=0.0001 else "<0.0001", f"α = {alpha}"),
    (c5, "default", "Cohen's d", f"{d:.3f}", interpret_d(d)),
    (c6, "green" if sig else "red", "Kesimpulan", "Signifikan ✓" if sig else "Tidak Sig ✗", f"power={power_est:.0%}"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── Interpretasi ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
ci_str = (f"[{fmt_rp(lo)}, {fmt_rp(hi)}]"
          if metric_col not in ["jumlah_invoice","total_item"]
          else f"[{lo:,.1f}, {hi:,.1f}]")

if sig:
    st.markdown(f"""
    <div class="result-box result-sig">
        ✅ <b>Perbedaan SIGNIFIKAN</b> secara statistik (p={p_val:.4f} &lt; α={alpha}).<br>
        Grup <b>{label_b}</b> memiliki rata-rata {metric_label} lebih
        {"tinggi" if mean_b>mean_a else "rendah"} <b>{abs(lift_pct):.1f}%</b>
        dibanding <b>{label_a}</b>.<br>
        95% CI: <b>{ci_str}</b> · Cohen's d = <b>{d:.3f}</b> ({interpret_d(d)}) · Power = <b>{power_est:.0%}</b><br>
        Mann-Whitney U p = <b>{p_mw:.4f}</b> (konfirmasi non-parametrik)
    </div>""", unsafe_allow_html=True)
else:
    ss_needed = sample_size_needed(mean_a, std_a, 10, alpha=alpha)
    ss_txt = f"Butuh ≈ <b>{ss_needed:,} sampel/grup</b> untuk mendeteksi efek 10% dengan power 80%." if ss_needed else ""
    st.markdown(f"""
    <div class="result-box result-nosig">
        ⚠️ <b>Perbedaan TIDAK SIGNIFIKAN</b> (p={p_val:.4f} ≥ α={alpha}). Belum cukup bukti menolak H₀.<br>
        Lift observasi: <b>{lift_pct:+.1f}%</b> · 95% CI: <b>{ci_str}</b> · Cohen's d = <b>{d:.3f}</b> ({interpret_d(d)})<br>
        {ss_txt}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs Visualisasi ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Distribusi & Boxplot",
    "📉  Confidence Interval",
    "🔬  Uji Normalitas",
    "📋  Detail Statistik",
])

# ── TAB 1: Distribusi & Boxplot ───────────────────────────────────────────────
with tab1:
    scale_u = metric_col not in ["jumlah_invoice","total_item"]
    yunit   = "(Juta Rp)" if scale_u else ""

    # ── Histogram (kiri) ─────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Distribusi Harian — {metric_label} {yunit}</div>', unsafe_allow_html=True)
    fig_hist2 = go.Figure()
    for data, label, color in [(data_a, label_a, COL_A), (data_b, label_b, COL_B)]:
        y = data/1e6 if scale_u else data
        fig_hist2.add_trace(go.Histogram(
            x=y, name=label, opacity=0.65,
            marker_color=color, nbinsx=40,
        ))
    fig_hist2.update_layout(
        template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=300, barmode="overlay",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=f"{metric_label} {yunit}",
        yaxis_title="Frekuensi",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        font=dict(family=FONT),
    )
    st.plotly_chart(fig_hist2, use_container_width=True)

    # ── Boxplot (kanan) ──────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">Boxplot — {metric_label} {yunit}</div>', unsafe_allow_html=True)
    fig_box2 = go.Figure()
    for data, label, color in [(data_a, label_a, COL_A), (data_b, label_b, COL_B)]:
        y = data/1e6 if scale_u else data
        fig_box2.add_trace(go.Box(
            y=y, name=label, marker_color=color,
            boxmean="sd", boxpoints="outliers",
            line=dict(width=1.5),
        ))
    fig_box2.update_layout(
        template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title=f"{metric_label} {yunit}",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        font=dict(family=FONT),
    )
    st.plotly_chart(fig_box2, use_container_width=True)

    st.markdown('<div class="section-header">Violin Plot — Sebaran Lengkap</div>', unsafe_allow_html=True)
    fig_v = go.Figure()
    for data, label, color in [(data_a, label_a, COL_A), (data_b, label_b, COL_B)]:
        y = data/1e6 if scale_u else data
        fig_v.add_trace(go.Violin(y=y, name=label, line_color=color, box_visible=True, meanline_visible=True, opacity=0.7))
    fig_v.update_layout(
        template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        height=300, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_title=f"{metric_label} {'(Juta Rp)' if scale_u else ''}",
        font=dict(family=FONT),
    )
    st.plotly_chart(fig_v, use_container_width=True)

# ── TAB 2: Confidence Interval ────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns(2)
    scale  = 1e6 if metric_col not in ["jumlah_invoice","total_item"] else 1
    unit   = "Juta Rp" if scale == 1e6 else "unit"
    diff_s, lo_s, hi_s = diff/scale, lo/scale, hi/scale

    with col_l:
        st.markdown('<div class="section-header">Confidence Interval — Selisih Mean (B − A)</div>', unsafe_allow_html=True)
        fig_ci = go.Figure()
        fig_ci.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=0.5,
                         line=dict(color="rgba(220,38,38,0.5)", width=1.5, dash="dot"))
        fig_ci.add_trace(go.Scatter(x=[lo_s,hi_s], y=[0,0], mode="lines",
                                    line=dict(color=COL_A, width=7), name="95% CI"))
        fig_ci.add_trace(go.Scatter(x=[diff_s], y=[0], mode="markers",
                                    marker=dict(color=COL_B, size=14, symbol="diamond"), name="Δ mean"))
        fig_ci.add_annotation(x=diff_s, y=0.28, text=f"Δ = {diff_s:.2f} {unit}",
                               showarrow=False, font=dict(size=12, color="#1a2332"))
        fig_ci.update_layout(
            template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
            height=220, margin=dict(l=0,r=0,t=10,b=20),
            xaxis_title=f"Selisih ({unit})",
            yaxis=dict(visible=False, range=[-0.6,0.6]), font=dict(family=FONT),
        )
        st.plotly_chart(fig_ci, use_container_width=True)

        ci_interp = (f"✅ Seluruh CI positif → B secara konsisten lebih tinggi" if lo_s > 0
                     else f"❌ Seluruh CI negatif → B secara konsisten lebih rendah" if hi_s < 0
                     else f"⚠️ CI mencakup 0 → perbedaan mungkin tidak bermakna")
        cls = "result-sig" if lo_s>0 else "result-warn" if hi_s<0 else "result-nosig"
        st.markdown(f'<div class="result-box {cls}" style="font-size:0.82rem">{ci_interp}<br>CI [{lo_s:.2f}, {hi_s:.2f}] {unit}</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Mean ± 95% CI per Grup</div>', unsafe_allow_html=True)
        means = [mean_a/scale, mean_b/scale]
        ses   = [std_a/np.sqrt(len(data_a))/scale*1.96, std_b/np.sqrt(len(data_b))/scale*1.96]
        fig_mean = go.Figure()
        for lbl, mean, se, color in [(label_a,means[0],ses[0],COL_A),(label_b,means[1],ses[1],COL_B)]:
            fig_mean.add_trace(go.Bar(x=[lbl], y=[mean], name=lbl, marker_color=color, opacity=0.85,
                                      width=0.4, error_y=dict(type="data",array=[se],color="#475569",thickness=1.5,width=8)))
        fig_mean.update_layout(
            template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title=f"Mean {metric_label} ({unit})", showlegend=False, font=dict(family=FONT),
        )
        st.plotly_chart(fig_mean, use_container_width=True)

    if mode in ["Kategori vs Kategori","Weekday vs Weekend","Peak Month vs Normal Month"]:
        st.markdown('<div class="section-header">Rolling Mean (30 hari) — Tren Perbandingan</div>', unsafe_allow_html=True)
        if mode == "Kategori vs Kategori":
            sub_a = df[df["umkm_kategori"]==label_a].sort_values("tanggal")
            sub_b = df[df["umkm_kategori"]==label_b].sort_values("tanggal")
        elif mode == "Weekday vs Weekend":
            dfw   = df[df["umkm_kategori"].isin(cats_filter) & df["tahun"].isin(years_filter)]
            sub_a = dfw[dfw["is_weekend"]==0].sort_values("tanggal")
            sub_b = dfw[dfw["is_weekend"]==1].sort_values("tanggal")
        else:
            dfp   = df[df["umkm_kategori"].isin(cats_filter) & df["tahun"].isin(years_filter)]
            sub_a = dfp[dfp["is_peak_month"]==0].sort_values("tanggal")
            sub_b = dfp[dfp["is_peak_month"]==1].sort_values("tanggal")
        roll_a = sub_a.set_index("tanggal")[metric_col].rolling("30D").mean()/1e6
        roll_b = sub_b.set_index("tanggal")[metric_col].rolling("30D").mean()/1e6
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=roll_a.index, y=roll_a.values, name=label_a,
                                       line=dict(color=COL_A,width=2), fill="tozeroy", fillcolor="rgba(26,111,219,0.07)"))
        fig_roll.add_trace(go.Scatter(x=roll_b.index, y=roll_b.values, name=label_b,
                                       line=dict(color=COL_B,width=2), fill="tozeroy", fillcolor="rgba(224,82,82,0.07)"))
        fig_roll.update_layout(
            template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Juta Rp (rolling 30D)", xaxis_title=None,
            legend=dict(orientation="h", y=1.05), font=dict(family=FONT),
        )
        st.plotly_chart(fig_roll, use_container_width=True)

# ── TAB 3: Uji Normalitas ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Shapiro-Wilk Test (normalitas sampel)</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    for col, data, label, stat, p_sw, is_normal in [
        (col_l, data_a, label_a, stat_a, p_sw_a, norm_a),
        (col_r, data_b, label_b, stat_b, p_sw_b, norm_b),
    ]:
        with col:
            st.markdown(f"**{label}**")
            cls = "result-sig" if is_normal else "result-warn"
            msg = "Distribusi normal" if is_normal else "Distribusi tidak normal → Mann-Whitney lebih tepat"
            st.markdown(f'<div class="result-box {cls}" style="font-size:0.82rem">W = {stat:.4f} · p = {p_sw:.4f}<br>{msg}</div>', unsafe_allow_html=True)
            (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
            x_line = np.array([min(osm), max(osm)])
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                         marker=dict(color=COL_A if col==col_l else COL_B, size=4, opacity=0.55)))
            fig_qq.add_trace(go.Scatter(x=x_line, y=slope*x_line+intercept, mode="lines",
                                         line=dict(color="#dc2626", width=1.5, dash="dash")))
            fig_qq.update_layout(
                template=PTMPL, paper_bgcolor=PAPER, plot_bgcolor=PLOT,
                height=260, margin=dict(l=0,r=0,t=30,b=0),
                xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles",
                showlegend=False, font=dict(family=FONT),
                title=dict(text=f"Q-Q Plot — {label}", font=dict(size=12), x=0.5),
            )
            st.plotly_chart(fig_qq, use_container_width=True)

    if not norm_a or not norm_b:
        st.markdown('<div class="info-box">💡 <b>Catatan:</b> Salah satu distribusi <b>tidak normal</b>. Gunakan hasil <b>Mann-Whitney U</b> sebagai uji utama.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">✅ Kedua distribusi <b>normal</b>. Welch\'s t-test adalah pilihan tepat.</div>', unsafe_allow_html=True)

# ── TAB 4: Detail Statistik ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Tabel Perbandingan Statistik Deskriptif</div>', unsafe_allow_html=True)
    scale = 1e6 if metric_col not in ["jumlah_invoice","total_item"] else 1
    unit  = "Juta Rp" if scale == 1e6 else "unit"

    stat_table = pd.DataFrame({
        "Statistik": ["N","Mean","Median","Std Dev","Min","Q1 (25%)","Q3 (75%)","Max","Skewness","Kurtosis"],
        label_a: [f"{len(data_a):,}", f"{mean_a/scale:.2f} {unit}", f"{med_a/scale:.2f} {unit}",
                  f"{std_a/scale:.2f} {unit}", f"{np.min(data_a)/scale:.2f}", f"{np.percentile(data_a,25)/scale:.2f}",
                  f"{np.percentile(data_a,75)/scale:.2f}", f"{np.max(data_a)/scale:.2f}",
                  f"{stats.skew(data_a):.3f}", f"{stats.kurtosis(data_a):.3f}"],
        label_b: [f"{len(data_b):,}", f"{mean_b/scale:.2f} {unit}", f"{med_b/scale:.2f} {unit}",
                  f"{std_b/scale:.2f} {unit}", f"{np.min(data_b)/scale:.2f}", f"{np.percentile(data_b,25)/scale:.2f}",
                  f"{np.percentile(data_b,75)/scale:.2f}", f"{np.max(data_b)/scale:.2f}",
                  f"{stats.skew(data_b):.3f}", f"{stats.kurtosis(data_b):.3f}"],
    })
    st.dataframe(stat_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Ringkasan Hasil Pengujian</div>', unsafe_allow_html=True)
    test_table = pd.DataFrame({
        "Uji": ["Welch's t-test (parametrik)","Mann-Whitney U (non-parametrik)","Shapiro-Wilk A","Shapiro-Wilk B"],
        "Statistik": [f"t = {t_stat:.4f}", f"U = {u_stat:,.0f}", f"W = {stat_a:.4f}", f"W = {stat_b:.4f}"],
        "p-value": [f"{p_val:.4f}" if p_val>=0.0001 else "<0.0001",
                    f"{p_mw:.4f}"  if p_mw>=0.0001  else "<0.0001",
                    f"{p_sw_a:.4f}", f"{p_sw_b:.4f}"],
        f"Kesimpulan (α={alpha})": [
            "Signifikan ✓" if sig else "Tidak Signifikan ✗",
            "Signifikan ✓" if p_mw<alpha else "Tidak Signifikan ✗",
            "Normal ✓" if norm_a else "Tidak Normal ✗",
            "Normal ✓" if norm_b else "Tidak Normal ✗",
        ],
    })
    st.dataframe(test_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Effect Size & Power</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val in [
        (c1, "Cohen's d",      f"{d:.3f} ({interpret_d(d)})"),
        (c2, "Lift (%)",       f"{lift_pct:+.2f}%"),
        (c3, "Post-hoc power", f"{power_est:.1%}"),
        (c4, "Δ Mean",         fmt_rp(diff) if scale==1e6 else f"{diff:,.1f}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#e2e8f0; margin:2rem 0 1rem;">
<div style="text-align:center; font-size:0.75rem; color:#94a3b8; padding-bottom:1rem;">
    A/B Testing Module · Dashboard Keuangan UMKM · CC26-PSU367 · Coding Camp 2026 powered by DBS Foundation
</div>
""", unsafe_allow_html=True)
