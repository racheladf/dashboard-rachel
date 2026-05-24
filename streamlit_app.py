import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from collections import Counter
import warnings

# Mengabaikan pesan peringatan agar tampilan tetap bersih
warnings.filterwarnings("ignore")

# =====================================================================
# LANGKAH 1: PENGATURAN AWAL & TEMA
# =====================================================================
# Mengatur judul tab browser dan layout menjadi lebar (wide)
st.set_page_config(page_title="E-Commerce Analytics", page_icon="🛍️", layout="wide")

# Mengatur palet warna agar seragam dan indah dipandang
PALETTE    = ["#38BDF8", "#818CF8", "#34D399", "#FB923C", "#F472B6", "#FACC15", "#A78BFA"]
GRID_COLOR = "#334155"
ACCENT     = "#38BDF8"
WARN       = "#FACC15"
POS        = "#34D399"

# Fungsi untuk menerapkan tema pada grafik Plotly
def terapkan_tema(fig, height=380):
    fig.update_layout(
        font=dict(family="sans-serif", size=12),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bordercolor=GRID_COLOR, borderwidth=1),
        hoverlabel=dict(bordercolor=ACCENT),
        height=height
    )
    return fig

# =====================================================================
# LANGKAH 2: FUNGSI PERAMALAN (TRIPLE EXPONENTIAL SMOOTHING)
# =====================================================================
def hitung_double_exponential(data_series, alpha, beta, langkah_kedepan):
    n = len(data_series)
    levels, trends = np.zeros(n), np.zeros(n)
    levels[0] = data_series[0]
    trends[0] = 0 if n <= 1 else (data_series[1] - data_series[0])
    for t in range(1, n):
        levels[t] = alpha * data_series[t] + (1 - alpha) * (levels[t-1] + trends[t-1])
        trends[t] = beta * (levels[t] - levels[t-1]) + (1 - beta) * trends[t-1]
    prediksi = [levels[-1] + m * trends[-1] for m in range(1, langkah_kedepan + 1)]
    return levels + trends, prediksi

def hitung_holt_winters(data_series, alpha, beta, gamma, L, langkah_kedepan):
    n = len(data_series)
    if n < 2 * L or L <= 1:
        return hitung_double_exponential(data_series, alpha, beta, langkah_kedepan)

    level = np.mean(data_series[:L])
    trend = np.mean(data_series[L:2*L] - data_series[:L]) / L
    seasonals = list(data_series[:L] - level)

    levels, trends, smoothed = np.zeros(n), np.zeros(n), np.zeros(n)
    levels[0], trends[0], smoothed[0] = level, trend, data_series[0]

    for t in range(1, n):
        val = data_series[t]
        s_prev = seasonals[t - L] if t >= L else seasonals[t % L]
        levels[t] = alpha * (val - s_prev) + (1 - alpha) * (levels[t-1] + trends[t-1])
        trends[t] = beta * (levels[t] - levels[t-1]) + (1 - beta) * trends[t-1]

        if t >= L: seasonals.append(gamma * (val - levels[t]) + (1 - gamma) * s_prev)
        else: seasonals[t] = gamma * (val - levels[t]) + (1 - gamma) * s_prev
        smoothed[t] = levels[t] + s_prev

    prediksi = [levels[-1] + m * trends[-1] + seasonals[n - L + (m - 1) % L] for m in range(1, langkah_kedepan + 1)]
    return smoothed, prediksi

def optimasi_holt_winters(data_series, L, langkah_kedepan):
    best_mse, best_params = float("inf"), (0.2, 0.1, 0.2)
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for b in [0.05, 0.1, 0.2]:
            for g in [0.1, 0.3, 0.5]:
                smoothed, _ = hitung_holt_winters(data_series, a, b, g, L, langkah_kedepan)
                mse = np.mean((data_series - smoothed) ** 2)
                if mse < best_mse:
                    best_mse, best_params = mse, (a, b, g)
    return best_params

# =====================================================================
# LANGKAH 3: MEMBACA DATA DARI GOOGLE SHEETS & PREPROCESSING
# =====================================================================
st.title("🛍️ E-Commerce Customer Behavior Analytics")
st.caption("Workshop Data Analitik | Live Customer Performance Monitor")
st.divider()

@st.cache_data(ttl=60)
def ambil_data():
    sheet_id = "1RFF-5XhqHbtHqXmzTVjgTmGekd763wyZsAEVhMJ-PAc"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    df = pd.read_csv(url)

    # ---------------------------------------------------------
    # A. Imputasi Missing Value
    # ---------------------------------------------------------
    for col in ["Customer_Rating", "Age", "Session_Duration_Minutes"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in ["City", "Device_Type", "Payment_Method", "Gender", "Product_Category"]:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ---------------------------------------------------------
    # B. Penghapusan Outlier (Metode IQR)
    # ---------------------------------------------------------
    kolom_outlier = ["Total_Amount", "Quantity"]
    for col in kolom_outlier:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        df = df[(df[col] >= batas_bawah) & (df[col] <= batas_atas)]

    # ---------------------------------------------------------
    # Persiapan Data Dasar (Wajib untuk grafik Streamlit)
    # ---------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Bulan"] = df["Date"].dt.to_period("M").astype(str)
    df["Tahun"] = df["Date"].dt.year
    df["Kelompok_Umur"] = pd.cut(
        df["Age"],
        bins=[0, 18, 25, 35, 45, 55, 100],
        labels=["≤18", "19-25", "26-35", "36-45", "46-55", "56+"],
    )
    df["Ada_Diskon"] = df["Discount_Amount"] > 0

    return df

try:
    data_awal = ambil_data()

    # =====================================================================
    # LANGKAH 4: MEMBUAT MENU FILTER DI SIDEBAR (SAMPING)
    # =====================================================================
    st.sidebar.header("Filter Data")

    min_tgl, max_tgl = data_awal["Date"].min().date(), data_awal["Date"].max().date()
    rentang_tanggal = st.sidebar.date_input("Rentang Tanggal", value=(min_tgl, max_tgl), min_value=min_tgl, max_value=max_tgl)

    pilih_kategori = st.sidebar.multiselect("Kategori Produk", options=sorted(data_awal["Product_Category"].unique()), default=sorted(data_awal["Product_Category"].unique()))
    pilih_kota = st.sidebar.multiselect("Kota (City)", options=sorted(data_awal["City"].unique()), default=sorted(data_awal["City"].unique()))
    pilih_perangkat = st.sidebar.multiselect("Perangkat (Device)", options=sorted(data_awal["Device_Type"].unique()), default=sorted(data_awal["Device_Type"].unique()))

    if st.sidebar.button("🔄 Sinkronisasi Data Terbaru"):
        st.cache_data.clear()
        st.rerun()

    data_filter = data_awal.copy()

    if isinstance(rentang_tanggal, tuple) and len(rentang_tanggal) == 2:
        data_filter = data_filter[(data_filter["Date"].dt.date >= rentang_tanggal[0]) & (data_filter["Date"].dt.date <= rentang_tanggal[1])]

    data_filter = data_filter[
        data_filter["Product_Category"].isin(pilih_kategori) &
        data_filter["City"].isin(pilih_kota) &
        data_filter["Device_Type"].isin(pilih_perangkat)
    ]

    st.sidebar.divider()
    st.sidebar.markdown(f"**📊 Data Terfilter:** `{len(data_filter):,}` dari `{len(data_awal):,}` baris")

    if data_filter.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
        st.stop()

    # =====================================================================
    # LANGKAH 5: MENAMPILKAN ANGKA RINGKASAN (KPI METRICS)
    # =====================================================================
    k1, k2, k3, k4 = st.columns(4)

    total_revenue = data_filter['Total_Amount'].sum()
    persentase_diskon = (data_filter['Discount_Amount'].sum() / total_revenue * 100) if total_revenue > 0 else 0

    k1.metric("💰 Total Revenue", f"₺{total_revenue:,.0f}", f"{len(data_filter):,} Transaksi")
    k2.metric("🏷️ Total Diskon", f"₺{data_filter['Discount_Amount'].sum():,.0f}", f"{persentase_diskon:.1f}% dari Revenue")
    k3.metric("⏱️ Avg Session Duration", f"{data_filter['Session_Duration_Minutes'].mean():.1f} Mnt", f"{data_filter['Pages_Viewed'].mean():.1f} hal dilihat")
    k4.metric("⭐ Avg Rating", f"{data_filter['Customer_Rating'].mean():.2f} / 5", f"{data_filter['Delivery_Time_Days'].mean():.1f} hari avg kirim")

    st.divider()

    # =====================================================================
    # LANGKAH 6: MEMBUAT HALAMAN TABS
    # =====================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Executive Summary & KPI",
        "👥 Demografi & Segmentasi",
        "🏷️ Strategi Produk & Harga",
        "📋 Data Center",
    ])

    # ---------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # ---------------------------------------------------------
    with tab1:
        per_bulan = data_filter.groupby("Bulan").agg(Revenue=("Total_Amount", "sum"), Transactions=("Order_ID", "count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=per_bulan["Bulan"], y=per_bulan["Revenue"], name="Revenue (₺)", marker_color=ACCENT), secondary_y=False)
        fig.add_trace(go.Scatter(x=per_bulan["Bulan"], y=per_bulan["Transactions"], name="Volume Transaksi", line=dict(color=WARN, width=3)), secondary_y=True)
        fig = terapkan_tema(fig, 350)
        fig.update_layout(title="Tren Pendapatan & Volume Transaksi Bulanan", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            per_kategori = data_filter.groupby("Product_Category")["Total_Amount"].sum().reset_index().sort_values("Total_Amount")
            fig = px.bar(per_kategori, x="Total_Amount", y="Product_Category", orientation="h", color="Total_Amount",
                         color_continuous_scale=["#1E293B", ACCENT], title="Revenue per Kategori Produk", text=per_kategori["Total_Amount"].apply(lambda v: f"₺{v:,.0f}"))
            st.plotly_chart(terapkan_tema(fig.update_coloraxes(showscale=False), 350), use_container_width=True)
        with c2:
            per_bayar = data_filter.groupby("Payment_Method")["Total_Amount"].sum().reset_index()
            fig = px.pie(per_bayar, values="Total_Amount", names="Payment_Method", color_discrete_sequence=PALETTE, hole=0.5, title="Proporsi Metode Pembayaran")
            st.plotly_chart(terapkan_tema(fig, 350), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: DEMOGRAFI & SEGMENTASI PELANGGAN
    # ---------------------------------------------------------
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            per_umur = data_filter.groupby("Kelompok_Umur", observed=True).agg(Jumlah=("Customer_ID", "nunique")).reset_index()
            fig = px.pie(per_umur, values="Jumlah", names="Kelompok_Umur", color_discrete_sequence=PALETTE, hole=0.5, title="Komposisi Kelompok Umur")
            st.plotly_chart(terapkan_tema(fig, 350), use_container_width=True)
        with c2:
            per_gender = data_filter.groupby("Gender").agg(Jumlah=("Customer_ID", "nunique")).reset_index()
            fig = px.pie(per_gender, values="Jumlah", names="Gender", color_discrete_sequence=[ACCENT, "#F472B6", POS], hole=0.5, title="Komposisi Gender")
            st.plotly_chart(terapkan_tema(fig, 350), use_container_width=True)
        with c3:
            per_kota = data_filter.groupby("City").agg(Jumlah=("Customer_ID", "nunique")).reset_index().sort_values("Jumlah").tail(10)
            fig = px.bar(per_kota, x="Jumlah", y="City", orientation="h", color="Jumlah", color_continuous_scale=["#1E293B", ACCENT], title="Top 10 Kota Terbanyak Pelanggan")
            st.plotly_chart(terapkan_tema(fig.update_coloraxes(showscale=False), 350), use_container_width=True)

        st.subheader("💰 Perbandingan Rata-rata Pengeluaran")
        c1, c2 = st.columns(2)
        with c1:
            kota_top = data_filter.groupby("City")["Total_Amount"].count().nlargest(10).index.tolist()
            spending_kota = data_filter[data_filter["City"].isin(kota_top)].groupby("City")["Total_Amount"].mean().reset_index().sort_values("Total_Amount")
            fig = px.bar(spending_kota, x="Total_Amount", y="City", orientation="h", color="Total_Amount", color_continuous_scale=["#1E293B", ACCENT], title="Rata-rata Pengeluaran - Top 10 Kota")
            st.plotly_chart(terapkan_tema(fig.update_coloraxes(showscale=False), 350), use_container_width=True)
        with c2:
            spending_device = data_filter.groupby("Device_Type")["Total_Amount"].mean().reset_index().sort_values("Total_Amount", ascending=False)
            fig = px.bar(spending_device, x="Device_Type", y="Total_Amount", color="Device_Type", color_discrete_sequence=PALETTE, title="Rata-rata Pengeluaran per Perangkat")
            st.plotly_chart(terapkan_tema(fig, 350), use_container_width=True)

        st.subheader("🚚 Waktu Pengiriman & Distribusi Rating")
        c1, c2 = st.columns(2)
        with c1:
            kota_5 = data_filter.groupby("City")["Order_ID"].count().nlargest(5).index.tolist()
            fig = px.histogram(data_filter[data_filter["City"].isin(kota_5)], x="Delivery_Time_Days", color="City", nbins=30, color_discrete_sequence=PALETTE, title="Sebaran Waktu Kirim (Top 5 Kota)")
            st.plotly_chart(terapkan_tema(fig, 350), use_container_width=True)
        with c2:
            dist_rating = data_filter["Customer_Rating"].value_counts().sort_index().reset_index()
            dist_rating.columns = ["Rating", "Jumlah"]
            fig = px.bar(dist_rating, x="Rating", y="Jumlah", color="Rating", color_continuous_scale=["#EF4444", "#F97316", "#EAB308", "#84CC16", "#22C55E"], title="Distribusi Rating Pelanggan")
            st.plotly_chart(terapkan_tema(fig.update_coloraxes(showscale=False), 350), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: STRATEGI PRODUK & HARGA (DENGAN FORECASTING)
    # ---------------------------------------------------------
    with tab3:
        st.subheader("📦 Kontribusi Revenue per Kategori Produk")
        rev_cat = data_filter.groupby("Product_Category").agg(Revenue=("Total_Amount", "sum")).reset_index()
        fig = px.treemap(rev_cat, path=["Product_Category"], values="Revenue", color="Revenue", color_continuous_scale=["#1E293B", ACCENT, "#818CF8"])
        st.plotly_chart(terapkan_tema(fig.update_coloraxes(showscale=False), 400), use_container_width=True)

        # Pengaturan Model Forecasting
        fc_col1, fc_col2 = st.columns([1, 2])
        with fc_col1:
            st.markdown("#### ⚙️ Konfigurasi Prediksi")
            selected_cat = st.selectbox("Pilih Kategori Produk:", options=sorted(data_filter["Product_Category"].unique()))
            L_period = st.number_input("Periode Musiman (Minggu):", min_value=2, max_value=12, value=4)

            alpha, beta, gamma = 0.2, 0.1, 0.2
            auto_fit = st.checkbox("Optimasi Otomatis", value=True)
            if not auto_fit:
                alpha = 1.0 - (st.slider("Level Smoothing:", 0, 99, 80) / 100.0)
                beta = 1.0 - (st.slider("Trend Smoothing:", 0, 99, 90) / 100.0)
                gamma = 1.0 - (st.slider("Seasonal Smoothing:", 0, 99, 75) / 100.0)

        with fc_col2:
            df_cat = data_filter[data_filter["Product_Category"] == selected_cat]
            if not df_cat.empty:
                df_weekly = df_cat.groupby(df_cat["Date"].dt.to_period("W")).agg(Quantity=("Quantity", "sum")).reset_index()
                df_weekly["Tanggal_Minggu"] = df_weekly["Date"].dt.start_time.sort_values()

                if (data_filter["Date"].max() - df_weekly["Tanggal_Minggu"].iloc[-1]).days + 1 < 7:
                    df_weekly = df_weekly.iloc[:-1]

                y = df_weekly["Quantity"].values
                if len(y) >= 2:
                    future_steps = 8

                    if auto_fit:
                        alpha, beta, gamma = optimasi_holt_winters(y, int(L_period), future_steps)
                        st.success(f"🤖 **Parameter Optimal:** α=`{alpha:.2f}`, β=`{beta:.2f}`, γ=`{gamma:.2f}`")

                    smoothed, forecast = hitung_holt_winters(y, alpha, beta, gamma, int(L_period), future_steps)
                    forecast = np.clip(forecast, 0, None)
                    future_dates = [df_weekly["Tanggal_Minggu"].max() + pd.Timedelta(weeks=i+1) for i in range(future_steps)]

                    fig_fc = go.Figure([
                        go.Scatter(x=df_weekly["Tanggal_Minggu"], y=y, name="Data Aktual", mode="lines+markers", line=dict(color="rgba(56, 189, 248, 0.4)", width=2)),
                        go.Scatter(x=df_weekly["Tanggal_Minggu"], y=smoothed, name="Hasil Fitting", line=dict(color=ACCENT, width=3)),
                        go.Scatter(x=[df_weekly["Tanggal_Minggu"].iloc[-1]] + future_dates, y=[y[-1]] + list(forecast), name="Proyeksi 8 Minggu", line=dict(color=WARN, width=3, dash="dash"))
                    ])
                    fig_fc.update_layout(title=f"Peramalan Kuantitas {selected_cat}", hovermode="x unified")
                    st.plotly_chart(terapkan_tema(fig_fc, 420), use_container_width=True)

                    m1, m2 = st.columns(2)
                    m1.metric("📊 Rata-rata Aktual Historis", f"{np.mean(y):.1f} Item / minggu")
                    m2.metric("🔮 Total Proyeksi Permintaan", f"{np.sum(forecast):.0f} Item dalam 8 mggu depan")
                else: st.warning("Data kurang (minimal butuh 2 minggu data lengkap).")

        st.subheader("🛒 Produk Terpopuler di Setiap Kota & Produk Sering Dibeli Bersama")
        cust_cats = data_filter.groupby("Customer_ID")["Product_Category"].apply(set).reset_index()
        cust_multi = cust_cats[cust_cats["Product_Category"].apply(len) >= 2]

        if not cust_multi.empty:
            pairs_counter = Counter([pair for cat_set in cust_multi["Product_Category"] for pair in combinations(sorted(cat_set), 2)])
            top_pairs = pd.DataFrame([(p[0], p[1], count) for p, count in pairs_counter.most_common(10)], columns=["Kategori A", "Kategori B", "Jumlah"])
            top_pairs["Pasangan"] = top_pairs["Kategori A"] + " + " + top_pairs["Kategori B"]

            c_m1, c_m2 = st.columns(2)
            with c_m1:
                fig = px.bar(top_pairs.sort_values("Jumlah"), x="Jumlah", y="Pasangan", orientation="h", color="Jumlah", color_continuous_scale=["#1E293B", ACCENT], title="Top Pasangan Kategori (Cross-Selling)")
                st.plotly_chart(terapkan_tema(fig.update_coloraxes(showscale=False), 350), use_container_width=True)
            with c_m2:
                st.dataframe(top_pairs[["Kategori A", "Kategori B", "Jumlah"]], use_container_width=True)
                top_1 = top_pairs.iloc[0]
                st.success(f"💡 **Rekomendasi Bundling:** Pelanggan yang membeli **{top_1['Kategori A']}** sangat sering membeli **{top_1['Kategori B']}**.")

    # ---------------------------------------------------------
    # TAB 4: DATA CENTER & UNDUH
    # ---------------------------------------------------------
    with tab4:
        st.subheader("📋 Data Center & Eksplorasi Data")

        csv_data = data_filter.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Unduh Data Terfilter (CSV)", data=csv_data, file_name="ecommerce_data_terfilter.csv", mime="text/csv")

        with st.expander("Tabel Pivot (Untuk Eksplorasi Cepat)", expanded=True):
            pv1, pv2, pv3, pv4 = st.columns(4)
            pv_row = pv1.selectbox("Baris:", ["City", "Gender", "Product_Category", "Payment_Method", "Device_Type", "Is_Returning_Customer"])
            pv_col = pv2.selectbox("Kolom:", ["Kelompok_Umur", "Is_Returning_Customer", "Ada_Diskon", "Gender"])
            pv_val = pv3.selectbox("Nilai:", ["Total_Amount", "Discount_Amount", "Quantity", "Customer_Rating", "Session_Duration_Minutes"])
            pv_agg = pv4.selectbox("Agregasi:", ["sum", "mean", "count"])

            pivot = data_filter.pivot_table(index=pv_row, columns=pv_col, values=pv_val, aggfunc=pv_agg).round(2)
            st.dataframe(pivot, use_container_width=True)

        st.subheader("Log Transaksi Pelanggan")
        st.dataframe(data_filter[["Order_ID", "Customer_ID", "Date", "City", "Gender", "Product_Category", "Unit_Price", "Quantity", "Total_Amount", "Payment_Method"]].reset_index(drop=True), use_container_width=True, height=400)

except Exception as e:
    st.error("Gagal memuat data. Pastikan koneksi internet aktif untuk mengakses Google Sheets.")
    st.exception(e)

st.divider()
st.caption("📊 E-Commerce Dashboard | Diperbarui untuk Pemula dengan Sumber Data Google Sheets")