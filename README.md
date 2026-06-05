# 💰 Dashboard Keuangan UMKM

**Sistem Informasi Manajemen Keuangan UMKM Terintegrasi**  
Coding Camp 2026 powered by DBS Foundation · Tim **CC26-PSU367**

[![Streamlit App]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://umkm-finance.streamlit.app](https://data-sciencegit-pyxfhoatmwrpe2sq8iekei.streamlit.app/))

---

## 📌 Deskripsi

Dashboard interaktif untuk visualisasi dan analitik arus kas UMKM, mencakup:

- **Tren & Arus Kas** — pemasukan, pengeluaran, net cash flow, moving average
- **Analisis per Kategori** — 4 kategori UMKM: Warung Kelontong, Usaha Kuliner, Toko Pakaian, Jasa & Servis
- **Pola Temporal** — harian, mingguan, bulanan, heatmap
- **Korelasi & Distribusi** — matriks korelasi, boxplot, lag features
- **Data Explorer** — filter, preview, download CSV

## 🗂️ Struktur Repo

```
umkm_dashboard/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── data/
│   ├── umkm_cashflow_final.csv   # Dataset utama (2.568 baris, 22 kolom)
│   ├── umkm_train.csv
│   ├── umkm_validation.csv
│   └── umkm_test.csv
└── README.md
```

## 📊 Dataset

| Atribut | Nilai |
|---------|-------|
| Sumber pemasukan | UCI Online Retail II (data nyata, CC BY 4.0) |
| Sumber pengeluaran | Data sintetik UMKM Indonesia |
| Periode | 1 Des 2009 – 9 Des 2011 |
| Total baris | 2.568 |
| Total fitur | 22 kolom |
| Kategori | Warung Kelontong, Usaha Kuliner, Toko Pakaian, Jasa & Servis |

## 🚀 Cara Deploy ke Streamlit Cloud

1. **Fork / push repo ini ke GitHub**
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **"New app"**
4. Isi:
   - Repository: `username/umkm_dashboard`
   - Branch: `main`
   - Main file path: `app.py`
5. Klik **"Deploy!"**

## 💻 Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👥 Tim

| Nama | ID | Role |
|------|----|------|
| Fidia Dewi Wulandari Batu Bara | CACC189D6X0705 | AI Engineer |
| Kenia Nurma Feblia | CDCC189D6X0733 | Data Scientist |
| Ammar Siraj Ananda | CACC189D6Y1344 | AI Engineer |
| Mohammad Dimas Al Shiddiq | CFCC189D6Y1409 | Full-Stack Web Dev |
| Bunga Diva Putri Wijaya | CDCC299D6X1668 | Data Scientist |
| Azzahra Faranisa | CFCC189D6X2776 | Full-Stack Web Dev |

---

*Coding Camp 2026 powered by DBS Foundation*
