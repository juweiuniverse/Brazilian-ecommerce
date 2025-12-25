# Olist E-Commerce Dashboard

**Deskripsi singkat**
Aplikasi Streamlit yang mem-visualisasikan hasil analisis pada dataset publik "Brazilian E-Commerce Public Dataset" (Olist). Aplikasi ini memuat data, melakukan pembersihan dan agregasi, lalu menampilkan dashboard untuk menjawab pertanyaan bisnis seperti produk terlaris, pengaruh keterlambatan pada kepuasan pelanggan, metode pembayaran, dan performa seller.

**Dataset**
- Dataset: brazilian-ecommerce (paket Olist)
- Sumber: Dari Kaglle `olistbr/brazilian-ecommerce`
- File utama:
  - `olist_orders_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_order_payments_dataset.csv`
  - `olist_order_reviews_dataset.csv`
  - `olist_products_dataset.csv`
  - `olist_sellers_dataset.csv`
  - `product_category_name_translation.csv`

**Cara menjalankan (lokal)**
1. Buat virtual environment dan aktifkan (Windows PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
2. Install dependensi:
```powershell
pip install -r requirements.txt
```
3. Siapkan dataset:
- Disini menggunakan API Kaggle

4. Jalankan dashboard:
```powershell
streamlit run app.py
```

**Untuk melihat secara online dapat mengakses: https://brazilian-ecommerce-olist6.streamlit.app/**

**Ringkasan insight hasil analisis**
- Produk terlaris tidak selalu memberikan revenue tertinggi, ada produk dengan volume besar namun unit price rendah.
- Keterlambatan pengiriman cenderung menurunkan rata-rata `review_score` (kepuasan pelanggan).
- Metode pembayaran yang paling populer dan berkontribusi besar ke revenue adalah **credit_card**.
- Terdapat variasi performa seller; beberapa negara bagian (misalnya SP) memiliki jumlah order tinggi, sementara beberapa lain memiliki rata-rata waktu pengiriman lebih cepat.

**Keamanan & kredensial**
- Contoh file untuk testing lokal ada di `.streamlit/secrets.toml.example` — salin dan isi `.streamlit/secrets.toml` untuk pengujian lokal.

