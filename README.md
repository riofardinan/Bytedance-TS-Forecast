# Bytedance-TS-Forecast

Aplikasi prediksi workload (usage / QPS) per layanan cloud menggunakan model LSTM. Mendukung empat layanan: **FAAS**, **PAAS**, **IAAS**, dan **RDS**, masing-masing dengan metrik dan model terlatih sendiri.

---

## Deskripsi Proyek

Proyek ini memprediksi nilai metrik workload **langkah berikutnya** (next-step forecast) dari riwayat 12 langkah terakhir. Model LSTM dilatih dengan data time series dari ByteDance/CloudTimeSeriesData. Pengguna dapat mengunggah CSV time series atau memakai sample dari folder `sample/`, lalu mendapatkan prediksi.

---

## Fitur

- **Multi-service:** Pilih layanan (FAAS, PAAS, IAAS, RDS) dengan metrik masing-masing (QPS atau CPU Usage).
- **Input fleksibel:** Upload CSV time series atau gunakan sample dari folder `sample/`.
- **Kolom CSV:** Kolom wajib `usage`; opsional `instance_code` atau `instance_id` untuk akurasi lebih baik.
- **Prediksi LSTM:** Model memakai 12 nilai terakhir (lookback) dan menampilkan prediksi nilai berikutnya.
- **Forecast Result:** Tampilan hasil prediksi dalam card.

---

## Requirement

- Python 3.9+

---

## Instalasi

1. Clone Project
  ```bash
   git clone https://github.com/riofardinan/fog-placement.git 
  ```
2. Buat dan aktifkan virtual environment (opsional tapi disarankan):
  ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   # atau: venv\Scripts\activate   # Windows
  ```
3. Install requirements
  ```bash
   pip install -r requirements.txt
  ```
4. Jalankan aplikasi:
  ```bash
   streamlit run app.py
  ```
   Buka URL yang ditampilkan di terminal (biasanya `http://localhost:8501`).

---

## Link

- **App**: [https://bytedance-ts-forecast.streamlit.app/](https://bytedance-ts-forecast.streamlit.app/)
- **Dataset:** [https://huggingface.co/datasets/ByteDance/CloudTimeSeriesData](https://huggingface.co/datasets/ByteDance/CloudTimeSeriesData)

