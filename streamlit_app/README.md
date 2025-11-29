# Streamlit Inference App

Aplikasi Streamlit ringan untuk menjalankan inferensi Retail Speech Understanding menggunakan checkpoint Stage-2 manual (`cnn_bilstm_stage2_manual-colab.pt`).

## Struktur Folder

```
streamlit_app/
├── app.py                # UI dan alur inferensi Streamlit
├── audio_utils.py        # Utilitas pemrosesan audio (log-Mel, normalisasi, penyimpanan)
├── model_utils.py        # Definisi model CNN-BiLSTM dan pemuatan label map
├── requirements.txt      # Dependensi minimal untuk aplikasi
└── README.md             # Berkas ini
```

Pastikan berkas berikut tersedia di root proyek (satu level di atas folder `streamlit_app`):

- `cnn_bilstm_stage2_manual-colab.pt`
- `label_maps_google_tts.json`

## Menjalankan Aplikasi

1. **Aktifkan environment** (opsional) dan instal dependensi:
   ```powershell
   cd "M:\ITERA\Semester 7\Deep Learning\Tubes"
   pip install -r streamlit_app/requirements.txt
   ```

2. **Jalankan aplikasi**:
   ```powershell
   streamlit run streamlit_app/app.py
   ```

3. **Fitur utama**:
   - **Upload Audio**: unggah file audio (wav/mp3/flac/m4a) berdurasi 0.5–5 detik.
   - **Rekam Langsung**: gunakan modul `streamlit-audiorec` untuk merekam lewat browser.
   - **Simpan Audio**: centang opsi untuk menyimpan salinan audio ke folder `streamlit_uploads/`.
   - Prediksi berisi intent, product ID, quantity, serta logits lengkap untuk debugging.

## Catatan

- Aplikasi otomatis berjalan di CPU maupun GPU jika tersedia (CUDA).
- Modul `streamlit-audiorec` perlu diinstal agar tab rekam bekerja.
- Jika ingin menampilkan probabilitas bukannya logits, ubah fungsi `render_prediction` pada `app.py` dengan menerapkan `scipy.special.softmax`.
- Rentang durasi audio mengikuti konfigurasi training (0.5–5 detik). File di luar rentang ini akan ditolak secara otomatis.
