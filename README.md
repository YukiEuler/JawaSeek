# JawaSeek 🔍

**JawaSeek** adalah aplikasi Cross-Lingual Information Retrieval (CLIR) berbasis Streamlit yang memungkinkan pencarian dokumen Wikipedia bahasa Jawa menggunakan query dalam bahasa Indonesia. Sistem ini mendukung berbagai metode retrieval, termasuk TF-IDF berbasis kamus, mBERT, LaBSE, dan model fine-tuned.

## Fitur

- Pencarian dokumen Jawa dengan query bahasa Indonesia
- Metode pencarian:
  - 📖 Dictionary-Based + TF-IDF
  - 🤖 mBERT (Semantic Search)
  - 🟢 LaBSE (Semantic Search)
  - 🟣 LaBSE Fine-tuned
  - 🟠 mBERT Fine-tuned
- Hasil pencarian menampilkan judul, kategori, skor, dan preview artikel
- Klik judul untuk membuka artikel asli di Wikipedia Jawa

<!-- ## Demo

![demo screenshot](demo_screenshot.png) -->

## Cara Menjalankan

1. **Clone repo ini dan install dependensi:**
    ```bash
    git clone https://github.com/username/jawaseek.git
    cd jawaseek
    pip install -r requirements.txt
    ```

2. **Pastikan file dataset (`clir_wikidata_id-jv.json`) ada di folder utama repo.**

3. **Jalankan aplikasi:**
    ```bash
    streamlit run app.py
    ```

4. **Akses aplikasi di browser:**  
   Biasanya di [http://localhost:8501](http://localhost:8501)

## Deployment

- **Streamlit Cloud:**  
  Upload seluruh repo (termasuk dataset) ke GitHub, lalu deploy via [streamlit.io/cloud](https://streamlit.io/cloud).
- **Dataset:**  
  Untuk dataset kecil-menengah, cukup letakkan di repo GitHub bersama kode.

## Model

- LaBSE Fine-tuned: [`HyacinthiaIca/LaBSE-Indonesia-Jawa-Wikipedia`](https://huggingface.co/HyacinthiaIca/LaBSE-Indonesia-Jawa-Wikipedia)
- mBERT Fine-tuned: [`HyacinthiaIca/mBERT-Indonesia-Jawa-Wikipedia`](https://huggingface.co/HyacinthiaIca/mBERT-Indonesia-Jawa-Wikipedia)

Model akan otomatis diunduh saat aplikasi dijalankan.

## Struktur Folder

```
.
├── app.py
├── clir_wikidata_id-jv.json
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- Lihat `requirements.txt` untuk dependensi lengkap

## Credits

- Dataset & Model: [HyacinthiaIca di HuggingFace](https://huggingface.co/HyacinthiaIca)

---

**JawaSeek** – Cross-Lingual Information Retrieval Indonesia ↔️ Jawa  
Powered by Streamlit, mBERT, LaBSE