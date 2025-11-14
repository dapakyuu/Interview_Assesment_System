<<<<<<< daffa
# AI Interview Assessment System — Instruksi Proyek

Prototype untuk mengunggah video interview dan mengirim payload JSON ke backend FastAPI.

## Ringkasan singkat

- Frontend: Upload.html (ditopang Upload.css, Upload.js)
- Backend: FastAPI (payload_video.ipynb)
- Penyimpanan lokal: `uploads/` (file video) dan `received_payloads/` (payload JSON)

## Prasyarat

- Python 3.8+
- pip
- (opsional) ngrok untuk membuat public URL

## Setup cepat (lokal)

1. Clone atau taruh project di mesin lokal:
   ```
   d:\Coding\Interview_Assesment_System-main\
   ```
2. Buat virtualenv dan aktifkan:
   - Windows:
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS / Linux:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependensi:
   ```
   pip install fastapi uvicorn python-multipart nest-asyncio pyngrok
   ```

## Menjalankan server

- Dari notebook: buka `payload_video.ipynb` dan jalankan cell yang menjalankan uvicorn (port default 8888).
- Atau jalankan manual:
  ```
  uvicorn payload_video:app --host 0.0.0.0 --port 8888
  ```
  Server menyajikan:
- Static uploads: `http://127.0.0.1:8888/uploads/<filename>`
- Form uji: `http://127.0.0.1:8888/upload_form`

## Menggunakan frontend (Upload.html)

- Buka `Upload.html` di browser. Untuk fitur fetch ke API, lebih baik sajikan file via local static server yang sama origin, atau jalankan fronted dari server yang sama untuk menghindari isu CORS.
- Alur:
  1. Pilih/seret video → client otomatis upload ke `/upload_file`.
  2. Setelah file tersimpan, client membangun payload JSON (mengandung URL file) lalu POST ke `/upload`.
  3. Jika server mengembalikan properti `redirect`, client akan mengarahkan browser ke halaman tersebut (pastikan file dashboard tersedia dan ejaan sesuai, mis. `halaman_dashboard.html` atau `halaman_dasboard.html` sesuai konfigurasi server).

## Endpoint penting

- POST /upload_file  
  multipart field: `file`  
  Response: { success, url, name }

- POST /upload  
  body: JSON payload  
  Response: { success, saved_as, url, redirect? }

- DELETE /delete_file  
  body: { "name": "<safe_name>" }

- GET /last_payload  
  metadata payload terakhir

- GET /last_payload/content  
  isi payload terakhir (JSON)

## Troubleshooting singkat

- Redirect tidak terjadi:
  - Periksa response body JSON untuk properti `redirect`.
  - Periksa header `Location` bila server menggunakan header.
  - Pastikan path redirect benar dan file dashboard tersedia di server.
- CORS: server notebook sudah mengizinkan origin wildcard untuk development. Kencangkan sebelum produksi.
- Jika file tidak muncul di `/uploads/`, cek folder `uploads/` dan permission.
- Debug: buka DevTools → Network untuk memeriksa request/response; Console untuk error JS.

## Catatan deploy

- Ganti DEFAULT_BASE_URL di client menjadi domain produksi.
- Untuk file besar, pertimbangkan penyimpanan cloud (S3/GCS) dan pre-signed URL.
- Amankan endpoint dengan autentikasi dan validasi input pada produksi.

## Lisensi

MIT — gunakan dan modifikasi sesuai kebutuhan.

---

Selamat mengembangkan — jika butuh bantuan menyesuaikan redirect atau deployment, beri tahu detail environment Anda.
=======
# Interview Assesment System
Capstone Project Asah by Dicoding 2025
>>>>>>> main
