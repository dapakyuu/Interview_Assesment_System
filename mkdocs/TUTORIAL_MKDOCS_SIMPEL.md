# 📚 Tutorial MkDocs - Super Simpel!

## Apa itu MkDocs?

MkDocs itu tools untuk bikin website dokumentasi dari file Markdown (.md).

**Analoginya:**

- Kamu punya file Word (.docx) → Diconvert jadi website yang cantik
- File Markdown (.md) → Jadi website dokumentasi dengan menu, search, dll

---

## 🎯 Cara Pakai MkDocs (3 Langkah)

### Langkah 1: Install MkDocs

Buka **PowerShell** atau **Command Prompt** sebagai **Administrator**, jalankan:

```powershell
# Allow script execution (hanya sekali)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Masuk ke folder project
cd d:\Interview_Assesment_System-main

# Activate virtual environment
.venv\Scripts\activate

# Install MkDocs
pip install mkdocs mkdocs-material pymdown-extensions mkdocs-minify-plugin
```

### Langkah 2: Lihat Preview

Masih di terminal yang sama, jalankan:

```powershell
python -m mkdocs serve
```

**Output yang muncul:**

```
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.52 seconds
INFO    -  [12:34:56] Watching paths for changes: 'docs', 'mkdocs.yml'
INFO    -  [12:34:56] Serving on http://127.0.0.1:8000/
```

### Langkah 3: Buka Browser

Buka browser (Chrome/Firefox/Edge), ketik di address bar:

```
http://127.0.0.1:8000
```

**BOOM!** 🎉 Dokumentasi kamu sudah jalan!

---

## 📖 Apa yang Bisa Kamu Lakukan?

### 1. Edit Dokumentasi

Semua file dokumentasi ada di folder `docs/`:

```
docs/
├── index.md              ← Homepage
├── getting-started/      ← Tutorial memulai
├── features/             ← Penjelasan fitur
├── api/                  ← Dokumentasi API
└── ...
```

**Cara edit:**

1. Buka file `.md` pakai VS Code / text editor
2. Edit isinya (pakai Markdown syntax)
3. Save
4. **Browser auto-refresh!** Gak perlu restart server

### 2. Tambah Halaman Baru

**Contoh:** Mau bikin halaman "Tips & Tricks"

1. Buat file baru: `docs/tips-tricks.md`
2. Isi dengan konten:

   ```markdown
   # Tips & Tricks

   ## Tip 1: Gunakan GPU

   Untuk processing lebih cepat, gunakan GPU...
   ```

3. Daftarkan di `mkdocs.yml` bagian `nav`:
   ```yaml
   nav:
     - Home: index.md
     - Tips & Tricks: tips-tricks.md # ← Tambahkan ini
   ```

### 3. Tambah Gambar

1. Copy gambar ke folder `docs/assets/images/`
2. Referensi di markdown:
   ```markdown
   ![Screenshot Dashboard](assets/images/dashboard.png)
   ```

---

## 🎨 Fitur-Fitur Keren

### Search (Pencarian)

Ketik di search box → Langsung muncul hasil!

### Dark Mode

Klik icon moon/sun di pojok kanan atas

### Copy Code

Hover di code block → Muncul tombol "Copy"

### Navigation

Menu di kiri → Otomatis dari struktur folder `docs/`

---

## 🚀 Deploy ke Internet (GitHub Pages)

Kalau mau dokumentasi bisa diakses orang lain (gak cuma localhost):

```powershell
# Pastikan GitHub repository udah ada
python -m mkdocs gh-deploy
```

Dokumentasi akan online di:

```
https://username.github.io/Interview_Assesment_System-main/
```

---

## 🛠️ Command Penting

| Command                      | Fungsi                                 |
| ---------------------------- | -------------------------------------- |
| `python -m mkdocs serve`     | Jalankan server lokal (preview)        |
| `python -m mkdocs build`     | Build static website ke folder `site/` |
| `python -m mkdocs gh-deploy` | Deploy ke GitHub Pages                 |
| `python -m mkdocs --help`    | Lihat semua command available          |

---

## 📝 Markdown Syntax Cheat Sheet

````markdown
# Heading 1

## Heading 2

### Heading 3

**Bold text**
_Italic text_

[Link text](https://example.com)

![Image](path/to/image.png)

- Bullet point 1
- Bullet point 2

1. Numbered list
2. Item 2

`inline code`

```python
# Code block
def hello():
    print("Hello!")
```
````

```

---

## ❓ Troubleshooting

### Error: "mkdocs: command not found"

**Solusi:** Install MkDocs atau gunakan `python -m mkdocs` instead

### Error: "Port 8000 already in use"

**Solusi:**
1. Tutup terminal yang menjalankan mkdocs sebelumnya
2. Atau gunakan port lain: `python -m mkdocs serve --dev-addr=127.0.0.1:8001`

### Browser tidak auto-refresh

**Solusi:** Hard refresh browser (Ctrl+F5)

### Gambar tidak muncul

**Solusi:**
1. Check path gambar (case-sensitive!)
2. Pastikan gambar ada di folder `docs/assets/`

---

## 🎓 Resources Tambahan

- [MkDocs Official Docs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Markdown Guide](https://www.markdownguide.org/)

---

## 💡 Tips Pro

1. **Live Reload:** File di-edit → Browser auto-refresh
2. **Search:** Gunakan Ctrl+K untuk quick search
3. **Structure:** Organizer file dengan folder yang jelas
4. **Images:** Compress gambar agar loading cepat
5. **Git:** Commit perubahan dokumentasi bersama code

---

**Selamat mencoba!** 🚀

Kalau masih bingung, tanya aja! 😊
```
