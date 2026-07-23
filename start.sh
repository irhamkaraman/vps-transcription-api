#!/bin/bash
PORT=80
echo "🚀 Memulai server transkripsi di port $PORT..."

# Aktifkan virtual environment
source venv/bin/activate

# Jalankan uvicorn di background menggunakan nohup
nohup uvicorn app:app --host 0.0.0.0 --port $PORT > server.log 2>&1 &

echo "✅ Server berhasil berjalan di background!"
echo "📜 Untuk melihat log real-time, ketik: tail -f server.log"