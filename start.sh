#!/bin/bash
PORT=8000

echo "🚀 Memulai server transkripsi di port $PORT..."

# Aktifkan virtual environment
source venv/bin/activate

# ============================================
# DOWNLOAD MODEL (jika belum ada)
# ============================================
# Ganti model di sini jika perlu: tiny, base, small, medium, large-v3
MODEL="base"

if [ ! -d "models/$MODEL" ]; then
    echo "⏳ Download model Whisper [$MODEL]..."
    python download_model.py $MODEL
fi

# ============================================
# SET ENVIRONMENT VARIABLE
# ============================================
export WHISPER_MODEL=$MODEL

# ============================================
# JALANKAN SERVER
# ============================================
echo "✅ Model siap! Memulai server..."
nohup uvicorn app:app --host 0.0.0.0 --port $PORT > server.log 2>&1 &

echo "✅ Server berhasil berjalan di background!"
echo "📜 Untuk melihat log real-time, ketik: tail -f server.log"
echo "📋 Untuk mengganti model, edit baris MODEL= di file ini, lalu restart"
