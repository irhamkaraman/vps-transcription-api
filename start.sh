#!/bin/bash
PORT_V1=8000
PORT_V2=8001

echo "🚀 Memulai server transkripsi..."

# Aktifkan virtual environment
source venv/bin/activate

# ============================================
# SET ENVIRONMENT VARIABLE
# ============================================

# ============================================
# JALANKAN SERVER V1
# ============================================
echo "✅ Memulai server Versi 1 (OpenAI Whisper) di port $PORT_V1..."
nohup uvicorn app:app --host 0.0.0.0 --port $PORT_V1 > server_v1.log 2>&1 &

# ============================================
# JALANKAN SERVER V2
# ============================================
echo "✅ Memulai server Versi 2 (WhisperX + RoBERTa) di port $PORT_V2..."
# Pindah ke directory v2 agar relative path model terbaca
cd whisperx-roberta-version
nohup uvicorn app_v2:app --host 0.0.0.0 --port $PORT_V2 > ../server_v2.log 2>&1 &
cd ..

echo "✅ Kedua Server berhasil berjalan di background!"
echo "📜 Untuk melihat log V1, ketik: tail -f server_v1.log"
echo "📜 Untuk melihat log V2, ketik: tail -f server_v2.log"
