#!/bin/bash
PORT=8000
echo "🔍 Menghentikan server uvicorn di port $PORT..."

# 1. Hentikan menggunakan pkill (sangat efektif untuk proses python/uvicorn)
pkill -f "uvicorn app:app" 2>/dev/null

# 2. Pastikan port dilepaskan menggunakan fuser
if command -v fuser >/dev/null 2>&1; then
    fuser -k ${PORT}/tcp >/dev/null 2>&1
fi

# 3. Fallback menggunakan lsof jika masih ada yang tersisa
if command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -ti:$PORT)
    if [ ! -z "$PID" ]; then
        kill -9 $PID 2>/dev/null
    fi
fi

echo "🛑 Server transkripsi berhasil dimatikan."