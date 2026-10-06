#!/bin/bash
PORT=8000
echo "🔍 Mencari proses di port $PORT..."

# Cari PID yang menggunakan port 8000 dan matikan
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    lsof -ti:$PORT | xargs kill -9
    echo "🛑 Server di port $PORT berhasil dimatikan."
else
    echo "⚠️ Tidak ada server yang sedang berjalan di port $PORT."
fi