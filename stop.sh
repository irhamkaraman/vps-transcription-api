#!/bin/bash
PORT_V1=8000
PORT_V2=8001

echo "🔍 Mencari proses server yang sedang berjalan..."

# Matikan V1
if lsof -Pi :$PORT_V1 -sTCP:LISTEN -t >/dev/null ; then
    lsof -ti:$PORT_V1 | xargs kill -9
    echo "🛑 Server V1 di port $PORT_V1 berhasil dimatikan."
else
    echo "⚠️ Tidak ada server V1 yang berjalan di port $PORT_V1."
fi

# Matikan V2
if lsof -Pi :$PORT_V2 -sTCP:LISTEN -t >/dev/null ; then
    lsof -ti:$PORT_V2 | xargs kill -9
    echo "🛑 Server V2 di port $PORT_V2 berhasil dimatikan."
else
    echo "⚠️ Tidak ada server V2 yang berjalan di port $PORT_V2."
fi
