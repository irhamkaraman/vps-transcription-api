import os
import shutil
import tempfile
import json
import asyncio
import time
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# KONFIGURASI MODEL — Edit di sini jika perlu
# ============================================
MODEL_NAME = "medium"          # "large-v3" | "medium" | "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4               # PAKAI SEMUA CORE (server kamu 4 core)
BEAM_SIZE = 1                 # 1 = paling cepat, 3-5 = lebih akurat tapi lambat
VAD_FILTER = True             # SKIP DRAU, jauh lebih cepat!
WORD_TIMESTAMPS = False       # FALSE = lebih cepat (tidak perlu timestamp per kata)

# Mapping kode bahasa Laravel → kode Whisper
LANG_MAP = {
    "id": "id",   # Indonesia
    "en": "en",   # English
    "zh": "zh",   # Chinese
}

print(f"⏳ Memuat model Whisper [{MODEL_NAME}] ke memori...", flush=True)
t0 = time.time()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS
)
print(f"✅ Model [{MODEL_NAME}] siap menerima request! ({time.time()-t0:.1f}s)", flush=True)


def format_duration(ms: float) -> str:
    """Format milidetik menjadi HH:MM:SS,mmm"""
    total = ms / 1000
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def log_progress(msg: str, elapsed: float, level: str = "INFO"):
    """Logging terstandar: [LEVEL] [elapsed] pesan"""
    ts = format_duration(elapsed)
    print(f"[{level}] ⏱️ {ts} — {msg}", flush=True)


@app.post("/api/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    callback_url: str = Form(...),
    language: str = Form("auto"),   # <-- BARU: terima parameter bahasa dari Laravel
):
    print(f"\n{'='*55}", flush=True)
    print(f"📥 Menerima file: {file.filename}", flush=True)
    print(f"🔗 Callback URL: {callback_url}", flush=True)
    print(f"🌐 Bahasa target: {language}", flush=True)
    print(f"{'='*55}\n", flush=True)

    temp_file_path = None
    try:
        ext = os.path.splitext(file.filename)[1] or ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        print(f"⚙️ File disimpan sementara di: {temp_file_path}", flush=True)

        # Jalankan proses transkripsi di background
        background_tasks.add_task(
            process_transcription_background,
            temp_file_path,
            callback_url,
            language
        )

        return {"status": "queued", "message": "File diterima dan sedang diproses di background."}

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print(f"❌ Terjadi kesalahan awal: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}"
        )


def process_transcription_background(
    temp_file_path: str,
    callback_url: str,
    language: str = "auto"
):
    start_time = time.time()

    # Resolusi kode bahasa untuk Whisper
    whisper_lang = LANG_MAP.get(language, None)
    if whisper_lang is None:
        whisper_lang = None  # auto-detect jika tidak dikenali
        print(f"⚠️  Bahasa '{language}' tidak dikenali, akan auto-detect", flush=True)

    try:
        elapsed = lambda: time.time() - start_time

        print("\n" + "="*55, flush=True)
        print(f"🔊 MULAI TRANSKRIPSI", flush=True)
        print(f"   Model      : {MODEL_NAME}", flush=True)
        print(f"   Bahasa     : {language} → {whisper_lang or 'auto'}", flush=True)
        print(f"   Threads    : {CPU_THREADS}", flush=True)
        print(f"   Beam size  : {BEAM_SIZE}", flush=True)
        print(f"   VAD filter : {VAD_FILTER}", flush=True)
        print(f"   Start time : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("="*55 + "\n", flush=True)

        # === LANGKAH 1: Load audio ke memori ===
        log_progress(f"Memuat file audio ke memori...", elapsed())
        load_start = time.time()

        segments, info = model.transcribe(
            temp_file_path,
            language=whisper_lang,           # <-- Paksa bahasa (atau auto-detect)
            beam_size=BEAM_SIZE,             # <-- 1 = super cepat di CPU
            vad_filter=VAD_FILTER,           # <-- Skip bagian sunyi
            word_timestamps=WORD_TIMESTAMPS, # <-- FALSE = lebih cepat
        )

        load_time = time.time() - load_start
        elapsed_total = time.time() - start_time
        log_progress(f"Model selesai memproses audio! ({load_time:.1f}s)", elapsed_total())

        # === LANGKAH 2: Proses hasil ===
        if info:
            lang = getattr(info, "language", "unknown") or "unknown"
            prob = getattr(info, "language_probability", 0) or 0
            log_progress(f"🌍 Terdeteksi bahasa: {lang} (probabilitas: {prob:.2%})", elapsed_total())

        final_transcription = []
        segment_count = 0

        # === LANGKAH 3: Ambil segment per segment ===
        log_progress(f"Memulai ekstraksi segmen...", elapsed_total())

        for i, segment in enumerate(segments, 1):
            segment_count += 1
            seg_elapsed = time.time() - start_time

            # Escape HTML
            esc_text = segment.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            entry = {
                "text_html": esc_text,
                "speaker": "Unknown",
                "timestamp": f"{int(segment.start)//60:02d}:{int(segment.start)%60:02d} - {int(segment.end)//60:02d}:{int(segment.end)%60:02d}",
                "start_sec": round(segment.start, 2),
                "end_sec": round(segment.end, 2),
            }
            final_transcription.append(entry)

            # Log setiap segmen (SANGAT DETAIL)
            text_preview = segment.text.strip()[:60]
            if text_preview:
                log_progress(
                    f"Segmen #{i:03d}  │  {int(segment.start)//60:02d}:{int(segment.start)%60:02d} → {int(segment.end)//60:02d}:{int(segment.end)%60:02d}  │  \"{text_preview}...\"",
                    seg_elapsed()
                )
            else:
                log_progress(
                    f"Segmen #{i:03d}  │  {int(segment.start)//60:02d}:{int(segment.start)%60:02d} → {int(segment.end)//60:02d}:{int(segment.end)%60:02d}  │  (kosong/sunyi)",
                    seg_elapsed()
                )

        # === LANGKAH 4: Kirim hasil ke Laravel (Webhook) ===
        log_progress(f"Transkripsi selesai! Total {segment_count} segmen. Mengirim webhook...", elapsed_total())

        import requests
        try:
            response = requests.post(
                callback_url,
                json={"transcription": final_transcription},
                timeout=30
            )
            response.raise_for_status()
            log_progress(f"✅ Webhook BERHASIL terkirim ke Laravel! (HTTP {response.status_code})", elapsed_total())
        except Exception as http_err:
            log_progress(f"❌ Gagal mengirim webhook: {http_err}", elapsed_total(), "ERROR")

        print("\n" + "="*55, flush=True)
        print(f"🏁 TRANSKRIPSI SELESAI — Total waktu: {elapsed_total():.1f}s ({elapsed_total()/60:.1f} menit)", flush=True)
        print(f"   Total segmen: {segment_count}", flush=True)
        print(f"{'='*55}\n", flush=True)

    except Exception as e:
        elapsed_total = time.time() - start_time
        log_progress(f"❌ ERROR FATAL: {str(e)}", elapsed_total(), "ERROR")
        import requests
        try:
            requests.post(
                callback_url,
                json={"transcription": [], "error": str(e)},
                timeout=10
            )
            log_progress(f"Webhook error berhasil dikirim", elapsed_total())
        except:
            log_progress(f"Gagal kirim webhook error", elapsed_total(), "ERROR")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"🧹 File sementara berhasil dibersihkan.", flush=True)
