import os
import shutil
import tempfile
import json
import time
import threading
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, BackgroundTasks
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
# KONFIGURASI MODEL — Optimal untuk 4-core CPU
# ============================================
MODEL_NAME = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4
BEAM_SIZE = 1
VAD_FILTER = True
WORD_TIMESTAMPS = False
CHUNK_LENGTH = 30  # proses audio per 30 detik (lebih responsif)

# Mapping bahasa Laravel → Whisper
LANG_MAP = {
    "id": "id",
    "en": "en",
    "zh": "zh",
}

# ============================================
# GLOBAL — Progress tracking thread-safe
# ============================================
progress_state = {
    "active": False,
    "percentage": 0,
    "current_segment": 0,
    "total_segments": 0,
    "callback_url": "",
    "transcription": [],
    "error": None,
}


# ============================================
# LOAD MODEL
# ============================================
print(f"⏳ Memuat model Whisper [{MODEL_NAME}] ke memori...", flush=True)
t0 = time.time()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS
)
print(f"✅ Model [{MODEL_NAME}] siap! ({time.time()-t0:.1f}s)", flush=True)


def log(msg: str):
    """Logging dengan timestamp"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================
# WEBSOCKET ALTERNATIVE: Kirim progress ke cPanel
# ============================================
def send_progress_webhook(progress_pct: int, segment_count: int, callback_url: str):
    """Kirim progress ke cPanel tanpa memperlambat transkripsi"""
    import requests
    import urllib3
    urllib3.disable_warnings()

    try:
        # Kirim progress saja, jangan tunggu response lama
        requests.post(
            callback_url,
            json={
                "status": "progress",
                "progress": progress_pct,
                "segments_processed": segment_count
            },
            timeout=10,
            verify=False
        )
        log(f"📡 Progress {progress_pct}% terkirim ke cPanel")
    except Exception as e:
        log(f"⚠️ Gagal kirim progress: {e}")


# ============================================
# PROGRESS MONITOR THREAD — Kirim update ke cPanel setiap N detik
# ============================================
def progress_monitor_thread(callback_url: str, check_interval: int = 5):
    """Thread terpisah yang memantau progress dan mengirim ke cPanel"""
    import requests
    import urllib3
    urllib3.disable_warnings()

    last_sent = 0
    milestone_sent = set()

    while progress_state["active"]:
        time.sleep(check_interval)

        current_pct = progress_state["percentage"]
        current_seg = progress_state["current_segment"]

        # Log heartbeat setiap check_interval detik
        log(f"💓 Heartbeat: {current_pct}% | Segmen: {current_seg} | Status: RUNNING")

        # Kirim milestone progress (25%, 50%, 75%) agar tidak spam
        if current_pct >= 25 and 25 not in milestone_sent:
            milestone_sent.add(25)
            send_progress_webhook(25, current_seg, callback_url)
        elif current_pct >= 50 and 50 not in milestone_sent:
            milestone_sent.add(50)
            send_progress_webhook(50, current_seg, callback_url)
        elif current_pct >= 75 and 75 not in milestone_sent:
            milestone_sent.add(75)
            send_progress_webhook(75, current_seg, callback_url)


# ============================================
# API ENDPOINT
# ============================================
@app.post("/api/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    callback_url: str = Form(...),
    language: str = Form("id"),  # default Indonesia
):
    log(f"\n{'='*50}")
    log(f"📥 File: {file.filename}")
    log(f"🔗 Callback: {callback_url}")
    log(f"🌐 Bahasa: {language}")
    log(f"{'='*50}")

    temp_file_path = None
    try:
        ext = os.path.splitext(file.filename)[1] or ".m4a"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        log(f"⚙️ Temp file: {temp_file_path}")

        background_tasks.add_task(
            process_transcription_background,
            temp_file_path,
            callback_url,
            language
        )

        return {"status": "queued", "message": "Processing started in background."}

    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        log(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# BACKGROUND PROCESSING — Inti dari perbaikan
# ============================================
def process_transcription_background(
    temp_file_path: str,
    callback_url: str,
    language: str = "id"
):
    start_time = time.time()

    whisper_lang = LANG_MAP.get(language, None)
    log(f"🌐 Bahasa target: {language} → {whisper_lang or 'auto'}")

    # Reset progress state
    progress_state["active"] = True
    progress_state["percentage"] = 0
    progress_state["current_segment"] = 0
    progress_state["total_segments"] = 0
    progress_state["callback_url"] = callback_url
    progress_state["transcription"] = []
    progress_state["error"] = None

    # Mulai thread monitor progress ke cPanel
    monitor = threading.Thread(
        target=progress_monitor_thread,
        args=(callback_url, 5),  # update setiap 5 detik
        daemon=True
    )
    monitor.start()
    log("🧵 Progress monitor thread dimulai")

    try:
        log(f"\n🔊 MULAI TRANSKRIPSI")
        log(f"   Model: {MODEL_NAME}")
        log(f"   Threads: {CPU_THREADS}")
        log(f"   Beam: {BEAM_SIZE}")
        log(f"   VAD: {VAD_FILTER}")
        log(f"   Chunk: {CHUNK_LENGTH}s")
        log(f"   Mulai: {time.strftime('%H:%M:%S')}\n")

        # === TRANSKRIPSI (tanpa progress_callback karena tidak didukung) ===
        segments, info = model.transcribe(
            temp_file_path,
            language=whisper_lang,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            word_timestamps=WORD_TIMESTAMPS,
            chunk_length=CHUNK_LENGTH,
        )

        # === PROSES SEGMENTS ===
        final_transcription = []
        segment_count = 0

        log(f"✅ Model selesai! Mengambil segmen... (bahasa: {info.language} @ {info.language_probability:.0%})")

        for i, segment in enumerate(segments, 1):
            segment_count += 1
            progress_state["current_segment"] = segment_count

            esc_text = segment.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            entry = {
                "text_html": esc_text,
                "speaker": "Unknown",
                "timestamp": f"{int(segment.start)//60:02d}:{int(segment.start)%60:02d} - {int(segment.end)//60:02d}:{int(segment.end)%60:02d}",
                "start_sec": round(segment.start, 2),
                "end_sec": round(segment.end, 2),
            }
            final_transcription.append(entry)

            # Log setiap segmen
            preview = segment.text.strip()[:80]
            log(f"   Segmen #{i:03d}: [{int(segment.start)//60:02d}:{int(segment.start)%60:02d}] {preview}")

        progress_state["total_segments"] = segment_count
        elapsed = time.time() - start_time
        log(f"\n📊 Total segmen: {segment_count}")
        log(f"⏱️  Durasi transkripsi: {elapsed:.1f}s ({elapsed/60:.1f} menit)")

        # === KIRIM HASIL KE CPANEL ===
        log(f"📤 Mengirim hasil ke cPanel...")
        progress_state["transcription"] = final_transcription
        send_progress_webhook(100, segment_count, callback_url)

        # Kirim hasil lengkap
        import requests
        import urllib3
        urllib3.disable_warnings()

        try:
            response = requests.post(
                callback_url,
                json={"transcription": final_transcription},
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            log(f"✅ Webhook BERHASIL! HTTP {response.status_code}")
        except Exception as e:
            log(f"❌ Webhook gagal: {e}")
            progress_state["error"] = str(e)

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"\n❌ ERROR FATAL setelah {elapsed:.1f}s: {e}")
        progress_state["error"] = str(e)

        # Kirim error ke cPanel
        import requests
        import urllib3
        urllib3.disable_warnings()
        try:
            requests.post(
                callback_url,
                json={"transcription": [], "error": str(e)},
                timeout=10,
                verify=False
            )
        except:
            pass

    finally:
        progress_state["active"] = False
        monitor.join(timeout=2)

        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            log(f"🧹 Temp file dibersihkan")

        elapsed_total = time.time() - start_time
        log(f"\n🏁 SELESAI! Total waktu: {elapsed_total:.1f}s")
