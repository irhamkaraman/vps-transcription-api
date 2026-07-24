import os
import shutil
import tempfile
import json
import time
import subprocess
import sys
from pathlib import Path
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
# KONFIGURASI MODEL — Baca dari environment variable
# ============================================
# Cara ganti model: set WHISPER_MODEL=medium di environment
# Model yang tersedia: tiny, base, small, medium, large, large-v2, large-v3
# Rekomendasi untuk CPU: base (cepat) atau medium (akurat tapi lebih lambat)

MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
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

# Informasi model untuk logging
MODEL_INFO = {
    "tiny":   "Tiny   (39 MB) — Paling cepat, akurasi rendah",
    "base":   "Base   (74 MB) — Cepat, akurasi sedang (REKOMENDASI)",
    "small":  "Small  (244 MB) — Sedang cepat, akurasi cukup",
    "medium": "Medium (769 MB) — Lambat di CPU, akurasi tinggi",
    "large":  "Large  (1.5 GB) — Sangat lambat di CPU, akurasi sangat tinggi",
    "large-v2": "Large-v2 (1.5 GB) — Lambat, akurasi sangat tinggi",
    "large-v3": "Large-v3 (1.5 GB) — Lambat, akurasi sangat tinggi",
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
    "logs": [],
    "processing_started_at": None,
    "transcription_started_at": None,
    "transcription_completed_at": None,
    "total_duration_sec": 0,
}


def log(msg: str):
    """Logging dengan timestamp + simpan ke progress_state"""
    ts = time.strftime("%H:%M:%S")
    log_entry = {"time": ts, "msg": msg, "timestamp": time.time()}
    progress_state["logs"].append(log_entry)
    print(f"[{ts}] {msg}", flush=True)


def send_webhook(callback_url: str, payload: dict, timeout: int = 15):
    """Kirim webhook ke Laravel dengan SSL verify=False"""
    import requests
    import urllib3
    urllib3.disable_warnings()
    try:
        response = requests.post(
            callback_url,
            json=payload,
            timeout=timeout,
            verify=False
        )
        response.raise_for_status()
        log(f"📡 Webhook terkirim: HTTP {response.status_code}")
        return True
    except Exception as e:
        log(f"⚠️ Webhook gagal: {str(e)}")
        return False


# ============================================
# PRE-PROCESS AUDIO: Convert ke WAV 16kHz Mono
# ============================================
def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Konversi audio ke WAV 16kHz mono (format optimal untuk Whisper).
    Ini alasan utama kenapa script Extracting_Dataset jauh lebih cepat.
    """
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", output_path
        ], capture_output=True, check=True, timeout=30)
        return True
    except subprocess.CalledProcessError as e:
        log(f"❌ FFmpeg convert gagal: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        log(f"❌ FFmpeg tidak ditemukan di sistem!")
        return False
    except subprocess.TimeoutExpired:
        log(f"❌ FFmpeg timeout (30s)")
        return False


# ============================================
# LOAD MODEL
# ============================================
print(f"\n{'='*50}", flush=True)
print(f"🚀 VPS TRANSCRIPTION API", flush=True)
print(f"{'='*50}", flush=True)
print(f"⏳ Memuat model Whisper [{MODEL_NAME}]: {MODEL_INFO.get(MODEL_NAME, 'Unknown')}", flush=True)
print(f"   Device: {DEVICE} | Threads: {CPU_THREADS} | Beam: {BEAM_SIZE}", flush=True)
print(f"   VAD Filter: {VAD_FILTER} | Chunk: {CHUNK_LENGTH}s", flush=True)
print(f"{'='*50}\n", flush=True)

t0 = time.time()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS
)
load_time = time.time() - t0
print(f"✅ Model [{MODEL_NAME}] siap! ({load_time:.1f}s)\n", flush=True)


# ============================================
# API ENDPOINT
# ============================================
@app.post("/api/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    callback_url: str = Form(...),
    language: str = Form("id"),
):
    log(f"\n{'='*50}")
    log(f"📥 File: {file.filename}")
    log(f"🔗 Callback: {callback_url}")
    log(f"🌐 Bahasa: {language}")
    log(f"{'='*50}")

    temp_file_path = None
    wav_file_path = None
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
# BACKGROUND PROCESSING
# ============================================
def process_transcription_background(
    temp_file_path: str,
    callback_url: str,
    language: str = "id"
):
    start_time = time.time()
    wav_file_path = None

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
    progress_state["logs"] = []
    progress_state["processing_started_at"] = time.time()
    progress_state["transcription_started_at"] = None
    progress_state["transcription_completed_at"] = None
    progress_state["total_duration_sec"] = 0

    try:
        # === Kirim progress awal: 5% - Pre-processing dimulai ===
        log(f"\n🔧 PRE-PROCESSING AUDIO")
        log(f"   Konversi ke WAV 16kHz mono...")

        progress_state["percentage"] = 5
        send_webhook(callback_url, {
            "status": "progress",
            "progress": 5,
            "message": "Pre-processing audio (konversi ke WAV)...",
            "logs": list(progress_state["logs"])
        })

        # Konversi ke WAV 16kHz mono
        wav_file_path = temp_file_path + ".wav"
        if not convert_to_wav(temp_file_path, wav_file_path):
            raise Exception("FFmpeg gagal mengkonversi audio ke WAV")

        log(f"✅ Audio dikonversi ke WAV: {os.path.getsize(wav_file_path) / 1024:.1f} KB")

        # === Kirim progress: 15% - Transkripsi dimulai ===
        log(f"\n🔊 MULAI TRANSKRIPSI")
        log(f"   Model: {MODEL_NAME}")
        log(f"   Threads: {CPU_THREADS}")
        log(f"   Beam: {BEAM_SIZE}")
        log(f"   VAD: {VAD_FILTER}")
        log(f"   Chunk: {CHUNK_LENGTH}s")
        log(f"   Mulai: {time.strftime('%H:%M:%S')}")

        progress_state["percentage"] = 15
        send_webhook(callback_url, {
            "status": "progress",
            "progress": 15,
            "message": f"Transkripsi dimulai dengan model {MODEL_NAME}...",
            "logs": list(progress_state["logs"])
        })

        # === TRANSKRIPSI ===
        progress_state["transcription_started_at"] = time.time()
        segments, info = model.transcribe(
            wav_file_path,  # Gunakan file WAV yang sudah dikonversi
            language=whisper_lang,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            word_timestamps=WORD_TIMESTAMPS,
            chunk_length=CHUNK_LENGTH,
        )
        transcription_elapsed = time.time() - progress_state["transcription_started_at"]

        # === Kirim progress: 60% - Transkripsi selesai ===
        progress_state["transcription_completed_at"] = time.time()
        progress_state["percentage"] = 60
        log(f"\n✅ Transkripsi selesai! Durasi: {transcription_elapsed:.1f}s")
        log(f"   Bahasa terdeteksi: {info.language} ({info.language_probability:.0%})")

        send_webhook(callback_url, {
            "status": "progress",
            "progress": 60,
            "message": f"Transkripsi selesai dalam {transcription_elapsed:.1f}s. Memulai parse segmen...",
            "logs": list(progress_state["logs"])
        })

        # === PROSES SEGMENTS ===
        final_transcription = []
        segment_count = 0

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

        # === Kirim progress: 80% - Segmen selesai ===
        progress_state["total_segments"] = segment_count
        progress_state["percentage"] = 80
        progress_state["transcription"] = final_transcription

        send_webhook(callback_url, {
            "status": "progress",
            "progress": 80,
            "message": f"Parse segmen selesai: {segment_count} segmen ditemukan.",
            "transcription": final_transcription,
            "logs": list(progress_state["logs"])
        })

        # === Kirim hasil final ===
        progress_state["percentage"] = 100
        log(f"\n📊 Total segmen: {segment_count}")
        log(f"📤 Mengirim hasil final ke Laravel...")

        total_elapsed = time.time() - start_time
        progress_state["total_duration_sec"] = round(total_elapsed, 1)

        response = requests.post(
            callback_url,
            json={
                "transcription": final_transcription,
                "progress": 100,
                "message": f"Transkripsi selesai! {segment_count} segmen, durasi {total_elapsed:.1f}s",
                "logs": list(progress_state["logs"]),
                "total_segments": segment_count,
                "total_duration_sec": round(total_elapsed, 1)
            },
            timeout=30,
            verify=False
        )
        response.raise_for_status()
        log(f"✅ Webhook BERHASIL! HTTP {response.status_code}")
        log(f"🏁 SELESAI! Total waktu: {total_elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"\n❌ ERROR FATAL setelah {elapsed:.1f}s: {e}")
        progress_state["error"] = str(e)
        progress_state["percentage"] = 0
        send_webhook(callback_url, {
            "transcription": [],
            "error": str(e),
            "progress": 0,
            "message": f"Error: {str(e)}",
            "logs": list(progress_state["logs"])
        })

    finally:
        progress_state["active"] = False
        # Cleanup temp files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            log(f"🧹 Temp file dibersihkan")
        if wav_file_path and os.path.exists(wav_file_path):
            os.remove(wav_file_path)
            log(f"🧹 WAV file dibersihkan")
