import os
import shutil
import tempfile
import time
import subprocess
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# KONFIGURASI MODEL & OPENAI
# ============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY tidak ditemukan di environment. API request akan gagal!", flush=True)

# Mapping bahasa Laravel → Whisper (OpenAI menggunakan format ISO-639-1)
LANG_MAP = {
    "id": "id",
    "en": "en",
    "zh": "zh",
}

# ============================================
# KONFIGURASI CALLBACK — URL Laravel cPanel untuk menerima webhook
# ============================================
# Jika VPS dan cPanel di server yang SAMA, gunakan localhost
# Jika BERBEDA server, gunakan IP/domain eksternal
CALLBACK_BASE_URL = os.getenv("CALLBACK_BASE_URL", "https://temaniskripsi.id")

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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        
        # BYPASS DNS: Jika VPS mengira temaniskripsi.id adalah localhost (127.0.0.1)
        if "temaniskripsi.id" in callback_url:
            callback_url = callback_url.replace("temaniskripsi.id", "103.180.164.146")
            headers["Host"] = "temaniskripsi.id"

        response = requests.post(
            callback_url,
            json=payload,
            timeout=timeout,
            headers=headers,
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
# ============================================
# LOAD MODEL
# ============================================
print(f"\n{'='*50}", flush=True)
print(f"🚀 VPS TRANSCRIPTION API (OPENAI WHISPER)", flush=True)
print(f"{'='*50}", flush=True)
print(f"⏳ Mode Proxy API aktif. Semua audio akan dikirim ke OpenAI.", flush=True)
print(f"{'='*50}\n", flush=True)

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
        # === Kirim progress awal: 5% - Pre-processing ===
        log(f"\n🔧 PRE-PROCESSING AUDIO")
        log(f"   Menyiapkan file audio asli untuk OpenAI...")

        progress_state["percentage"] = 5
        send_webhook(callback_url, {
            "status": "progress",
            "progress": 5,
            "message": "Menyiapkan file audio...",
            "logs": list(progress_state["logs"])
        })

        # OpenAI Whisper API mendukung m4a, mp3, wav, dll secara native!
        # Kita LANGSUNG kirim file aslinya tanpa perlu FFmpeg
        final_audio_path = temp_file_path
        file_size_mb = os.path.getsize(final_audio_path) / 1024 / 1024
        log(f"✅ Audio siap dikirim: {file_size_mb:.2f} MB")
        
        target_model = "gpt-4o-mini-transcribe-2025-12-15"

        # === Kirim progress: 15% - Transkripsi dimulai ===
        log(f"\n🔊 MULAI TRANSKRIPSI VIA OPENAI")
        log(f"   Model: {target_model}")
        log(f"   Mulai: {time.strftime('%H:%M:%S')}")
        log(f"   Mengirim file berukuran {file_size_mb:.2f} MB ke OpenAI...")

        progress_state["percentage"] = 15
        send_webhook(callback_url, {
            "status": "progress",
            "progress": 30,
            "message": f"Mengirim audio ke OpenAI Whisper...",
            "logs": list(progress_state["logs"])
        })

        # === TRANSKRIPSI OPENAI ===
        progress_state["transcription_started_at"] = time.time()
        
        # Kirim ke OpenAI
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        files = {
            "file": (os.path.basename(final_audio_path), open(final_audio_path, "rb"), "audio/m4a")
        }
        lang_name = "Indonesia" if whisper_lang == "id" else ("Inggris" if whisper_lang == "en" else whisper_lang)
        data = {
            "model": target_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
            "temperature": "0.1",
            "prompt": f"Berikut adalah transkripsi rekaman percakapan dan bimbingan dalam bahasa {lang_name}."
        }
        if whisper_lang:
            data["language"] = whisper_lang

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log(f"❌ OpenAI API Error: {e.response.text}")
            raise e
        
        result = response.json()
        segments = result.get("segments", [])
        transcription_elapsed = time.time() - progress_state["transcription_started_at"]

        # === Kirim progress: 60% - Transkripsi selesai ===
        progress_state["transcription_completed_at"] = time.time()
        progress_state["percentage"] = 60
        log(f"\n✅ Transkripsi selesai dari OpenAI! Durasi API: {transcription_elapsed:.1f}s")

        send_webhook(callback_url, {
            "status": "progress",
            "progress": 60,
            "message": f"Transkripsi selesai dalam {transcription_elapsed:.1f}s. Memulai formatting segmen...",
            "logs": list(progress_state["logs"])
        })

        # === PROSES SEGMENTS ===
        final_transcription = []
        import re
        previous_text = ""
        segment_count = 0

        for segment in segments:
            esc_text = segment.get("text", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # --- FILTER HALUSINASI (POST-PROCESSOR) ---
            text_lower = esc_text.lower().strip()
            
            # 1. Filter kata template YouTube / Halusinasi
            if "subscribe" in text_lower or "like" in text_lower or "komen" in text_lower or "share" in text_lower or "terima kasih" in text_lower or "selamat menikmati" in text_lower:
                continue 

            previous_text = esc_text
            # ----------------------------------------------------

            segment_count += 1
            progress_state["current_segment"] = segment_count
            start_sec = segment.get("start", 0)
            end_sec = segment.get("end", 0)
            
            entry = {
                "text_html": esc_text,
                "speaker": "Unknown",
                "timestamp": f"{int(start_sec)//60:02d}:{int(start_sec)%60:02d} - {int(end_sec)//60:02d}:{int(end_sec)%60:02d}",
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
            }
            final_transcription.append(entry)

            # Log setiap segmen
            preview = esc_text.strip()[:80]
            log(f"   Segmen #{segment_count:03d}: [{int(start_sec)//60:02d}:{int(start_sec)%60:02d}] {preview}")

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

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        
        if "temaniskripsi.id" in callback_url:
            callback_url = callback_url.replace("temaniskripsi.id", "103.180.164.146")
            headers["Host"] = "temaniskripsi.id"
            
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
            timeout=120,
            headers=headers,
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
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
