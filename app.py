import os
import shutil
import tempfile
import time
import subprocess
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
def log(msg: str):
    """Logging dengan timestamp (Global)"""
    ts = time.strftime("%H:%M:%S")
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
print(f"\n{'='*50}", flush=True)
print(f"🚀 VPS TRANSCRIPTION API (OPENAI WHISPER + INDOBERT)", flush=True)
print(f"{'='*50}", flush=True)
print(f"⏳ Mode Proxy API aktif. Semua audio akan dikirim ke OpenAI.", flush=True)
print(f"{'='*50}\n", flush=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
print(f"Loading IndoBERT dari {MODEL_DIR}...", flush=True)
try:
    indobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    indobert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    indobert_model.eval()
    
    indobert_labels_list = list(indobert_model.config.id2label.values())
    advice_labels_ref = ['arahan_eksplisit', 'bimbingan_bertahap', 'dukungan_keputusan', 'jawaban_tegas', 'petunjuk_kontekstual']
    modes_labels_ref = ['otoritas', 'power_gaining', 'power_maintaining', 'power_over']

    advice_indices = [i for i, label in enumerate(indobert_labels_list) if label in advice_labels_ref]
    modes_indices = [i for i, label in enumerate(indobert_labels_list) if label in modes_labels_ref]
    print(f"✅ IndoBERT berhasil dimuat! Label tersedia: {len(indobert_labels_list)}", flush=True)
except Exception as e:
    print(f"⚠️ WARNING: Gagal meload model IndoBERT: {e}", flush=True)
    indobert_tokenizer = None
    indobert_model = None
    indobert_labels_list = []
    advice_indices = []
    modes_indices = []

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
    state = {
        "percentage": 0,
        "logs": [],
        "transcription_started_at": None,
        "error": None
    }
    
    def job_log(msg: str):
        ts = time.strftime("%H:%M:%S")
        log_entry = {"time": ts, "msg": msg, "timestamp": time.time()}
        state["logs"].append(log_entry)
        print(f"[{ts}] {msg}", flush=True)

    start_time = time.time()
    wav_file_path = None

    whisper_lang = LANG_MAP.get(language, None)
    job_log(f"🌐 Bahasa target: {language} → {whisper_lang or 'auto'}")

    # Reset progress state
    state["active"] = True
    state["percentage"] = 0
    state["current_segment"] = 0
    state["total_segments"] = 0
    state["callback_url"] = callback_url
    state["transcription"] = []
    state["error"] = None
    state["logs"] = []
    state["processing_started_at"] = time.time()
    state["transcription_started_at"] = None
    state["transcription_completed_at"] = None
    state["total_duration_sec"] = 0

    try:
        # === Kirim progress awal: 5% - Pre-processing ===
        job_log(f"\n🔧 PRE-PROCESSING AUDIO")
        job_log(f"   Menyiapkan file audio asli untuk OpenAI...")

        state["percentage"] = 5
        send_webhook(callback_url, {
            "status": "progress",
            "progress": 5,
            "message": "Menyiapkan file audio...",
            "logs": list(state["logs"])
        })

        # OpenAI Whisper API mendukung flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
        ext = os.path.splitext(temp_file_path)[1].lower()
        supported_exts = [".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"]
        
        final_audio_path = temp_file_path
        if ext not in supported_exts:
            job_log(f"⚠️ Format {ext} tidak didukung secara native, mengonversi ke .wav dengan FFmpeg...")
            wav_file_path = tempfile.mktemp(suffix=".wav")
            if ffmpeg_convert_to_wav(temp_file_path, wav_file_path):
                final_audio_path = wav_file_path
                job_log("✅ Konversi ke .wav berhasil")
            else:
                job_log("❌ Konversi gagal, mencoba mengirim aslinya...")
                
        file_size_mb = os.path.getsize(final_audio_path) / 1024 / 1024
        job_log(f"✅ Audio siap dikirim: {file_size_mb:.2f} MB")
        
        target_model = "whisper-1"

        # === Kirim progress: 15% - Transkripsi dimulai ===
        job_log(f"\n🔊 MULAI TRANSKRIPSI VIA OPENAI")
        job_log(f"   Model: {target_model}")
        job_log(f"   Mulai: {time.strftime('%H:%M:%S')}")
        job_log(f"   Mengirim file berukuran {file_size_mb:.2f} MB ke OpenAI...")

        state["percentage"] = 15
        send_webhook(callback_url, {
            "status": "progress",
            "progress": 30,
            "message": f"Mengirim audio ke OpenAI Whisper...",
            "logs": list(state["logs"])
        })

        # === TRANSKRIPSI OPENAI ===
        state["transcription_started_at"] = time.time()
        
        # Kirim ke OpenAI
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        files = {
            "file": (os.path.basename(final_audio_path), open(final_audio_path, "rb"))
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
            job_log(f"❌ OpenAI API Error: {e.response.text}")
            raise e
        
        result = response.json()
        segments = result.get("segments", [])
        transcription_elapsed = time.time() - state["transcription_started_at"]

        # === Kirim progress: 60% - Transkripsi selesai ===
        state["transcription_completed_at"] = time.time()
        state["percentage"] = 60
        job_log(f"\n✅ Transkripsi selesai dari OpenAI! Durasi API: {transcription_elapsed:.1f}s")

        send_webhook(callback_url, {
            "status": "progress",
            "progress": 60,
            "message": f"Transkripsi selesai dalam {transcription_elapsed:.1f}s. Memulai formatting segmen...",
            "logs": list(state["logs"])
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
            state["current_segment"] = segment_count
            start_sec = segment.get("start", 0)
            end_sec = segment.get("end", 0)
            
            # --- INDOBERT ANALYSIS ---
            advice_giving = ""
            modes_of_interaction = ""
            if indobert_tokenizer and indobert_model and esc_text.strip():
                try:
                    inputs = indobert_tokenizer([esc_text], padding=True, truncation=True, max_length=128, return_tensors="pt")
                    with torch.no_grad():
                        outputs = indobert_model(**inputs)
                        probs = torch.sigmoid(outputs.logits).numpy()[0]
                        
                        # Advice Giving
                        advice_giving_labels = []
                        advice_probs = probs[advice_indices]
                        if len(advice_probs) > 0:
                            if np.max(advice_probs) > 0.5:
                                for idx in advice_indices:
                                    if probs[idx] > 0.5:
                                        advice_giving_labels.append(indobert_labels_list[idx])
                            else:
                                advice_giving_labels.append(indobert_labels_list[advice_indices[np.argmax(advice_probs)]])
                            advice_giving = ", ".join(advice_giving_labels)
                            
                        # Modes of Interaction
                        modes_labels_list = []
                        modes_probs = probs[modes_indices]
                        if len(modes_probs) > 0:
                            if np.max(modes_probs) > 0.5:
                                for idx in modes_indices:
                                    if probs[idx] > 0.5:
                                        modes_labels_list.append(indobert_labels_list[idx])
                            else:
                                modes_labels_list.append(indobert_labels_list[modes_indices[np.argmax(modes_probs)]])
                            modes_of_interaction = ", ".join(modes_labels_list)
                except Exception as e:
                    job_log(f"⚠️ IndoBERT error on segment {segment_count}: {e}")

            entry = {
                "text_html": esc_text,
                "speaker": "Unknown",
                "timestamp": f"{int(start_sec)//60:02d}:{int(start_sec)%60:02d} - {int(end_sec)//60:02d}:{int(end_sec)%60:02d}",
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "advice_giving": advice_giving,
                "modes_of_interaction": modes_of_interaction,
            }
            final_transcription.append(entry)

            # Log setiap segmen
            preview = esc_text.strip()[:80]
            job_log(f"   Segmen #{segment_count:03d}: [{int(start_sec)//60:02d}:{int(start_sec)%60:02d}] {preview} | Advice: {advice_giving} | Modes: {modes_of_interaction}")

        # === Kirim progress: 80% - Segmen selesai ===
        state["total_segments"] = segment_count
        state["percentage"] = 80
        state["transcription"] = final_transcription

        send_webhook(callback_url, {
            "status": "progress",
            "progress": 80,
            "message": f"Parse segmen selesai: {segment_count} segmen ditemukan.",
            "transcription": final_transcription,
            "logs": list(state["logs"])
        })

        # === Kirim hasil final ===
        state["percentage"] = 100
        job_log(f"\n📊 Total segmen: {segment_count}")
        job_log(f"📤 Mengirim hasil final ke Laravel...")

        total_elapsed = time.time() - start_time
        state["total_duration_sec"] = round(total_elapsed, 1)

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
                "logs": list(state["logs"]),
                "total_segments": segment_count,
                "total_duration_sec": round(total_elapsed, 1)
            },
            timeout=120,
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        job_log(f"✅ Webhook BERHASIL! HTTP {response.status_code}")
        job_log(f"🏁 SELESAI! Total waktu: {total_elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        job_log(f"\n❌ ERROR FATAL setelah {elapsed:.1f}s: {e}")
        state["error"] = str(e)
        state["percentage"] = 0
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        send_webhook(callback_url, {
            "transcription": [],
            "error": str(e),
            "progress": 0,
            "message": f"Error: {str(e)}",
            "logs": list(state["logs"])
        })

    finally:
        state["active"] = False
        # Cleanup temp files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            job_log(f"🧹 Temp file dibersihkan")
