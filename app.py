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
            if convert_to_wav(temp_file_path, wav_file_path):
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


# ============================================
# V2 ENDPOINTS (OpenAI Diarize + RoBERTa MultiTask)
# ============================================
import asyncio
import contextlib
from collections import Counter

import torch.nn as nn
from fastapi import WebSocket, WebSocketDisconnect
from transformers import AutoConfig, AutoModel

V2_DIARIZE_MODEL = os.getenv("V2_DIARIZE_MODEL", "gpt-4o-transcribe-diarize")
V2_BASE_MODEL = "cahya/roberta-base-indonesian-522M"
V2_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "whisperx-roberta-version", "models", "multi_task_roberta_v2",
)
# URUTAN HARUS SAMA DENGAN SAAT TRAINING (LabelEncoder = alfabetis).
# Sumber: notebook Train_Model_RoBERTa. Jumlah kelas sudah diverifikasi dari .pt (6 advice, 3 modes).
V2_ADVICE_CLASSES = [
    "arahan_eksplisit", "bimbingan_bertahap", "dukungan_keputusan",
    "jawaban_tegas", "otoritas", "petunjuk_kontekstual",
]
V2_MODES_CLASSES = ["power_gaining", "power_maintaining", "power_over"]
# "longest" = pembicara dengan durasi bicara terpanjang dianggap Dosen; "first" = yang pertama bicara
V2_SPEAKER_RULE = os.getenv("V2_SPEAKER_RULE", "longest").lower()

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


class MultiTaskGraphRoBERTa(nn.Module):
    """Arsitektur identik dengan notebook training."""

    def __init__(self, num_advice_classes=6, num_modes_classes=3, model_name=V2_BASE_MODEL):
        super().__init__()
        # Bobot asli ada di file .pt, jadi cukup bangun dari config (tanpa unduh bobot 500MB)
        self.roberta = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden_dim = self.roberta.config.hidden_size
        self.classifier_advice = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_advice_classes)
        )
        self.classifier_modes = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_modes_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier_advice(pooled_output), self.classifier_modes(pooled_output)


v2_tokenizer = None
v2_model = None
try:
    print(f"Loading RoBERTa V2 dari {V2_MODEL_DIR}...", flush=True)
    v2_tokenizer = AutoTokenizer.from_pretrained(V2_MODEL_DIR)
    _m = MultiTaskGraphRoBERTa(len(V2_ADVICE_CLASSES), len(V2_MODES_CLASSES))
    _sd = torch.load(os.path.join(V2_MODEL_DIR, "multi_task_roberta_model.pt"), map_location="cpu")
    _res = _m.load_state_dict(_sd, strict=False)
    _missing = [k for k in _res.missing_keys if "position_ids" not in k]
    if _missing:
        raise RuntimeError(f"State dict tidak cocok, key hilang: {_missing[:5]}")
    _m.eval()
    v2_model = _m
    print("✅ RoBERTa V2 berhasil dimuat!", flush=True)
except Exception as e:
    print(f"⚠️ WARNING: Gagal meload RoBERTa V2: {e}", flush=True)
    v2_tokenizer = None
    v2_model = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        # State terakhir per job, dikirim ulang saat browser (re)connect
        self.last_state: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, slug: str):
        await websocket.accept()
        self.active_connections[slug] = websocket
        if slug in self.last_state:
            try:
                await websocket.send_json(self.last_state[slug])
            except Exception:
                self.disconnect(slug, websocket)

    def disconnect(self, slug: str, websocket: WebSocket = None):
        # Jangan hapus koneksi baru jika yang menutup adalah koneksi lama
        if websocket is not None and self.active_connections.get(slug) is not websocket:
            return
        self.active_connections.pop(slug, None)

    async def send_progress(self, slug: str, message: dict):
        self.last_state[slug] = message
        while len(self.last_state) > 200:  # batasi memori
            self.last_state.pop(next(iter(self.last_state)))
        ws = self.active_connections.get(slug)
        if ws:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(slug, ws)


manager = ConnectionManager()
v2_semaphore = asyncio.Semaphore(2)  # batasi job V2 paralel agar RAM/CPU aman


@app.websocket("/api/v2/ws/{slug}")
async def websocket_endpoint(websocket: WebSocket, slug: str):
    await manager.connect(websocket, slug)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect(slug, websocket)


# ---------- Tahap 1: transkripsi + diarization (OpenAI) ----------
def v2_diarize(audio_path: str, language: str) -> list:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    lang = LANG_MAP.get(language)

    def _call(with_lang: bool):
        data = {
            "model": V2_DIARIZE_MODEL,
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }
        if with_lang and lang:
            data["language"] = lang
        with open(audio_path, "rb") as f:
            return requests.post(
                url, headers=headers,
                files={"file": (os.path.basename(audio_path), f)},
                data=data, timeout=900,
            )

    resp = _call(True)
    if resp.status_code == 400 and lang:
        log(f"⚠️ OpenAI 400 dengan parameter language, mengulang tanpa language: {resp.text[:200]}")
        resp = _call(False)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")

    segments = []
    for s in resp.json().get("segments", []):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "speaker": s.get("speaker") or "UNKNOWN",
            "text": text,
            "start": float(s.get("start") or 0),
            "end": float(s.get("end") or 0),
        })
    return segments


# ---------- Tahap 2: pemetaan speaker -> peran ----------
def v2_map_speakers(segments: list) -> list:
    durations = Counter()
    order = []
    for s in segments:
        durations[s["speaker"]] += max(0.0, s["end"] - s["start"])
        if s["speaker"] not in order:
            order.append(s["speaker"])
    ranked = order if V2_SPEAKER_RULE == "first" else [sp for sp, _ in durations.most_common()]
    names = {}
    for i, sp in enumerate(ranked):
        names[sp] = "Dosen" if i == 0 else ("Mahasiswa" if i == 1 else f"Pembicara {i + 1}")
    log(f"👥 Pemetaan pembicara ({V2_SPEAKER_RULE}): {names}")
    for s in segments:
        s["speaker"] = names[s["speaker"]]
    return segments


# ---------- Tahap 3: klasifikasi RoBERTa ----------
def v2_classify(texts: list) -> list:
    results = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = v2_tokenizer(batch, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        with torch.no_grad():
            logits_adv, logits_mod = v2_model(enc["input_ids"], enc["attention_mask"])
        adv_idx = torch.argmax(logits_adv, dim=1).tolist()
        mod_idx = torch.argmax(logits_mod, dim=1).tolist()
        for a, m in zip(adv_idx, mod_idx):
            results.append((V2_ADVICE_CLASSES[a], V2_MODES_CLASSES[m]))
    return results


# ---------- Tahap 4: graph ----------
def v2_build_graph(items: list) -> dict:
    def speaker_id(name: str) -> str:
        return "Mhs" if name == "Mahasiswa" else name

    nodes = {}
    edge_counts = Counter()

    def add_node(node_id, label, group):
        nodes.setdefault(node_id, {"id": node_id, "label": label, "group": group})

    prev = None
    for it in items:
        sid = speaker_id(it["speaker"])
        add_node(sid, it["speaker"], "speaker")

        aid = f"advice:{it['advice_giving']}"
        add_node(aid, it["advice_giving"].replace("_", " "), "advice")
        edge_counts[(sid, aid)] += 1

        mid = f"mode:{it['modes_of_interaction']}"
        add_node(mid, it["modes_of_interaction"].replace("_", " "), "mode")
        edge_counts[(sid, mid)] += 1

        if prev is not None and prev != sid:  # pergantian giliran bicara
            edge_counts[(prev, sid)] += 1
        prev = sid

    edges = [{"from": a, "to": b, "label": f"{n}x"} for (a, b), n in edge_counts.items()]
    return {"nodes": list(nodes.values()), "edges": edges}


async def _creep_progress(slug: str, start: int, end: int, message: str, step: int = 2, interval: float = 4.0):
    """Naikkan progress pelan-pelan selama tahap panjang (OpenAI) supaya UI tidak terlihat macet."""
    p = start
    while p < end:
        await asyncio.sleep(interval)
        p = min(end, p + step)
        await manager.send_progress(slug, {"progress": p, "message": message})


async def process_audio_pipeline(slug: str, callback_url: str, audio_path: str, language: str):
    creep = None
    try:
        async with v2_semaphore:
            await manager.send_progress(slug, {"progress": 5, "message": "Audio diterima, memulai proses..."})
            if v2_model is None or v2_tokenizer is None:
                raise RuntimeError("Model RoBERTa V2 tidak termuat di server.")
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY belum dikonfigurasi di server.")

            # 1) Transkripsi + diarization
            await manager.send_progress(slug, {"progress": 15, "message": "Transkripsi & identifikasi pembicara..."})
            creep = asyncio.create_task(_creep_progress(slug, 15, 55, "Transkripsi & identifikasi pembicara..."))
            t0 = time.time()
            segments = await asyncio.to_thread(v2_diarize, audio_path, language)
            creep.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await creep
            creep = None
            log(f"✅ [{slug}] Diarization selesai: {len(segments)} segmen ({time.time() - t0:.1f}s)")
            if not segments:
                raise RuntimeError("Tidak ada ucapan yang terdeteksi pada audio.")

            # 2) Peran pembicara
            await manager.send_progress(slug, {"progress": 60, "message": "Memetakan peran pembicara..."})
            segments = v2_map_speakers(segments)

            # 3) Klasifikasi RoBERTa
            await manager.send_progress(slug, {"progress": 65, "message": "Klasifikasi segmen dengan RoBERTa Multi-Task..."})
            t1 = time.time()
            preds = await asyncio.to_thread(v2_classify, [s["text"] for s in segments])
            log(f"✅ [{slug}] Klasifikasi selesai ({time.time() - t1:.1f}s)")

            transcription = []
            for s, (adv, mod) in zip(segments, preds):
                transcription.append({
                    "speaker": s["speaker"],
                    "text": s["text"],
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2),
                    "advice_giving": adv,
                    "modes_of_interaction": mod,
                })

            # 4) Graph
            await manager.send_progress(slug, {"progress": 85, "message": "Membangun Graph Nodes & Edges..."})
            graph_data = v2_build_graph(transcription)

            # 5) Kirim ke Laravel (retry), baru beri sinyal selesai
            await manager.send_progress(slug, {"progress": 95, "message": "Mengirim hasil ke server..."})
            payload = {"transcription": transcription, "graph_data": graph_data}
            sent = False
            for attempt in range(3):
                sent = await asyncio.to_thread(send_webhook, callback_url, payload, 60)
                if sent:
                    break
                await asyncio.sleep(2 * (attempt + 1))
            if not sent:
                raise RuntimeError("Gagal mengirim hasil ke server Laravel (webhook).")

            await manager.send_progress(slug, {"progress": 100, "message": "Selesai!"})
            await manager.send_progress(slug, {"progress": 100, "status": "completed"})
    except Exception as e:
        log(f"❌ [{slug}] Pipeline V2 gagal: {e}")
        await manager.send_progress(slug, {"status": "failed", "message": f"Gagal memproses audio: {e}"})
    finally:
        if creep is not None:
            creep.cancel()
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.post("/api/v2/transcribe")
async def transcribe_v2(
    background_tasks: BackgroundTasks,
    slug: str = Form(...),
    callback_url: str = Form(...),
    language: str = Form("id"),
    file: UploadFile = File(...),
):
    ext = os.path.splitext(file.filename or "")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name
    log(f"📥 V2 diterima: slug={slug} file={file.filename} bahasa={language}")
    await manager.send_progress(slug, {"progress": 2, "message": "Audio diterima server AI..."})
    background_tasks.add_task(process_audio_pipeline, slug, callback_url, audio_path, language)
    return {"status": "processing_started"}
