import os
import shutil
import tempfile
import json
import asyncio
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

print("⏳ Memuat model Whisper ke memori...", flush=True)
# Menggunakan compute_type int8 dan model 'small' atau 'medium' agar cepat di CPU 4-Core
model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=4)
print("✅ Model siap menerima request!", flush=True)


@app.post("/api/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    callback_url: str = Form(...)
):
  print(f"📥 Menerima file: {file.filename}", flush=True)
  print(f"🔗 Callback URL: {callback_url}", flush=True)

  temp_file_path = None
  try:
    ext = os.path.splitext(file.filename)[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
      shutil.copyfileobj(file.file, temp_file)
      temp_file_path = temp_file.name

    print(f"⚙️ File disimpan sementara di: {temp_file_path}", flush=True)
    
    # Jalankan proses transkripsi di background
    background_tasks.add_task(process_transcription_background, temp_file_path, callback_url)

    return {"status": "queued", "message": "File diterima dan sedang diproses di background."}

  except Exception as e:
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    print(f"❌ Terjadi kesalahan awal: {str(e)}", flush=True)
    raise HTTPException(
        status_code=500, detail=f"Internal Server Error: {str(e)}"
    )

def process_transcription_background(temp_file_path: str, callback_url: str):
    import requests
    import time
    try:
        print("⏳ Mulai proses pembedahan audio oleh Whisper di background...", flush=True)
        segments, info = model.transcribe(temp_file_path, beam_size=5)
        print(f"🌍 Terdeteksi bahasa: {info.language} dengan probabilitas {info.language_probability}", flush=True)
        
        final_transcription = []
        last_webhook_time = 0
        
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}", flush=True)
            
            # Format sesuai kebutuhan database Laravel
            start_min = int(segment.start // 60)
            start_sec = int(segment.start % 60)
            end_min = int(segment.end // 60)
            end_sec = int(segment.end % 60)
            
            # Escape HTML simple
            esc_text = segment.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            final_transcription.append({
                "text_html": esc_text,
                "speaker": "Unknown",
                "timestamp": f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
            })
            
            # Throttling webhook: kirim webhook maksimal tiap 2 detik
            current_time = time.time()
            if current_time - last_webhook_time > 2.0:
                try:
                    requests.post(callback_url, json={"status": "progress", "transcription": final_transcription}, timeout=5)
                    last_webhook_time = current_time
                except Exception as http_err:
                    print(f"⚠️ Gagal mengirim webhook progres: {http_err}", flush=True)
            
        print("✅ Transkripsi selesai! Mengirim hasil akhir ke Laravel (Webhook)...", flush=True)
        
        # Kirim hasil akhir (status: completed)
        try:
            response = requests.post(callback_url, json={"status": "completed", "transcription": final_transcription}, timeout=30)
            response.raise_for_status()
            print("🚀 Webhook completed berhasil terkirim ke Laravel!", flush=True)
        except Exception as http_err:
            print(f"⚠️ Gagal mengirim webhook completed ke Laravel: {http_err}", flush=True)
            
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat proses background: {str(e)}", flush=True)
        import requests
        try:
            # Kirim webhook kosong/error agar Laravel tahu proses gagal
            requests.post(callback_url, json={"transcription": [], "error": str(e)}, timeout=10)
        except:
            pass
            
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print("🧹 File sementara berhasil dibersihkan.", flush=True)