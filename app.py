import os
import shutil
import tempfile
import json
import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

app = FastAPI()

print("⏳ Memuat model Whisper ke memori...", flush=True)
# Menggunakan compute_type int8 untuk menghemat RAM dan thread cpu secukupnya
model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=2)
print("✅ Model siap menerima request!", flush=True)


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
  print(f"📥 Menerima file: {file.filename}", flush=True)

  temp_file_path = None
  try:
    ext = os.path.splitext(file.filename)[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
      shutil.copyfileobj(file.file, temp_file)
      temp_file_path = temp_file.name

    print(f"⚙️ File disimpan sementara di: {temp_file_path}", flush=True)
    print("⏳ Mulai proses pembedahan audio oleh Whisper...", flush=True)

    async def generate_transcription():
      try:
        # VAD & Inisialisasi awal transkripsi
        segments, info = await asyncio.to_thread(
            model.transcribe, temp_file_path, beam_size=5
        )
        
        print(f"🌍 Terdeteksi bahasa: {info.language} dengan probabilitas {info.language_probability}", flush=True)
        
        iterator = iter(segments)
        while True:
            # Mengambil segmen teks satu per satu di thread terpisah agar tidak block event loop
            try:
                segment = await asyncio.to_thread(next, iterator)
            except StopIteration:
                break
                
            # Cetak ke server.log secara real-time
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}", flush=True)
            
            # Kirim data ke client (Postman/cPanel) secara streaming sebagai NDJSON / Chunked JSON
            yield json.dumps({
                "status": "processing",
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }) + "\n"

        print("✅ Transkripsi selesai!", flush=True)
        yield json.dumps({"status": "success", "message": "Transkripsi selesai"}) + "\n"

      except Exception as e:
        print(f"❌ Terjadi kesalahan saat streaming transkripsi: {str(e)}", flush=True)
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        
      finally:
        if temp_file_path and os.path.exists(temp_file_path):
          os.remove(temp_file_path)
          print("🧹 File sementara berhasil dibersihkan.", flush=True)

    # Mengembalikan StreamingResponse dengan mimetype x-ndjson (Newline Delimited JSON)
    return StreamingResponse(generate_transcription(), media_type="application/x-ndjson")

  except Exception as e:
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    print(f"❌ Terjadi kesalahan awal: {str(e)}", flush=True)
    raise HTTPException(
        status_code=500, detail=f"Internal Server Error: {str(e)}"
    )