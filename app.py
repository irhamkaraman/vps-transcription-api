import os
import shutil
import tempfile
import torch
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile

torch.backends.mkldnn.enabled = False
torch.set_num_threads(2)

app = FastAPI()

print("⏳ Memuat model Whisper ke memori...")
model = whisper.load_model("large-v3")
print("✅ Model siap menerima request!")


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
  print(f"📥 Menerima file: {file.filename}")

  temp_file_path = None
  try:
    ext = os.path.splitext(file.filename)[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
      shutil.copyfileobj(file.file, temp_file)
      temp_file_path = temp_file.name

    print(f"⚙️ File disimpan sementara di: {temp_file_path}")
    print("⏳ Mulai proses pembedahan audio oleh Whisper...")

    result = model.transcribe(temp_file_path, fp16=False)
    print("✅ Transkripsi sukses!")

    return {
        "status": "success",
        "filename": file.filename,
        "language": result.get("language", "id"),
        "text": result["text"].strip(),
    }

  except Exception as e:
    print(f"❌ Terjadi kesalahan: {str(e)}")
    raise HTTPException(
        status_code=500, detail=f"Internal Server Error: {str(e)}"
    )

  finally:
    if temp_file_path and os.path.exists(temp_file_path):
      os.remove(temp_file_path)
      print("🧹 File sementara berhasil dibersihkan.")