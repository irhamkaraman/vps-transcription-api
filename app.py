import os
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile
import whisper

app = FastAPI(title="Whisper Offline Transcription API")

# Load model ke memori RAM saat server menyala
print("⏳ Memuat model Whisper ke memori...")
model = whisper.load_model("large-v3")
print("✅ Model siap menerima request!")

UPLOAD_DIR = "storage/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Memproses transkripsi untuk: {file.filename}")
        result = model.transcribe(file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "language": result["language"],
            "text": result["text"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)