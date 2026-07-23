import whisper

print("[Start] Mulai mengunduh model Whisper (base)...")

model = whisper.load_model("large-v3")
print("[Complete] Model berhasil diunduh dan siap digunakan!")