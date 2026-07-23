import whisper

print("⏳ Mulai mengunduh model Whisper (base)...")

model = whisper.load_model("base")
print("✅ Model berhasil diunduh dan siap digunakan!")