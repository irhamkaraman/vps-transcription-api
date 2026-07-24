from faster_whisper import WhisperModel

print("[Start] Mulai mengunduh model Whisper (medium)...")

model = WhisperModel("medium", device="cpu", compute_type="int8")
print("[Complete] Model berhasil diunduh dan siap digunakan!")