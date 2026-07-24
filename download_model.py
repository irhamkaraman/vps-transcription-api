"""
Download Whisper Model untuk VPS Transcription API

Cara pakai:
  python download_model.py              # Download model base
  python download_model.py medium       # Download model medium
  python download_model.py large-v3     # Download model large-v3

Model akan disimpan di folder: models/
"""

import os
import sys
import time
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("[ERROR] Library 'faster-whisper' belum terinstall!")
    print("        Jalankan: pip install faster-whisper")
    sys.exit(1)

# Folder penyimpanan model
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Daftar model yang tersedia
AVAILABLE_MODELS = {
    "tiny":   "39 MB",
    "base":   "74 MB",
    "small":  "244 MB",
    "medium": "769 MB",
    "large":  "1.5 GB",
    "large-v2": "1.5 GB",
    "large-v3": "1.5 GB",
}

# Model default yang digunakan
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "base")


def download_model(model_name: str):
    """Download model Whisper menggunakan faster-whisper"""
    print(f"\n{'='*50}")
    print(f"🚀 DOWNLOAD WHISPER MODEL")
    print(f"{'='*50}")
    print(f"   Model: {model_name}")
    print(f"   Ukuran: {AVAILABLE_MODELS.get(model_name, 'Unknown')}")
    print(f"   Lokasi: {MODEL_DIR}")
    print(f"{'='*50}\n")

    if model_name not in AVAILABLE_MODELS:
        print(f"[ERROR] Model '{model_name}' tidak tersedia!")
        print(f"        Model tersedia: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    # Cek apakah model sudah ada
    model_cache_dir = MODEL_DIR / model_name
    if model_cache_dir.exists() and any(model_cache_dir.iterdir()):
        print(f"[INFO] Model '{model_name}' sudah ada di: {model_cache_dir}")
        print(f"[INFO] Gunakan: WHISPER_MODEL={model_name} python -m uvicorn app:app")
        return

    print(f"⏳ Downloading model '{model_name}'...")
    print(f"   Harap tunggu, ini bisa memakan waktu beberapa menit...\n")

    t0 = time.time()
    try:
        # faster-whisper akan otomatis download model ke cache folder
        # Kita paksa download dengan membuat instance model
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(MODEL_DIR)
        )
        load_time = time.time() - t0
        print(f"\n✅ Download berhasil! ({load_time:.1f}s)")
        print(f"   Model disimpan di: {MODEL_DIR / model_name}")
        print(f"\n{'='*50}")
        print(f"📋 CARA PAKAI:")
        print(f"{'='*50}")
        print(f"   1. Set environment variable:")
        print(f"      export WHISPER_MODEL={model_name}")
        print(f"\n   2. Jalankan server:")
        print(f"      python -m uvicorn app:app --host 0.0.0.0 --port 8000")
        print(f"\n   3. Atau pakai .env file:")
        print(f"      echo 'WHISPER_MODEL={model_name}' >> .env")
        print(f"      uvicorn app:app --host 0.0.0.0 --port 8000")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"\n❌ Download gagal: {e}")
        print(f"   Pastikan koneksi internet stabil")
        sys.exit(1)


if __name__ == "__main__":
    # Jika ada argumen command line, pakai itu
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = DEFAULT_MODEL

    download_model(model_name)
