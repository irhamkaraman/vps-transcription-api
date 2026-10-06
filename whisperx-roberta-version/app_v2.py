import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time
import requests
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Harusnya dibatasi ke domain temaniskripsi.id
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RoBERTa Architecture (Copied from Colab) ---
class MultiTaskGraphRoBERTa(nn.Module):
    def __init__(self, num_advice_classes, num_modes_classes, model_name="cahya/roberta-base-indonesian-522M"):
        super(MultiTaskGraphRoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
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
        logits_advice = self.classifier_advice(pooled_output)
        logits_modes = self.classifier_modes(pooled_output)
        return logits_advice, logits_modes

# Inisialisasi Model (Mockup structure, sesuaikan dengan load_state_dict asli)
# tokenizer = AutoTokenizer.from_pretrained(...)
# model = MultiTaskGraphRoBERTa(...)

# --- Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, slug: str):
        await websocket.accept()
        self.active_connections[slug] = websocket

    def disconnect(self, slug: str):
        if slug in self.active_connections:
            del self.active_connections[slug]

    async def send_progress(self, slug: str, message: dict):
        if slug in self.active_connections:
            try:
                await self.active_connections[slug].send_json(message)
            except WebSocketDisconnect:
                self.disconnect(slug)

manager = ConnectionManager()

# --- Endpoint Realtime WebSocket ---
@app.websocket("/api/v2/ws/{slug}")
async def websocket_endpoint(websocket: WebSocket, slug: str):
    await manager.connect(websocket, slug)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(slug)

# --- Background Task Pipeline ---
async def process_audio_pipeline(slug: str, callback_url: str):
    await manager.send_progress(slug, {"progress": 10, "message": "Memulai WhisperX diarization..."})
    await asyncio.sleep(2) # Mock processing time
    
    await manager.send_progress(slug, {"progress": 40, "message": "Klasifikasi segment dengan RoBERTa Multi-Task..."})
    await asyncio.sleep(2) # Mock processing time
    
    await manager.send_progress(slug, {"progress": 80, "message": "Membangun Graph Nodes & Edges..."})
    
    # Mockup Data Graph & Transcription
    transcription = [
        {"speaker": "Dosen", "text": "Coba perbaiki bab 2.", "advice_giving": "arahan_eksplisit", "modes_of_interaction": "power_over"}
    ]
    graph_data = {
        "nodes": [{"id": "Dosen", "label": "Dosen", "group": "speaker"}, {"id": "Mhs", "label": "Mahasiswa", "group": "speaker"}],
        "edges": [{"from": "Dosen", "to": "Mhs", "label": "power_over"}]
    }
    
    await manager.send_progress(slug, {"progress": 100, "message": "Selesai! Mengirim data ke server..."})
    
    # Send Webhook to Laravel
    requests.post(callback_url, json={"transcription": transcription, "graph_data": graph_data}, verify=False)
    
    # Beri sinyal ke frontend untuk redirect
    await manager.send_progress(slug, {"status": "completed"})

@app.post("/api/v2/transcribe")
async def transcribe_v2(background_tasks: BackgroundTasks, slug: str = Form(...), callback_url: str = Form(...), file: UploadFile = File(...)):
    # Simpan file sementara di sini
    background_tasks.add_task(process_audio_pipeline, slug, callback_url)
    return {"status": "processing_started"}
