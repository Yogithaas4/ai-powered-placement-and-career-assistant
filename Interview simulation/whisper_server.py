# whisper_server.py — runs on your laptop CPU
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile, os

app = FastAPI()

# CPU mode — int8 is fastest on CPU, good accuracy with base model
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.get("/health")
def health():
    return {"model": "base", "device": "cpu"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(await file.read())
        tmp = f.name
    segments, _ = model.transcribe(tmp)
    text = " ".join(s.text for s in segments)
    os.unlink(tmp)
    return {"text": text.strip()}