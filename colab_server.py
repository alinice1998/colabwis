import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import shutil
from alignment_engine import AlignmentEngine

app = FastAPI(title="Miqat Colab Alignment Server")

# Models will be downloaded to model_local by model_downloader.py in colab
WHISPER_PATH = "model_local/whisper"
WAV2VEC2_PATH = "model_local/wav2vec2"

# Ensure models exist or raise error early
if not os.path.exists(WHISPER_PATH) or not os.path.exists(WAV2VEC2_PATH):
    print("Warning: Models not found locally. Please run model_downloader.py first.")

alignment_engine = AlignmentEngine(WHISPER_PATH, WAV2VEC2_PATH)

os.makedirs("temp_audio", exist_ok=True)

@app.post("/align/cloud")
async def align_cloud(
    file: UploadFile = File(...),
    reference_text: str = Form(...),
    method: str = Form(...) # 'ctc' or 'whisper'
):
    if not reference_text:
        return JSONResponse({"status": "error", "message": "No reference text provided"}, status_code=400)
        
    file_path = f"temp_audio/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        if method == "whisper":
            alignments = alignment_engine.align_whisper(file_path, reference_text)
        else:
            alignments = alignment_engine.align_smart(file_path, reference_text)
            
        return JSONResponse({
            "status": "success",
            "method": method,
            "alignments": alignments
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
