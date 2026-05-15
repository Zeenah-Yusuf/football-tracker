from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
import os
import shutil
import time
import uuid
from pathlib import Path
from processor import FastDemoProcessor

app = FastAPI(
    title="Football Player Tracker API",
    description="AI-powered player tracking with TRUE jersey color detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("/tmp/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

processor = None

def get_processor():
    global processor
    if processor is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ROBOFLOW_API_KEY not configured")
        processor = FastDemoProcessor(api_key=api_key)
    return processor


@app.get("/")
async def root():
    return {
        "name": "Football Player Tracker API",
        "version": "2.0.0",
        "feature": "True Jersey Color Detection",
        "endpoints": {
            "POST /process": "Upload and process video",
            "GET /download/{job_id}/{type}": "Download processed video",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    max_duration: int = 15,
    confidence: int = 20
):
    """Upload and process a football video with true jersey colors"""
    
    allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska']
    if video.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use MP4, AVI, MOV, or MKV.")
    
    video.file.seek(0, 2)
    file_size = video.file.tell()
    video.file.seek(0)
    
    if file_size > 500 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 500MB.")
    
    job_id = str(uuid.uuid4())[:8]
    
    input_path = OUTPUT_DIR / f"{job_id}_input.mp4"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    try:
        proc = get_processor()
        
        start_time = time.time()
        tracked_path, birdseye_path, frame_count = proc.process_video(
            str(input_path), max_seconds=max_duration
        )
        elapsed = time.time() - start_time
        
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        final_tracked = job_dir / "tracked.mp4"
        final_birdseye = job_dir / "birdseye.mp4"
        
        shutil.move(tracked_path, final_tracked)
        shutil.move(birdseye_path, final_birdseye)
        
        os.unlink(input_path)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "frame_count": frame_count,
            "processing_time": round(elapsed, 1),
            "fps": round(frame_count / elapsed, 1) if elapsed > 0 else 0,
            "downloads": {
                "tracked": f"/download/{job_id}/tracked",
                "birdseye": f"/download/{job_id}/birdseye"
            },
            "preview": {
                "tracked": f"/outputs/{job_id}/tracked.mp4",
                "birdseye": f"/outputs/{job_id}/birdseye.mp4"
            }
        })
        
    except Exception as e:
        if input_path.exists():
            os.unlink(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{job_id}/{file_type}")
async def download_video(job_id: str, file_type: str):
    if file_type not in ['tracked', 'birdseye']:
        raise HTTPException(status_code=400, detail="Use 'tracked' or 'birdseye'")
    
    file_path = OUTPUT_DIR / job_id / f"{file_type}.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="video/mp4", filename=f"{file_type}_{job_id}.mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
