from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="VideoLM API", description="Video Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoAnalysisRequest(BaseModel):
    video_url: str
    num_frames: int = 5
    start_time: float = 0
    end_time: Optional[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    """Analyze a video and return mock results for testing"""
    try:
        return {
            "message": "Success",
            "results": {
                "captions": [
                    "A person walking down a street",
                    "Cars passing by in the background",
                    "Buildings visible on both sides"
                ],
                "concepts": [
                    "urban environment",
                    "pedestrian activity",
                    "traffic",
                    "architecture"
                ],
                "summary": "A typical urban scene unfolds as a person walks down a city street. Cars move steadily in the background while tall buildings line both sides of the thoroughfare, creating a quintessential cityscape."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
