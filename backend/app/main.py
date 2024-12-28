from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
from openai import OpenAI
import yt_dlp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Print environment variables for debugging
print("Environment variables:", {k: v for k, v in os.environ.items() if 'OPENAI' in k.upper()})

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoAnalysisRequest(BaseModel):
    video_url: str
    num_frames: int = 5
    start_time: float = 0
    end_time: Optional[float] = None
    additional_goals: str = ""
    model: str = "gpt-4"
    temperature: float = 0.7

# Helper Functions from app.py
def extract_frames_for_segment(video_path: str, start_time: float, end_time: float, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Cannot read video FPS")

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_segment_frames = max(end_frame - start_frame, 1)
    step = max(total_segment_frames // max(num_frames, 1), 1)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while cap.isOpened() and frame_idx <= end_time*fps and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_idx += 1

    cap.release()
    return frames

def compute_clip_embeddings(frames: List[Image.Image], model_version="openai/clip-vit-base-patch32") -> torch.Tensor:
    processor = CLIPProcessor.from_pretrained(model_version)
    model = CLIPModel.from_pretrained(model_version)
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
    return embeddings

def load_captioning_model():
    blip_model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(blip_model_name)
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
    model.eval()
    return processor, model

def generate_captions_for_frames(frames: List[Image.Image]) -> List[str]:
    if not frames:
        return []
    processor, model = load_captioning_model()
    captions = []
    for frame in frames:
        inputs = processor(frame, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=50, num_beams=5)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def extract_concepts_gpt(captions: List[str], model: str = "gpt-4", temperature: float = 0.7) -> List[str]:
    try:
        client = OpenAI()
        prompt = f"""Given these video frame captions:
{chr(10).join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])}

Extract 5-10 key visual concepts that could be detected by a computer vision model.
Focus on concrete, visual elements (objects, actions, settings).
Format as a simple comma-separated list.
Example: "person running, snowy mountain, camping tent, forest trail, backpack"
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts key visual concepts from video frame captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=100
        )

        concepts = response.choices[0].message.content.strip().split(",")
        return [concept.strip() for concept in concepts]
    except Exception as e:
        print(f"OpenAI API not available, using mock concepts: {str(e)}")
        # Return mock concepts based on captions
        return ["video scene", "visual element", "motion", "background", "object"]

def generate_summary(frame_captions: List[str], frame_concepts: List[List[tuple]], temperature: float, model: str, additional_goals: str) -> str:
    try:
        client = OpenAI()
        frame_descriptions = []
        for i, caption in enumerate(frame_captions):
            if i < len(frame_concepts):
                top_concepts_str = ", ".join([f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_concepts[i]])
                frame_descriptions.append(
                    f"Frame {i+1}: Caption='{caption}' WeightedConcepts={top_concepts_str}"
                )
            else:
                frame_descriptions.append(f"Frame {i+1}: Caption='{caption}'")

        system_message = (
            "You are a video summarizer who focuses on what's visually present. "
            "Write a short cinematic paragraph (3–5 sentences) capturing the frames. "
            "Highlight higher-weighted concepts more. If relevant, mention lower-weight items briefly."
        )

        user_prompt = f"""
We have frames from a video segment, each with an auto-generated caption and WeightedConcepts:

{chr(10).join(frame_descriptions)}

User's additional goals:
{additional_goals}

Your task:
1) Provide a concise, cinematic paragraph summarizing what's happening.
2) Emphasize higher-weighted concepts more.
3) Reference any lower-weight items if they help the narrative.
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=250
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API not available, using mock summary: {str(e)}")
        # Return mock summary based on captions
        return "This video shows various scenes and elements. " + " ".join(frame_captions[:2]) + " [Mock summary due to OpenAI API unavailability]"

@app.post("/analyze")
async def analyze_video(request: VideoAnalysisRequest):
    try:
        print(f"Starting video analysis for URL: {request.video_url}")
        # Check if it's a direct video URL or YouTube
        if request.video_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Detected direct video URL, starting download...")
            # For direct video URLs, download using requests
            import requests
            import tempfile
            
            try:
                response = requests.get(
                    request.video_url,
                    stream=True,
                    timeout=30,  # 30 seconds timeout for initial connection
                    headers={'User-Agent': 'Mozilla/5.0'}  # Add user agent to avoid some blocks
                )
                response.raise_for_status()
                print("Video download started successfully")
                
                # Create a temporary file with the correct extension
                ext = os.path.splitext(request.video_url)[1]
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                        total_size += len(chunk)
                        if total_size % (1024 * 1024) == 0:  # Log every MB
                            print(f"Downloaded {total_size / (1024 * 1024):.1f} MB")
                    video_path = temp_file.name
                print(f"Video downloaded successfully to {video_path}, total size: {total_size / (1024 * 1024):.1f} MB")
            except Exception as e:
                print(f"Error downloading video: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
            
            # Get video duration using OpenCV
            print("Opening video file with OpenCV...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Failed to open video file with OpenCV")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            print(f"Video opened successfully: {frame_count} frames, {fps} FPS, {duration:.1f} seconds duration")
            cap.release()
            
            video_info = {'duration': duration}
        else:
            # Use yt-dlp for YouTube and other streaming platforms
            with yt_dlp.YoutubeDL({'format': 'best'}) as ydl:
                info = ydl.extract_info(request.video_url, download=True)
                video_path = info['requested_downloads'][0]['filepath']
                video_info = info

        # Extract frames
        print("Extracting frames from video...")
        frames = extract_frames_for_segment(
            video_path,
            request.start_time,
            request.end_time or float(video_info['duration']),
            request.num_frames
        )
        print(f"Successfully extracted {len(frames)} frames")

        # Generate captions
        print("Generating captions for frames...")
        captions = generate_captions_for_frames(frames)
        print(f"Generated captions: {captions}")

        # Extract concepts
        print("Extracting concepts from captions...")
        concepts = extract_concepts_gpt(captions, request.model, request.temperature)
        print(f"Extracted concepts: {concepts}")

        # Generate summary
        print("Generating video summary...")
        summary = generate_summary(
            captions,
            [[("concept", 1.0)] for concept in concepts],  # Simplified concept weights
            request.temperature,
            request.model,
            request.additional_goals
        )
        print(f"Generated summary: {summary}")

        # Clean up downloaded video
        os.remove(video_path)

        return {
            "message": "Success",
            "results": {
                "captions": captions,
                "concepts": concepts,
                "summary": summary
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
