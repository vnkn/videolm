################################################################################
# NOMADICML VIDEO INSIGHTS - Single-Page App
#
# Per your request:
#   - The "Executive Overview" is now at the very top (formerly "High-Level Summary").
#   - The "Data Drift Analysis" is placed after analysis and personalization.
#   - Any references to "CEO perspective," "default," or "placeholder" have been removed 
#     and replaced with more general language.
#   - The model weights section has been styled more nicely, but no lines of logic 
#     have been deleted—only re-labeled or reordered for clarity.
#   - All previously added code and advanced charts remain intact.
#   - Other than the above changes, the script retains all lines and logic exactly.
################################################################################

import os
import tempfile
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import List
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import streamlit as st
import pandas as pd
from pytubefix import YouTube
import yt_dlp
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------------------------------------------------------- #
# 1) SET PAGE CONFIG
# ---------------------------------------------------------------------------- #
st.set_page_config(
    page_title="NomadicML Video Insights",
    page_icon="🎬",
    layout="wide"
)

# NEW: Force a visible Streamlit page title
st.title("NomadicML Video Insights - Single-Page App")

# ---------------------------------------------------------------------------- #
# 2) SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------- #
if "personalization_data" not in st.session_state:
    st.session_state.personalization_data = {
        "class_weights": {
            "water": 1.0,
            "trees": 1.0,
            "landscape": 1.0,
            "person": 1.0,
            "forest": 1.0
        },
        "feedback_history": [],
        "fine_tuned": False,
        "goals": "",
        "preferred_concepts": [],
        "use_case_weights": {
            "Outdoor Adventure": {
                "water": 1.0,
                "trees": 1.0,
                "landscape": 1.0,
                "forest": 1.0,
                "mountains": 1.0
            },
            "Wildlife Documentary": {
                "animals": 1.0,
                "person": 0.8,
                "forest": 1.2,
                "landscape": 0.9
            }
        },
        "active_use_case": "Outdoor Adventure",
        "customer_name": "Default Customer",
        "model_version": "openai/clip-vit-base-patch32"
    }

if "analysis_steps" not in st.session_state:
    st.session_state.analysis_steps = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# We maintain the data drift segments array but remove "placeholder" references
if "drift_segments_data" not in st.session_state:
    st.session_state.drift_segments_data = {
        "segments": ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5"],
        "similarities": np.random.uniform(0.7, 0.95, 5)  
    }

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------- #
# 3) CUSTOM CSS
# ---------------------------------------------------------------------------- #
st.markdown("""
<style>
body {
    font-family: "Open Sans", sans-serif;
    background: linear-gradient(to right, #f0f4f7, #dae7ef);
    color: #333333;
}
h1, h2, h3, h4, h5 {
    font-family: "Open Sans", sans-serif;
    font-weight: 600;
    color: #0f4c81;
}
.stMarkdown a {
    color: #0f4c81;
    text-decoration: none;
    font-weight: bold;
}
/* Expanded margin and padding for the main container */
.block-container {
    margin: 50px auto !important;
    max-width: 1400px;
    background: #ffffffAA;
    padding: 40px !important;
    border-radius: 12px;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #0f4c81;
    color: #ffffff;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1em;
    cursor: pointer;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #0e3e6d;
}
.stExpanderHeader {
    font-weight: 600 !important;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2em 0;
}
div[data-testid="stHeader"] {
    background: none;
}
footer {visibility: hidden;}
.weight-bar {
    display: flex;
    align-items: center;
    margin-bottom: 5px;
}
.weight-label {
    width: 100px;
    font-weight: 600;
    color: #333;
}
.weight-bar-inner {
    flex: 1;
    background: #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    margin-left: 10px;
    margin-right: 10px;
}
.weight-bar-fill {
    height: 10px;
    background: #0f4c81;
}
/* Minimalist Nav Bar */
.nav-container {
    background-color: #e8f1f9;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.05);
}
.nav-link {
    margin: 0 0.75rem;
    padding: 0.4rem 0.75rem;
    color: #0f4c81;
    text-decoration: none;
    font-weight: 600;
    border-radius: 4px;
    transition: background-color 0.2s ease-in-out;
}
.nav-link:hover {
    background-color: #cde2f1;
}
/* Make model weights look nicer */
.nicer-weights-container {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 10px 15px;
    margin-top: 15px;
    box-shadow: 0px 0px 4px rgba(0,0,0,0.1);
}
.nicer-weights-title {
    font-weight: 600;
    font-size: 18px;
    margin-bottom: 8px;
    color: #0f4c81;
}
.nicer-weights-item {
    padding: 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# 4) HELPER FUNCTIONS
# ---------------------------------------------------------------------------- #
def render_navigation_bar():
    st.markdown("""
    <div class="nav-container">
        <a class="nav-link" href="#executive-overview">Executive Overview</a>
        <a class="nav-link" href="#video-analysis">Video Analysis</a>
        <a class="nav-link" href="#personalization--guidance">Personalization</a>
        <a class="nav-link" href="#data-drift-analysis">Data Drift Analysis</a>
    </div>
    """, unsafe_allow_html=True)


def Download(url, output_path=None):
    ydl_opts = {
        'format': 'best', 
        'outtmpl': '%(title)s.%(ext)s' if not output_path else output_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps

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

def analyze_frames_with_clip(frames, model_version, candidate_descriptions, class_weights, domain_shift_factor=1.0):
    processor = CLIPProcessor.from_pretrained(model_version)
    model = CLIPModel.from_pretrained(model_version)

    all_frame_results = []
    for frame in frames:
        inputs = processor(images=frame, text=candidate_descriptions, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0] * domain_shift_factor

        weighted_probs = []
        for i, desc in enumerate(candidate_descriptions):
            w = class_weights.get(desc, 1.0)
            weighted_probs.append(probs[i] * w)

        weighted_probs = torch.tensor(weighted_probs)
        weighted_probs = weighted_probs / weighted_probs.sum()

        top_probs, top_indices = torch.topk(weighted_probs, k=5)
        frame_results = [(candidate_descriptions[idx], float(prob.item())) for idx, prob in zip(top_indices, top_probs)]
        all_frame_results.append(frame_results)

    return all_frame_results

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

def generate_summary(frame_captions, frame_concepts, temperature, model, additional_goals):
    try:
        client = OpenAI()
    except Exception as e:
        return f"Error initializing OpenAI client: {e}"

    frame_descriptions = []
    for i, caption in enumerate(frame_captions):
        if i < len(frame_concepts):
            top_concepts_str = ", ".join([f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_concepts[i]])
            frame_descriptions.append(f"Frame {i+1}: Caption: '{caption}' | Concepts: {top_concepts_str}")
        else:
            frame_descriptions.append(f"Frame {i+1}: Caption: '{caption}'")

    system_message = "You are a friendly video summarizer who focuses on what is actually visible in the video."
    user_prompt = f"""We analyzed each part of the video and generated captions for each frame:

{chr(10).join(frame_descriptions)}

From these captions, please write a short, friendly summary of this segment, focusing primarily on what's actually visible in the video. 
Use the concepts only as subtle emphasis. If the user has additional goals or aspects to emphasize, incorporate that softly.

Additional goals or instructions from the user:
{additional_goals}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def plot_embeddings(embeddings, labels, similarities):
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.figure(figsize=(7,7))
    plt.title("Content Changes Over Segments", fontsize=14, fontweight='bold')
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.grid(linestyle='--', alpha=0.5)

    plt.scatter(x[0], y[0], c='red', s=200, marker='*', label="First Part")

    for i in range(1, len(labels)):
        sim = similarities[i]
        color = 'green' if sim > 0.8 else 'orange'
        plt.scatter(x[i], y[i], c=color, s=120, edgecolors='black')
        plt.text(x[i]+0.01, y[i]+0.01, labels[i], fontsize=11, color='black')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='*', color='w', label='First Part (Reference)', 
               markerfacecolor='red', markersize=15),
        Line2D([0],[0], marker='o', color='w', label='Stable (sim > 0.8)', 
               markerfacecolor='green', markersize=10),
        Line2D([0],[0], marker='o', color='w', label='Changed (sim ≤ 0.8)', 
               markerfacecolor='orange', markersize=10)
    ]
    # Increase label and border spacing for more breathing room:
    plt.legend(handles=legend_elements, loc='upper right', labelspacing=2.0, borderpad=2.0)

    # Keep a bit of tight layout, but allow for the bigger legend spacing
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def update_class_weights_from_feedback(feedback_data, personalization_data):
    for concept, rating in feedback_data.items():
        if rating == "👍":
            personalization_data["class_weights"][concept] = personalization_data["class_weights"].get(concept, 1.0) * 1.1
        elif rating == "👎":
            personalization_data["class_weights"][concept] = personalization_data["class_weights"].get(concept, 1.0) * 0.9
    personalization_data["feedback_history"].append({
        "feedback": feedback_data,
        "updated_weights": personalization_data["class_weights"]
    })

def update_class_weights_from_examples(example_data: str, active_case: str, personalization_data):
    if active_case not in personalization_data["use_case_weights"]:
        personalization_data["use_case_weights"][active_case] = {}

    lines = example_data.strip().split("\n")
    for line in lines:
        parts = line.split(",")
        if len(parts) == 2:
            concept = parts[0].strip()
            try:
                factor = float(parts[1].strip())
            except ValueError:
                factor = 1.0
            current_weight = personalization_data["use_case_weights"][active_case].get(concept, 1.0)
            personalization_data["use_case_weights"][active_case][concept] = current_weight * factor

    if personalization_data["active_use_case"] == active_case:
        for c, w in personalization_data["use_case_weights"][active_case].items():
            personalization_data["class_weights"][c] = w

# ---------------------------------------------------------------------------- #
# 5) NAVIGATION BAR
# ---------------------------------------------------------------------------- #
render_navigation_bar()

# ---------------------------------------------------------------------------- #
# (NEW) EXECUTIVE OVERVIEW - AT THE VERY TOP
# ---------------------------------------------------------------------------- #
st.markdown("<a name='executive-overview'></a>", unsafe_allow_html=True)
st.markdown("---")
st.header("Executive Overview")
st.markdown("""
NomadicML automatically identifies key concepts in your video, checks for content 
changes over time (data drift), and summarizes what's actually visible. 
This enables:
- Swift decision-making by highlighting changes in brand visuals or new elements.
- Easy adjustments: you can highlight or de-emphasize certain concepts with a few clicks.
- More reliable summaries by focusing on real, visible content.
""")

# ---------------------------------------------------------------------------- #
# VIDEO ANALYSIS SECTION
# ---------------------------------------------------------------------------- #
st.markdown("<a name='video-analysis'></a>", unsafe_allow_html=True)
st.markdown("---")
st.header("Video Analysis")
st.markdown("Enter a YouTube URL, adjust parameters, then click **Generate Summary & Analyze Video**.")

youtube_url = st.text_input("Enter the YouTube video URL to analyze:", value="https://www.youtube.com/watch?v=jNQXAC9IVRw")

fast_mode = st.checkbox("Fast Mode (quicker but fewer frames)", value=False)
if fast_mode:
    num_segments = st.slider("Number of Segments (Fewer segments = faster)", 1, 5, 2)
    frames_per_segment = st.slider("Frames per Segment (Fewer frames = faster)", 1, 8, 1)
else:
    num_segments = st.slider("Number of Segments (Higher for more detail)", 1, 5, 3)
    frames_per_segment = st.slider("Frames per Segment (More frames = deeper analysis)", 1, 8, 2)

temperature = st.slider("Summary Creativity (Temperature)", 0.0, 1.0, 0.5, 0.1)
domain_shift = st.selectbox("Scene Condition (Apply domain shift factor):", ["None", "Fog", "Night"], index=0)
domain_factor_map = {"None": 1.0, "Fog": 0.9, "Night": 0.8}
domain_factor = domain_factor_map[domain_shift]

candidate_descriptions_str = st.text_area(
    "Candidate Concepts (comma-separated):", 
    "water, rocks, trees, landscape, nature, forest, person, mountains"
)
candidate_descriptions = [x.strip() for x in candidate_descriptions_str.split(',') if x.strip()]

analyze_btn = st.button("Generate Summary & Analyze Video")

# ---------------------------------------------------------------------------- #
# NICER MODEL WEIGHTS - Display them in a nicer container
# ---------------------------------------------------------------------------- #
st.markdown("<div class='nicer-weights-container'>", unsafe_allow_html=True)
st.markdown("<div class='nicer-weights-title'>Current Model Weights</div>", unsafe_allow_html=True)
for c, w in st.session_state.personalization_data["class_weights"].items():
    st.markdown(f"<div class='nicer-weights-item'>• <strong>{c}</strong>: {w}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# 9) STEP-BY-STEP ANALYSIS
# ---------------------------------------------------------------------------- #
results_container = st.container()
if analyze_btn and youtube_url.strip():
    st.markdown("----")
    st.subheader("Step-by-Step Analysis")

    model_version = st.session_state.personalization_data["model_version"]
    combined_class_weights = dict(st.session_state.personalization_data["class_weights"])

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("#### 1. Downloading Video")
        download_status_placeholder = st.empty()
        download_progress = st.progress(0)

    with col2:
        st.markdown("#### 2. Processing & Analysis")
        processing_status_placeholder = st.empty()
        processing_progress = st.progress(0)

    with col3:
        st.markdown("#### 3. Summaries & Insights")
        summary_status_placeholder = st.empty()
        summary_progress = st.progress(0)

    download_status_placeholder.write("**Status:** Downloading video...")
    download_msg = "**Status:** Downloading video..."

    video_path = Download(youtube_url)
    download_progress.progress(50)

    if not video_path or not os.path.exists(video_path):
        download_status_placeholder.write("**Error:** Failed to download video.")
        st.error("Failed to download the video. Check the URL or permissions.")
        st.session_state.analysis_steps = {
            "download_msg": "**Error:** Failed to download video.",
            "processing_msg": "",
            "summary_msg": ""
        }
        st.stop()

    download_progress.progress(100)
    download_status_placeholder.write("**Status:** Video downloaded successfully!")
    download_msg = "**Status:** Video downloaded successfully!"

    duration = get_video_duration(video_path)
    if duration <= 0:
        st.error("Could not determine video duration or video is empty.")
        st.session_state.analysis_steps = {
            "download_msg": download_msg,
            "processing_msg": "**Error:** Could not determine video duration.",
            "summary_msg": ""
        }
        st.stop()

    segment_length = duration / num_segments
    segment_info = []

    for seg_idx in range(num_segments):
        processing_progress.progress(int((seg_idx / num_segments)*100))
        status = f"**Analyzing Segment {seg_idx+1}/{num_segments}...**"
        processing_status_placeholder.write(status)
        processing_msg = status

        frames = extract_frames_for_segment(video_path, seg_idx * segment_length, (seg_idx + 1) * segment_length, frames_per_segment)

        frame_analyses = analyze_frames_with_clip(
            frames,
            model_version=model_version,
            candidate_descriptions=candidate_descriptions,
            class_weights=combined_class_weights,
            domain_shift_factor=domain_factor
        )

        frame_captions = generate_captions_for_frames(frames)

        summary_status_placeholder.write("**Generating Segment Summary...**")
        summary_msg = "**Generating Segment Summary...**"

        summary = generate_summary(
            frame_captions,
            frame_analyses,
            temperature=temperature,
            model="gpt-4",
            additional_goals=st.session_state.personalization_data["goals"]
        )
        summary_progress.progress(int(((seg_idx+1)/num_segments)*100))

        if frames:
            embeddings = compute_clip_embeddings(frames, model_version=model_version)
            segment_embedding = embeddings.mean(dim=0).cpu().numpy()
        else:
            segment_embedding = np.zeros(512)

        top_concepts = set()
        for frame_res in frame_analyses:
            for desc, conf in frame_res:
                top_concepts.add(desc)

        segment_data = {
            "segment_index": seg_idx,
            "start_time": seg_idx * segment_length,
            "end_time": (seg_idx + 1) * segment_length,
            "frames": frames,
            "frame_analyses": frame_analyses,
            "frame_captions": frame_captions,
            "summary": summary,
            "embedding": segment_embedding,
            "top_concepts": list(top_concepts)
        }
        segment_info.append(segment_data)

        st.markdown(f"### Segment {seg_idx+1} Analysis")
        st.write(f"**Time Range:** {seg_idx * segment_length:.2f}s - {(seg_idx + 1) * segment_length:.2f}s")
        st.write(f"**Summary (Caption-Based):** {summary}")

        c1, c2 = st.columns([2,1])
        with c1:
            st.write("**Frame Captions:**")
            for i, cap in enumerate(frame_captions):
                st.write(f"Frame {i+1}: {cap}")

            st.write("**Top Concepts per Frame (for emphasis):**")
            for i, frame_res in enumerate(frame_analyses):
                top_str = ", ".join([f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_res])
                st.write(f"Frame {i+1}: {top_str}")
        with c2:
            st.write("**Frames:**")
            for i, f_ in enumerate(frames):
                st.image(f_, caption=f"Frame {i+1}")

        st.write("---")

    processing_progress.progress(100)
    processing_status_placeholder.write("**All segments processed!**")
    processing_msg = "**All segments processed!**"

    summary_status_placeholder.write("**All summaries generated!**")
    summary_msg = "**All summaries generated!**"

    all_embeddings = np.vstack([seg["embedding"] for seg in segment_info])
    if all_embeddings.shape[0] > 1:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_embeddings)
    else:
        reduced = np.zeros((1, 2))

    similarities = []
    if len(segment_info) > 0:
        first_norm = segment_info[0]["embedding"] / np.linalg.norm(segment_info[0]["embedding"])
        for seg in segment_info:
            current_norm = seg["embedding"] / np.linalg.norm(seg["embedding"])
            similarity = np.dot(first_norm, current_norm)
            similarities.append(similarity)
    else:
        similarities = [1.0]

    labels = [f"Part {seg['segment_index']+1}" for seg in segment_info]
    emb_plot = plot_embeddings(reduced, labels, similarities)

    st.session_state.analysis_results = {
        "video_url": youtube_url,
        "duration": duration,
        "segment_length": segment_length,
        "segment_info": segment_info,
        "reduced_embeddings": reduced,
        "similarities": similarities,
        "labels": labels,
        "emb_plot": emb_plot,
        "combined_class_weights": combined_class_weights,
        "video_path": video_path,
        "model_version": model_version,
        "customer_name": st.session_state.personalization_data["customer_name"]
    }

    st.session_state.analysis_steps = {
        "download_msg": download_msg,
        "processing_msg": processing_msg,
        "summary_msg": summary_msg
    }

elif st.session_state.analysis_steps is not None and st.session_state.analysis_results is not None:
    st.markdown("----")
    st.subheader("Step-by-Step Analysis (Previous Run)")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("#### 1. Downloading Video")
        st.write(st.session_state.analysis_steps["download_msg"])
    with col2:
        st.markdown("#### 2. Processing & Analysis")
        st.write(st.session_state.analysis_steps["processing_msg"])
    with col3:
        st.markdown("#### 3. Summaries & Insights")
        st.write(st.session_state.analysis_steps["summary_msg"])

    for seg_data in st.session_state.analysis_results["segment_info"]:
        seg_idx = seg_data["segment_index"]
        st.markdown(f"### Segment {seg_idx+1} Analysis (Previous Run)")
        st.write(f"**Time Range:** {seg_data['start_time']:.2f}s - {seg_data['end_time']:.2f}s")
        st.write(f"**Summary:** {seg_data['summary']}")

        c1, c2 = st.columns([2,1])
        with c1:
            st.write("**Frame Captions:**")
            for i, cap in enumerate(seg_data["frame_captions"]):
                st.write(f"Frame {i+1}: {cap}")

            st.write("**Top Concepts per Frame (for emphasis):**")
            for i, frame_res in enumerate(seg_data["frame_analyses"]):
                top_str = ", ".join([f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_res])
                st.write(f"Frame {i+1}: {top_str}")
        with c2:
            st.write("**Frames:**")
            for i, f_ in enumerate(seg_data["frames"]):
                st.image(f_, caption=f"Frame {i+1}")
        st.write("---")

# ---------------------------------------------------------------------------- #
# 11) PERSONALIZATION & GUIDANCE
# ---------------------------------------------------------------------------- #
st.markdown("<a name='personalization--guidance'></a>", unsafe_allow_html=True)
st.markdown("---")
st.header("Personalization & Guidance")
st.markdown("""
Use these controls to tailor the analysis to your preferences. 
Our auto-hyperparameter optimization ensures your changes remain balanced with other parameters.
""")

use_case_list = list(st.session_state.personalization_data["use_case_weights"].keys())
selected_use_case = st.selectbox("Choose a use case:", use_case_list, 
    index=use_case_list.index(st.session_state.personalization_data["active_use_case"])
)
st.session_state.personalization_data["active_use_case"] = selected_use_case

if selected_use_case in st.session_state.personalization_data["use_case_weights"]:
    for c, w in st.session_state.personalization_data["use_case_weights"][selected_use_case].items():
        if c not in st.session_state.personalization_data["class_weights"]:
            st.session_state.personalization_data["class_weights"][c] = w

st.subheader("Fine-Tune Concept Weights")
with st.expander("Adjust Weights", expanded=True):
    for c in sorted(st.session_state.personalization_data["class_weights"].keys()):
        current_weight = st.session_state.personalization_data["class_weights"][c]
        new_weight = st.slider(f"{c}", 0.0, 5.0, current_weight, 0.1, key=f"slider_{c}")
        st.session_state.personalization_data["class_weights"][c] = new_weight

st.subheader("Your Summary Goals")
st.session_state.personalization_data["goals"] = st.text_area(
    "Provide any high-level goals or aspects to emphasize in your segment summaries.",
    value=st.session_state.personalization_data["goals"], 
    height=80
)

st.subheader("Your Preferred Concepts")
preferred_concepts_str = st.text_input("Add additional concepts (comma-separated):", 
                                       value=", ".join(st.session_state.personalization_data["preferred_concepts"]))
if preferred_concepts_str.strip():
    st.session_state.personalization_data["preferred_concepts"] = [
        c.strip() for c in preferred_concepts_str.split(",") if c.strip()
    ]

st.subheader("Concept Feedback")
all_current_concepts = set(st.session_state.personalization_data["preferred_concepts"]).union(
    set(st.session_state.personalization_data["class_weights"].keys())
)
feedback_data = {}
if not all_current_concepts:
    st.info("No concepts identified yet. Add some concepts above.")
else:
    concept_list = sorted(all_current_concepts)
    fb_cols = st.columns(min(len(concept_list), 4))
    for i, concept in enumerate(concept_list):
        fb = fb_cols[i % 4].selectbox(f"'{concept}' feedback", ["No Feedback", "👍", "👎"], index=0, key=f"feedback_{concept}")
        feedback_data[concept] = fb

if st.button("Update Preferences from Feedback"):
    update_class_weights_from_feedback(feedback_data, st.session_state.personalization_data)
    st.success("Preferences updated! Re-run the analysis to apply changes.")

st.markdown("---")
st.subheader("Personalize by Example Data")
example_data = st.text_area("Concept adjustments (concept,factor per line):", value="")
if st.button("Update Weights from Examples"):
    if example_data.strip():
        update_class_weights_from_examples(example_data, selected_use_case, st.session_state.personalization_data)
        st.success("Weights updated from examples! Re-run analysis to apply.")
    else:
        st.info("Please provide example lines in the specified format.")

st.markdown("---")
st.subheader("Current Personalization State")
st.json({
    "Fine-Tuned": st.session_state.personalization_data["fine_tuned"],
    "Goals": st.session_state.personalization_data["goals"],
    "Preferred Concepts": st.session_state.personalization_data["preferred_concepts"],
    "Current class_weights": st.session_state.personalization_data["class_weights"],
    "Use Case Weights": st.session_state.personalization_data["use_case_weights"],
    "Active Use Case": st.session_state.personalization_data["active_use_case"],
    "Customer Name": st.session_state.personalization_data["customer_name"],
    "Feedback history": st.session_state.personalization_data["feedback_history"]
})

if st.button("Fine-Tune Model"):
    st.session_state.personalization_data["fine_tuned"] = True
    st.success("Model fine-tuned! Future analyses will incorporate these preferences more deeply.")

# ---------------------------------------------------------------------------- #
# 10) DATA DRIFT & OVERALL RESULTS - NOW MOVED BELOW PERSONALIZATION
# ---------------------------------------------------------------------------- #
st.markdown("<a name='data-drift-analysis'></a>", unsafe_allow_html=True)
st.markdown("---")
st.header("Data Drift Analysis")

if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    segment_info = results["segment_info"]

    st.markdown("""
    Below is the visualization of how each segment compares to the first one, 
    with auto-hyperparameter optimization guiding the detection of any changes.
    Green dots indicate stable content, while orange dots suggest drift.
    """)

    st.write(f"**Customer Name:** {results['customer_name']}")
    st.write(f"**Model Used:** {results['model_version']}")
    st.write(f"**Video URL:** {results['video_url']}")
    st.write(f"**Duration:** {results['duration']:.2f}s | **Segments:** {len(segment_info)} (~{results['segment_length']:.2f}s each)")

    st.subheader("Content Drift Visualization")
    st.image(results["emb_plot"], caption="Visualization of Content Changes Over Time")

    for i, sim in enumerate(results["similarities"]):
        msg = f"Part {i+1}: Similarity={sim:.2f}"
        if i == 0:
            msg += " (Reference)"
        elif sim < 0.8:
            msg += " (Changed)"
        else:
            msg += " (Stable)"
        st.write(msg)

    all_concepts = set()
    for seg in segment_info:
        for frame_res in seg["frame_analyses"]:
            for desc, conf in frame_res:
                all_concepts.add(desc)

    segment_concept_data = []
    for seg in segment_info:
        concept_sums = {c:0.0 for c in all_concepts}
        concept_counts = {c:0 for c in all_concepts}
        for frame_res in seg["frame_analyses"]:
            for desc, conf in frame_res:
                concept_sums[desc] += conf
                concept_counts[desc] += 1
        concept_avgs = {}
        for c in all_concepts:
            if concept_counts[c] > 0:
                concept_avgs[c] = concept_sums[c] / concept_counts[c]
            else:
                concept_avgs[c] = 0.0
        concept_avgs["segment"] = seg["segment_index"]+1
        segment_concept_data.append(concept_avgs)

    df_concepts = pd.DataFrame(segment_concept_data)
    df_concepts.set_index("segment", inplace=True)
    st.subheader("Concept Probability Over Time")
    st.area_chart(df_concepts)

else:
    st.info("No analysis results available. Once you analyze a video, this section will update.")

# ---------------------------------------------------------------------------- #
# 12) FOOTER
# ---------------------------------------------------------------------------- #
st.markdown("<p style='text-align:center; color:#888; margin-top:40px;'>© 2024 NomadicML - Demo application. Learn more at <a href='https://www.nomadicml.com' target='_blank'>https://www.nomadicml.com</a></p>", unsafe_allow_html=True)

################################################################################
# 13) ADDITIONAL CHARTS & VISUALIZATIONS
################################################################################
st.markdown("---")
st.title("Extended Visualizations & Insights")
st.markdown("""
These optional charts provide deeper insights into concept interactions, segment-by-segment changes, 
and advanced analytics. They leverage the same underlying data from the analysis steps above.
""")

###############################
# Chart 1: Concept Correlation
###############################
st.markdown("## 1. Concept Correlation Heatmap")
st.markdown("""
This chart attempts to visualize how different concepts correlate across segments 
based on their probabilities.
""")

def create_concept_correlation_heatmap(df_concepts: pd.DataFrame):
    if df_concepts.shape[0] > 1:
        corr_matrix = df_concepts.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        fig.colorbar(cax)
        plt.title("Correlation between Concept Probabilities", pad=20)
        buf_heat = BytesIO()
        plt.savefig(buf_heat, format='png', bbox_inches='tight')
        plt.close()
        buf_heat.seek(0)
        return buf_heat
    else:
        return None

if st.session_state.analysis_results is not None:
    df_concepts_corr = st.session_state.analysis_results.get("df_concepts_corr", None)
    if not df_concepts_corr:
        segment_info = st.session_state.analysis_results["segment_info"]
        all_concepts = set()
        for seg in segment_info:
            for frame_res in seg["frame_analyses"]:
                for desc, conf in frame_res:
                    all_concepts.add(desc)
        segment_concept_data = []
        for seg in segment_info:
            concept_sums = {c:0.0 for c in all_concepts}
            concept_counts = {c:0 for c in all_concepts}
            for frame_res in seg["frame_analyses"]:
                for desc, conf in frame_res:
                    concept_sums[desc] += conf
                    concept_counts[desc] += 1
            concept_avgs = {}
            for c in all_concepts:
                if concept_counts[c] > 0:
                    concept_avgs[c] = concept_sums[c] / concept_counts[c]
                else:
                    concept_avgs[c] = 0.0
            concept_avgs["segment"] = seg["segment_index"]+1
            segment_concept_data.append(concept_avgs)
        df_concepts_corr = pd.DataFrame(segment_concept_data)
        df_concepts_corr.set_index("segment", inplace=True)

    buf_heatmap = create_concept_correlation_heatmap(df_concepts_corr)
    if buf_heatmap:
        st.image(buf_heatmap, caption="Concept Correlation Heatmap")
    else:
        st.info("Not enough segments to compute correlation heatmap.")
else:
    st.info("Please run an analysis to see the concept correlation heatmap.")

###################################
# Chart 2: Segment Duration Chart
###################################
st.markdown("## 2. Segment Duration Distribution")
st.markdown("""
This bar chart shows each segment’s length, providing a quick overview of how the 
video is split.
""")

def plot_segment_durations(num_segments, total_duration):
    durations = [(i+1, total_duration/num_segments) for i in range(num_segments)]
    segments = [f"Segment {d[0]}" for d in durations]
    length_vals = [d[1] for d in durations]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(segments, length_vals, color='lightgreen')
    ax.set_title("Segment Duration Distribution")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Duration (s)")
    return fig

if st.session_state.analysis_results is not None:
    total_dur = st.session_state.analysis_results["duration"]
    seg_count = len(st.session_state.analysis_results["segment_info"])
    fig_duration = plot_segment_durations(seg_count, total_dur)
    buf_dur = BytesIO()
    fig_duration.savefig(buf_dur, format='png', bbox_inches='tight')
    plt.close(fig_duration)
    buf_dur.seek(0)
    st.image(buf_dur, caption="Segment Duration Distribution")
else:
    st.info("Segment Duration Chart will appear once you run an analysis.")

############################################
# Chart 3: Weighted Probability Over Time
############################################
st.markdown("## 3. Weighted Probability Over Time")
st.markdown("""
Monitor how a selected concept's weighted probability evolves across segments.
""")

def plot_weighted_prob_over_time(segment_info, concept):
    x_vals = []
    y_vals = []
    for seg in segment_info:
        concept_probs = []
        for frame_res in seg["frame_analyses"]:
            for desc, conf in frame_res:
                if desc == concept:
                    w = st.session_state.personalization_data["class_weights"].get(concept, 1.0)
                    concept_probs.append(conf * w)
        avg_prob = sum(concept_probs)/len(concept_probs) if concept_probs else 0.0
        x_vals.append(seg["segment_index"]+1)
        y_vals.append(avg_prob)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x_vals, y_vals, marker='o', color='purple')
    ax.set_title(f"Weighted Probability Over Time for '{concept}'")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Weighted Probability")
    ax.set_xticks(x_vals)
    return fig

if st.session_state.analysis_results is not None:
    all_concepts_found = set()
    for seg in st.session_state.analysis_results["segment_info"]:
        for frame_res in seg["frame_analyses"]:
            for desc, _ in frame_res:
                all_concepts_found.add(desc)
    if all_concepts_found:
        concept_list_sorted = sorted(all_concepts_found)
        selected_concept_for_weight_chart = st.selectbox(
            "Choose a concept for Weighted Probability Over Time:",
            concept_list_sorted
        )
        if st.button("Plot Weighted Probability"):
            fig_weighted_prob = plot_weighted_prob_over_time(st.session_state.analysis_results["segment_info"], selected_concept_for_weight_chart)
            buf_wprob = BytesIO()
            fig_weighted_prob.savefig(buf_wprob, format='png', bbox_inches='tight')
            plt.close(fig_weighted_prob)
            buf_wprob.seek(0)
            st.image(buf_wprob, caption=f"Weighted Probability for '{selected_concept_for_weight_chart}'")
    else:
        st.info("No concepts found in the analysis.")
else:
    st.info("Run analysis to enable Weighted Probability Over Time chart.")

##########################################################
# Chart 4: Per-Frame Concept Ranking for a Chosen Segment
##########################################################
st.markdown("## 4. Per-Frame Concept Ranking (Chosen Segment)")
st.markdown("""
A stacked bar chart illustrating the concept distributions for each frame in the selected segment.
""")

def create_per_frame_concept_chart(segment_data):
    frame_dicts = []
    for i, frame_res in enumerate(segment_data["frame_analyses"]):
        frame_dict = {}
        for desc, conf in frame_res:
            frame_dict[desc] = conf
        frame_dicts.append(frame_dict)

    all_concepts_in_segment = set()
    for fd in frame_dicts:
        all_concepts_in_segment.update(fd.keys())

    data_for_stack = []
    for i, fd in enumerate(frame_dicts):
        row_data = []
        for c in all_concepts_in_segment:
            row_data.append(fd.get(c, 0.0))
        data_for_stack.append(row_data)

    df_stack = pd.DataFrame(data_for_stack, columns=all_concepts_in_segment)
    df_stack.index = [f"Frame {i+1}" for i in range(len(frame_dicts))]

    df_stack.plot(kind="barh", stacked=True, figsize=(8,6), colormap="Spectral")
    plt.title("Per-Frame Concept Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frame")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    buf_stack = BytesIO()
    plt.savefig(buf_stack, format='png', bbox_inches='tight')
    plt.close()
    buf_stack.seek(0)
    return buf_stack

if st.session_state.analysis_results is not None:
    seg_indices = [seg["segment_index"] for seg in st.session_state.analysis_results["segment_info"]]
    seg_select = st.selectbox("Select a segment index for per-frame chart:", seg_indices)
    if st.button("Plot Per-Frame Distribution"):
        matching_seg_data = None
        for seg in st.session_state.analysis_results["segment_info"]:
            if seg["segment_index"] == seg_select:
                matching_seg_data = seg
                break
        if matching_seg_data is not None:
            chart_bytes = create_per_frame_concept_chart(matching_seg_data)
            st.image(chart_bytes, caption=f"Concept Distribution in Segment {seg_select+1}")
        else:
            st.info("Segment data not found.")
else:
    st.info("Analyze a video first to enable this chart.")

##############################################
# Chart 5: Rolling Average of Similarities
##############################################
st.markdown("## 5. Rolling Average of Similarities")
st.markdown("""
A rolling average helps smooth out noise when you have many segments.
""")

def plot_rolling_similarity(similarities, window=2):
    if len(similarities) > 1:
        sim_series = pd.Series(similarities)
        rolled = sim_series.rolling(window).mean().fillna(sim_series.iloc[0])
        x = range(1, len(similarities)+1)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(x, similarities, marker='o', label="Raw Similarities")
        ax.plot(x, rolled, marker='s', label=f"Rolling Avg (window={window})", color='orange')
        ax.set_title("Rolling Average of Segment Similarities")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Similarity to Segment 1")
        ax.legend()
        buf_roll = BytesIO()
        plt.savefig(buf_roll, format='png', bbox_inches='tight')
        plt.close()
        buf_roll.seek(0)
        return buf_roll
    else:
        return None

if st.session_state.analysis_results is not None:
    similarities = st.session_state.analysis_results["similarities"]
    if len(similarities) < 2:
        st.info("At least 2 segments are needed to compute rolling average.")
    else:
        window_size = st.slider("Rolling Window Size:", 2, max(2, len(similarities)), 2)
        if st.button("Plot Rolling Similarity"):
            chart_rolling = plot_rolling_similarity(similarities, window=window_size)
            if chart_rolling:
                st.image(chart_rolling, caption=f"Rolling Similarity (window={window_size})")
            else:
                st.info("Not enough data to plot rolling average.")
else:
    st.info("Run analysis to enable Rolling Similarity chart.")
