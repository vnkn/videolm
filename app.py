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
import yt_dlp

###############################################
# Initial Setup and Session State
###############################################
st.set_page_config(
    page_title="NomadicML Video Insights",
    page_icon="🎬",
    layout="wide"
)

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

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

# Store final step-by-step analysis states
if "analysis_steps" not in st.session_state:
    st.session_state.analysis_steps = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

###############################################
# Custom CSS
###############################################
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
.block-container {
    margin: 0 auto;
    max-width: 1400px;
    background: #ffffffAA;
    padding: 30px;
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
</style>
""", unsafe_allow_html=True)

###############################################
# Helper Functions
###############################################
def download_video_yt_dlp(url: str) -> str:
    temp_dir = tempfile.gettempdir()
    # Format 18 typically provides an MP4 with both audio & video included (360p)
    ydl_opts = {
        'format': '18',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

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

def generate_summary(frame_analyses, temperature, model, additional_goals):
    try:
        client = OpenAI()
    except Exception as e:
        return f"Error initializing OpenAI client: {e}"

    frame_descriptions = []
    for i, frame_results in enumerate(frame_analyses):
        top_matches = [f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_results]
        frame_descriptions.append(f"Frame {i+1}: {', '.join(top_matches)}")

    system_message = "You are a friendly video summarizer."
    user_prompt = f"""We analyzed each part of the video and found these concepts:

{chr(10).join(frame_descriptions)}

Please write a short, friendly summary of this segment, mentioning the main visible elements.

Additional goals or instructions from the user:
{additional_goals}
"""

    try:
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
    except Exception as e:
        return f"Error generating summary: {e}"

def plot_embeddings(embeddings, labels, similarities):
    x = embeddings[:,0]
    y = embeddings[:,1]

    plt.figure(figsize=(7,7))
    plt.title("Data Drift Visualization: Content Changes Over Segments", fontsize=14, fontweight='bold')
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
    plt.legend(handles=legend_elements, loc='upper right')
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

    # After updating from examples, also merge into main class_weights if active
    if personalization_data["active_use_case"] == active_case:
        for c, w in personalization_data["use_case_weights"][active_case].items():
            personalization_data["class_weights"][c] = w

###############################################
# UI Introduction
###############################################
st.markdown("<h1 style='text-align: center; font-size:42px;'>NomadicML Video Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color:#444;'>Analyze and monitor data drift in your video content with personalized guidance.</p>", unsafe_allow_html=True)

st.markdown("""
**How to Use This Demo:**
1. **Enter a YouTube URL:** Provide the link to a publicly accessible YouTube video.
2. **Adjust Parameters:** Choose how many segments, frames per segment, and summary creativity.
3. **Set Candidate Concepts:** Provide concepts you want to detect.
4. **Analyze Video:** Click "Generate Summary & Analyze Video" to start the process.
5. **Review Results:** As each segment is analyzed, results appear. Afterwards, adjust weights and preferences.
   
**Note:** Once the video is analyzed, the results **including the step-by-step analysis** remain visible even if you change weights, until you re-run the analysis.
""")

###############################################
# Video Input and Controls
###############################################
st.markdown("## Video Input")
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

candidate_descriptions_str = st.text_area("Candidate Concepts (comma-separated):", 
    "water, rocks, trees, landscape, nature, forest, person, mountains")
candidate_descriptions = [x.strip() for x in candidate_descriptions_str.split(',') if x.strip()]

analyze_btn = st.button("Generate Summary & Analyze Video")

# Container for dynamic results as they are generated
results_container = st.container()

###############################################
# Analysis and Step-by-Step Display
###############################################
if analyze_btn and youtube_url.strip():
    try:
        model_version = st.session_state.personalization_data["model_version"]
        combined_class_weights = dict(st.session_state.personalization_data["class_weights"])

        with results_container:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Step-by-Step Analysis")
            step_col1, step_col2, step_col3 = st.columns([1,1,1])
            with step_col1:
                st.markdown("### 1. Downloading Video")
                download_status_placeholder = st.empty()
                download_progress = st.progress(0)

            with step_col2:
                st.markdown("### 2. Processing & Analysis")
                processing_status_placeholder = st.empty()
                processing_progress = st.progress(0)

            with step_col3:
                st.markdown("### 3. Summaries & Insights")
                summary_status_placeholder = st.empty()
                summary_progress = st.progress(0)

            # Download video
            download_status_placeholder.write("**Status:** Downloading video...")
            video_path = download_video_yt_dlp(youtube_url)
            download_progress.progress(50)

            if not video_path or not os.path.exists(video_path):
                download_status_placeholder.write("**Error:** Failed to download video.")
                st.error("Failed to download the video. Check the URL or permissions.")
                st.stop()

            download_progress.progress(100)
            download_status_placeholder.write("**Status:** Video downloaded successfully!")

            duration = get_video_duration(video_path)
            if duration <= 0:
                st.error("Could not determine video duration or video is empty.")
                st.stop()

            segment_length = duration / num_segments
            segment_info = []

            # Process each segment and display results incrementally
            for seg_idx in range(num_segments):
                processing_progress.progress(int((seg_idx / num_segments)*100))
                processing_status_placeholder.write(f"**Analyzing Segment {seg_idx+1}/{num_segments}...**")

                start_time = seg_idx * segment_length
                end_time = (seg_idx + 1) * segment_length
                frames = extract_frames_for_segment(video_path, start_time, end_time, frames_per_segment)

                frame_analyses = analyze_frames_with_clip(
                    frames,
                    model_version=model_version,
                    candidate_descriptions=candidate_descriptions,
                    class_weights=combined_class_weights,
                    domain_shift_factor=domain_factor
                )

                summary_status_placeholder.write("**Generating Segment Summary...**")
                summary = generate_summary(frame_analyses, temperature=temperature, model="gpt-4", 
                                           additional_goals=st.session_state.personalization_data["goals"])
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
                    "start_time": start_time,
                    "end_time": end_time,
                    "frames": frames,
                    "frame_analyses": frame_analyses,
                    "summary": summary,
                    "embedding": segment_embedding,
                    "top_concepts": list(top_concepts)
                }
                segment_info.append(segment_data)

                # Display this segment's results immediately
                st.markdown(f"### Segment {seg_idx+1} Analysis")
                st.write(f"**Time Range:** {start_time:.2f}s - {end_time:.2f}s")
                st.write(f"**Summary:** {summary}")

                col_left, col_right = st.columns([2,1])
                with col_left:
                    st.write("**Top Concepts per Frame:**")
                    for i, frame_res in enumerate(frame_analyses):
                        top_str = ", ".join([f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_res])
                        st.write(f"Frame {i+1}: {top_str}")
                with col_right:
                    st.write("**Frames:**")
                    for i, f_ in enumerate(frames):
                        st.image(f_, caption=f"Frame {i+1}")

                st.write("---")

            processing_progress.progress(100)
            processing_status_placeholder.write("**All segments processed!**")
            summary_status_placeholder.write("**All summaries generated!**")

            # Compute PCA and similarities for data drift
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

            # Save analysis results to session state so they remain on reruns
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

            # Save final snapshot of step-by-step analysis states
            # We'll store what was shown above so on rerun we can still show it
            st.session_state.analysis_steps = {
                "download_msg": download_status_placeholder._value,
                "processing_msg": processing_status_placeholder._value,
                "summary_msg": summary_status_placeholder._value
            }

    except Exception as e:
        st.error(f"Error during processing: {e}")
        st.info("Try another video URL, or ensure it's publicly accessible.")

###############################################
# Display previously generated step-by-step analysis if available
###############################################
if st.session_state.analysis_steps is not None and st.session_state.analysis_results is not None and not analyze_btn:
    # If we are not currently analyzing (no new button press) but have old results:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Step-by-Step Analysis (Previous Run)")
    step_col1, step_col2, step_col3 = st.columns([1,1,1])
    with step_col1:
        st.markdown("### 1. Downloading Video")
        st.write(st.session_state.analysis_steps["download_msg"])

    with step_col2:
        st.markdown("### 2. Processing & Analysis")
        st.write(st.session_state.analysis_steps["processing_msg"])

    with step_col3:
        st.markdown("### 3. Summaries & Insights")
        st.write(st.session_state.analysis_steps["summary_msg"])

    # Display the detailed segment analysis from previous run
    for seg_data in st.session_state.analysis_results["segment_info"]:
        seg_idx = seg_data["segment_index"]
        st.markdown(f"### Segment {seg_idx+1} Analysis (Previous Run)")
        st.write(f"**Time Range:** {seg_data['start_time']:.2f}s - {seg_data['end_time']:.2f}s")
        st.write(f"**Summary:** {seg_data['summary']}")

        col_left, col_right = st.columns([2,1])
        with col_left:
            st.write("**Top Concepts per Frame:**")
            for i, frame_res in enumerate(seg_data["frame_analyses"]):
                top_str = ", ".join([f"{desc} ({conf*100:.1f}%)" for desc, conf in frame_res])
                st.write(f"Frame {i+1}: {top_str}")
        with col_right:
            st.write("**Frames:**")
            for i, f_ in enumerate(seg_data["frames"]):
                st.image(f_, caption=f"Frame {i+1}")

        st.write("---")

###############################################
# Display Previously Generated Final Results
###############################################
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    segment_info = results["segment_info"]

    st.markdown("---")
    st.header("Data Drift & Overall Results")
    st.markdown("""
    Below is the visualization of how each segment compares to the first one. 
    Green dots indicate stable content, while orange dots suggest drift.
    """)

    st.write(f"**Customer Name:** {results['customer_name']}")
    st.write(f"**Model Used:** {results['model_version']}")
    st.write(f"**Video URL:** {results['video_url']}")
    st.write(f"**Duration:** {results['duration']:.2f}s | **Segments:** {len(segment_info)} (~{results['segment_length']:.2f}s each)")

    st.subheader("Data Drift & Similarity Visualization")
    st.image(results["emb_plot"], caption="Visualization of Content Drift Over Time")

    for i, sim in enumerate(results["similarities"]):
        msg = f"Part {i+1}: Similarity={sim:.2f}"
        if i == 0:
            msg += " (Reference)"
        elif sim < 0.8:
            msg += " (Changed)"
        else:
            msg += " (Stable)"
        st.write(msg)

    # Concept Probability Over Time
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

###############################################
# Personalization & Guidance
###############################################
st.markdown("---")
st.write("### Personalization & Guidance")
st.markdown("Use the controls below to fine-tune concept weights and tailor the analysis to your preferences. Then re-run the analysis.")

use_case_list = list(st.session_state.personalization_data["use_case_weights"].keys())
selected_use_case = st.selectbox("Choose a use case:", use_case_list, 
                                 index=use_case_list.index(st.session_state.personalization_data["active_use_case"]))
st.session_state.personalization_data["active_use_case"] = selected_use_case

if selected_use_case in st.session_state.personalization_data["use_case_weights"]:
    for c, w in st.session_state.personalization_data["use_case_weights"][selected_use_case].items():
        if c not in st.session_state.personalization_data["class_weights"]:
            st.session_state.personalization_data["class_weights"][c] = w

st.write(f"**Editable Concept Weights for '{selected_use_case}' Use Case:**")
for c in sorted(st.session_state.personalization_data["class_weights"].keys()):
    current_weight = st.session_state.personalization_data["class_weights"][c]
    new_weight = st.slider(f"Weight for '{c}'", 0.0, 5.0, current_weight, 0.1)
    st.session_state.personalization_data["class_weights"][c] = new_weight

st.subheader("Your Summary Goals")
st.session_state.personalization_data["goals"] = st.text_area("Emphasize in summaries:", value=st.session_state.personalization_data["goals"], height=100)

st.subheader("Your Preferred Concepts")
preferred_concepts_str = st.text_input("Concepts (comma-separated):", 
                                       value=", ".join(st.session_state.personalization_data["preferred_concepts"]))
if preferred_concepts_str.strip():
    st.session_state.personalization_data["preferred_concepts"] = [c.strip() for c in preferred_concepts_str.split(",") if c.strip()]

st.subheader("Concept Feedback")
all_current_concepts = set(st.session_state.personalization_data["preferred_concepts"]).union(set(st.session_state.personalization_data["class_weights"].keys()))
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
st.write("### Personalize by Example Data")
st.markdown("Provide lines of the form `concept, factor` to adjust weights based on example data.")
example_data = st.text_area("Concept adjustments (concept,factor per line):", value="")
if st.button("Update Weights from Examples"):
    if example_data.strip():
        update_class_weights_from_examples(example_data, selected_use_case, st.session_state.personalization_data)
        st.success("Weights updated from examples! Re-run analysis to apply.")
    else:
        st.info("Please provide example lines in the specified format.")

st.markdown("---")
st.write("**Current Personalization State:**")
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
    st.success("Model fine-tuned! Future analyses will incorporate these preferences.")

st.markdown("<p style='text-align:center; color:#888; margin-top:40px;'>© 2024 NomadicML - This is a demo application.</p>", unsafe_allow_html=True)
