import numpy as np
import cv2
import torch
import faiss
import json
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from project import config

device = torch.device("cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device) # Neural network for computing embeddings
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # Preprocessor for images and text
clip_model.eval()

def get_video_duration_and_fps(cap):
    """Get duration and fps from a video"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 0:
        duration = 0.0
        fps = 0.0
    else:
        duration = n_frames / fps
    return duration, fps

def sliding_windows(duration_sec, window_sec, stride_sec):
    """Generate sliding windows over the video duration"""
    t = 0.0
    output = [] # List of (start_sec, end_sec)
    while t + window_sec <= duration_sec:
        output.append((t, t + window_sec))
        t += stride_sec
    return output

def extract_video_metadata(video_path: Path):
    """
    Expected structure:
      project/project_data/YouCookII_downscaled/videos/<split>/<recipe_id>/<youtube_id>.mp4
    """
    rel = video_path.relative_to(config.VIDEO_ROOT)  # <split>/<recipe_id>/<file>.mp4
    split = rel.parts[0] if len(rel.parts) > 0 else "unknown"
    recipe_id = rel.parts[1] if len(rel.parts) > 1 else "unknown"
    youtube_id = video_path.stem # filename without extension
    return split, recipe_id, youtube_id

@torch.no_grad()
def embed_clip(cap, fps, start_sec, end_sec, n_samples):
    # Sample n uniformly spaced timestamps strictly inside the clip interval
    times = np.linspace(start_sec, end_sec, n_samples + 2)[1:-1]

    frames = []

    for time in times: # Extract frames at the specified timestamps
        frame_idx = int(time * fps) # Convert time in seconds to frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # Capture the frame at the specified index

        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Convert image from OpenCV's BGR format to standard RGB
        frames.append(frame_rgb)

    if len(frames) == 0:
        return None

    inputs = clip_processor(images=frames, return_tensors="pt") # Preprocess frames for CLIP
    inputs = {k: v.to(device) for k, v in inputs.items()} # Move all input tensors to the target device

    # Compute CLIP embeddings for all frames
    img_feats = clip_model.get_image_features(**inputs)  # (N, D)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    # Average all frame embeddings into a single clip representation
    clip_feat = img_feats.mean(dim=0)  # (D,)
    clip_feat = clip_feat / clip_feat.norm()

    return clip_feat.cpu().numpy().astype(np.float32)

@torch.no_grad()
def embed_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    txt = clip_model.get_text_features(**inputs)  # (1, D)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy().astype(np.float32)

def collect_videos(video_root: Path, splits):
    """Collect all .mp4 video paths"""
    videos = []
    for split in splits:
        split_dir = Path(str(video_root) + "/" + split)

        found = list(split_dir.rglob("*.mp4"))
        print(f"{split}: found {len(found)} videos")
        videos.extend(found)

    return [str(p) for p in videos]

def build_index(video_paths):
    metadata = []
    clip_index = faiss.IndexFlatIP(clip_model.config.projection_dim) # Create a FAISS index for inner product similarity search

    for video_path in video_paths:
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path)) # Open video file
        if not cap.isOpened():
            print(f"[WARNING] Cannot open {video_path}")
            continue

        try:
            duration, fps = get_video_duration_and_fps(cap)
            windows = sliding_windows(duration, config.WINDOW_SEC, config.STRIDE_SEC)
            split, recipe_id, youtube_id = extract_video_metadata(video_path)

            for (start, end) in windows: # Embed each clip (window)
                feat = embed_clip(cap, fps, start, end, config.N_SAMPLES)
                if feat is None:
                    continue

                # FAISS expects (n, d)
                clip_index.add(feat.reshape(1, -1)) # Add clip embedding to the FAISS index

                # Store metadata for each clip in the same order as they are added to the index
                metadata.append({
                    "video_path": str(video_path),
                    "split": split,
                    "recipe_id": recipe_id,
                    "youtube_id": youtube_id,
                    "start_sec": float(start),
                    "end_sec": float(end),
                })
        finally:
            cap.release() # Ensure video file is closed after processing

    return clip_index, metadata

def retrieve(index, metadata, query, topk):
    q = embed_text(query)  # (1, D)
    scores, ids = index.search(q, topk)

    results = []
    for score, idx in zip(scores[0], ids[0]): # Retrieve metadata for the top-k results
        m = metadata[int(idx)]
        results.append({"score": float(score), **m})
    return results


if __name__ == "__main__":
    all_videos = collect_videos(config.VIDEO_ROOT, splits=("validation","testing"))
    print("Total videos found:", len(all_videos))

    # FOR DEBUGGING ONLY
    all_videos = all_videos[:5] # Limit to first 5 videos
    print("Using", len(all_videos), "videos")

    clip_index, metadata =  build_index(all_videos)
    print("Total clips embedded:", len(metadata))

    faiss.write_index(clip_index, config.FAISS_CLIP_INDEX_PATH)
    with open(config.CLIP_METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    print("Saved FAISS index and metadata")

    # Retrieval sanity check
    for q in ["cutting", "pan", "mixing", "knife", "bacon"]:
        res = retrieve(clip_index, metadata, q, config.TOP_K_VIDEO_CLIPS)
        print("\nQuery:", q)
        for r in res:
            print(r)
