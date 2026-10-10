from pathlib import Path

# Database and dateset paths
DB_PATH = 'project/database/recipe.db'
CSV_PATH = 'project/project_data/full_dataset.csv'

# Recipe model settings
TITLE_BATCH_SIZE = 64
DIRECTIONS_BATCH_SIZE = 64

# Index paths for metadata
FAISS_TITLE_INDEX_PATH = "project/project_data/indexes/faiss_title.index"
FAISS_DIRECTIONS_INDEX_PATH = "project/project_data/indexes/faiss_directions.index"
RECIPE_IDS_PATH = "project/project_data/indexes/recipe_ids.npy"

# Retrieval settings
TOP_K_TITLE = 1
TOP_K_DIRECTIONS = 1
TOP_K_VIDEO_CLIPS = 1

# Video processing settings
VIDEO_ROOT = Path("project/project_data/YouCookII_downscaled/videos")
WINDOW_SEC = 5.0 # clip length in seconds
STRIDE_SEC = 2.5 # stride for sliding window
N_SAMPLES = 4 # number of frames to sample per clip

# Index path for video clips
FAISS_CLIP_INDEX_PATH = "project/project_data/indexes/clip_index.index"
CLIP_METADATA_PATH = "project/project_data/indexes/clip_metadata.json"

# Settings for chatbot
SYSTEM_PROMPT = "You are a helpful assistant that reformats recipes. Use ONLY the provided context exactly! Output ALWAYS ingredients first, then the numbered instructions steps! Do not include any extra commentary. Keep your response concise. Output ALWAYS the ingredients and the numbered instructions steps!"
MAX_CONTEXT_TOKENS = 2048
STOP_TOKENS = ["<|user|>", "<|system|>", "<|context|>"]
MAX_NEW_TOKENS = 550
TEMPERATURE = 0.2
USE_SAMPLING = True
TOP_P = 0.95