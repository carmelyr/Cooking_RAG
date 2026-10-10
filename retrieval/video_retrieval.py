import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, List
import faiss
from project import config

class VideoRetriever:
    def __init__(self, clip_index_path: str, clip_metadata_path: str):
        self.clip_index_path = clip_index_path
        self.clip_metadata_path = clip_metadata_path
    
    def _load_faiss_clips(self) -> faiss.Index:
        return faiss.read_index(self.clip_index_path)
    
    def _load_clip_metadata(self) -> Dict:
        with open(self.clip_metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    @torch.no_grad()
    def _embed_query(self, query: str): # Embed recipe instructions
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        inputs = clip_processor(
            text=[query], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77
        )
        #inputs = {k: v.to(device) for k, v in inputs.items()}
        txt = clip_model.get_text_features(**inputs) # CLIP text encoder to get instruction embedding (1, D)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return txt.cpu().numpy().astype(np.float32)
    
    def similarity_search(self, query:str, topk: int) -> List[Dict]:
        query_embedding = self._embed_query(query)
        scores, indicies = self._load_faiss_clips().search(query_embedding, topk)

        results = []
        for idx, score in zip(indicies[0], scores[0]):
            if idx == -1:
                continue
            clip_metadata = self._load_clip_metadata()[idx]
            clip_metadata['similarity_score'] = float(score) # Add similarity score to the clip metadata dict
            results.append(clip_metadata)

        return results
    
if __name__ == "__main__":
    retriever = VideoRetriever(
        clip_index_path= config.FAISS_CLIP_INDEX_PATH,
        clip_metadata_path= config.CLIP_METADATA_PATH
    )

    clips = retriever.similarity_search("How do i make onion rings?", config.TOP_K_VIDEO_CLIPS)
    print(clips)