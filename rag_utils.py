# backend/rag_utils.py
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH, RAW_DATA_FILE_PATH, RAG_K_REVIEWS

# Load the sentence-transformer model used for embeddings (same as data_processor)
ST_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_index = None
_reviews = None

def _ensure_model_loaded():
    global _model
    if _model is None:
        _model = SentenceTransformer(ST_MODEL_NAME)
    return _model

def _ensure_index_and_reviews_loaded():
    global _index, _reviews
    if _index is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Run data_processor.process_and_embed_data_faiss() first.")
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _reviews is None:
        if not os.path.exists(RAW_DATA_FILE_PATH):
            raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_FILE_PATH}.")
        with open(RAW_DATA_FILE_PATH, "r", encoding="utf-8") as f:
            _reviews = [line.strip() for line in f if line.strip()]
    return _index, _reviews

def search_reviews(query, top_k=None):
    """Return top N reviews from FAISS for hybrid selection."""
    if top_k is None:
        top_k = RAG_K_REVIEWS

    model = _ensure_model_loaded()
    index, reviews = _ensure_index_and_reviews_loaded()

    # create embedding (shape (1, dim))
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")

    # search more than top_k to give AI room to select
    search_k = max(top_k * 2, 10)
    D, I = index.search(q_emb, search_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(reviews):
            results.append(reviews[idx])
    return results  # return extra reviews; AI will select top 5 later
