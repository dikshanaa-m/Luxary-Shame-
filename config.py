# backend/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# 1. API Keys and Credentials
# ==============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("FATAL ERROR: GEMINI_API_KEY is not set in the environment or the .env file.")

# ==============================================================================
# 2. AI Model Configuration
# ==============================================================================

# Model for RAG Embedding (Note: This is an embedding-specific model)
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# Model for Advice Generation, Image Analysis, and Grounding - UPDATED
ADVICE_MODEL_NAME = "gemini-2.5-flash-lite"
VISION_MODEL_NAME = "gemini-2.5-flash-lite"

# ==============================================================================
# 3. File Paths and RAG Configuration
# ==============================================================================

# Path to your collected raw reviews (used for building the FAISS index)
RAW_DATA_FILE_PATH = "data/rawdata/raw.txt"

# FAISS Vector Store Persistence Path
FAISS_INDEX_PATH = "vstore/faiss_index"

# Number of reviews to retrieve from FAISS for RAG context
RAG_K_REVIEWS = 10

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Additional paths for consistency
RAW_DATA_PATH = "data/rawdata/raw.txt"
REVIEWS_FILE = "data/reviews.json"

# RAG Configuration
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.7