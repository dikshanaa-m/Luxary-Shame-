# backend/data_processor.py

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import config values
from config import (
    FAISS_INDEX_PATH,
    RAW_DATA_FILE_PATH
)

def process_and_embed_data_faiss():
    """Reads raw text, chunks it, creates embeddings, and saves the FAISS index."""

    # 1️⃣ Check if raw data file exists
    if not os.path.exists(RAW_DATA_FILE_PATH):
        raise FileNotFoundError(
            f"❌ Raw data file not found at {RAW_DATA_FILE_PATH}. "
            f"Please check your path in config.py or ensure the file exists."
        )

    # 2️⃣ Load data
    print(f"📄 Loading data from {RAW_DATA_FILE_PATH}...")
    loader = TextLoader(RAW_DATA_FILE_PATH)
    documents = loader.load()

    # 3️⃣ Split data into chunks
    print("🔹 Splitting data into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    if len(docs) == 0:
        raise ValueError("❌ No text chunks created. Please ensure raw.txt has valid text content.")

    print(f"✅ Data split into {len(docs)} chunks.")

    # 4️⃣ Initialize Sentence Transformer
    print("⚙️ Initializing Sentence Transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 5️⃣ Generate embeddings
    print("🧠 Generating embeddings for all chunks...")
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("❌ No embeddings generated. Check input data content.")

    # 6️⃣ Create FAISS index
    print("💾 Creating FAISS index...")
    dimension = embeddings.shape[1]
    import faiss
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"✅ FAISS index successfully saved to {FAISS_INDEX_PATH}!")

if __name__ == "__main__":
    process_and_embed_data_faiss()
