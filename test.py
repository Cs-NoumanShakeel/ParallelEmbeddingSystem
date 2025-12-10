import numpy as np
from src.vectorstore import CHROMAVectorStore
from src.embedding import EmbeddingPipeline

# -------------------------
# Load vector store
# -------------------------
store = CHROMAVectorStore()
store.load()

print("\n[TEST] Total vectors in DB:", store.collection.count())

# -------------------------
# Generate query embedding
# -------------------------
emb = EmbeddingPipeline(load_model=True)

query = "insurance ordinance section"
query_embedding = emb.embed_texts([query])

print("\n[TEST] Query embedding shape:", query_embedding.shape)
print("[TEST] First 10 values of embedding:")
print(query_embedding[0][:10])   # ✅ real vector values
