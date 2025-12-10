import os
import uuid
import numpy as np
import chromadb
from typing import List, Dict, Any, Optional


class CHROMAVectorStore:
    """
    Vector Store is a SINK.
    It does NOT chunk, preprocess, or embed.
    It only stores worker outputs.
    """

    def __init__(
        self,
        collection_name: str = "legal_texts_pipeline",
        persistent_directory: str = "./src/vectorStore",
    ):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        print("[VectorStore] Initializing ChromaDB")

        os.makedirs(self.persistent_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persistent_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Embeddings produced by parallel workers"},
        )

        print(f"[VectorStore] Collection: {self.collection_name}")
        print(f"[VectorStore] Existing items: {self.collection.count()}")

    def load(self):
        print(f"[VectorStore] Loaded collection: {self.collection_name}")
        print(f"[VectorStore] Existing items: {self.collection.count()}")

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ):
        """
        Called by:
        - Workers directly
        - OR Aggregator

        All lists must be the same length.
        """

        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("texts, embeddings, and metadata length mismatch")

        if ids is None:
            ids = [f"chunk_{uuid.uuid4().hex}" for _ in texts]

        print(f"[VectorStore] Inserting {len(texts)} embeddings")

        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )

            print(f"[VectorStore] Insert success")
            print(f"[VectorStore] Total items: {self.collection.count()}")

        except Exception as e:
            print(f"[VectorStore] Insert failed: {e}")
            raise
