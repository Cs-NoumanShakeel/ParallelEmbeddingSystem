from typing import List
import numpy as np
import os
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingPipeline:
    """
    This class can be used in TWO places:

    1) Controller (Producer)
       - only uses chunk_documents()

    2) Worker Nodes (Consumers)
       - uses preprocess_texts() + embed_texts()

    Chunking and embedding are intentionally decoupled.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        load_model: bool = True,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None

        if load_model:
            self._load_model()

    # ------------------------
    # Model Loading (Workers)
    # ------------------------
    def _load_model(self):
        load_dotenv()
        token = os.getenv("HF_API_TOKEN")

        print(f"[Worker] Loading embedding model: {self.model_name}")

        # HuggingFaceEmbeddings for local models
        # If you need API-based embeddings, use HuggingFaceInferenceAPIEmbeddings
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        if token:
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            # For local models without API token
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

        print("[Worker] Model loaded successfully")

    # ------------------------
    # Chunking (Controller)
    # ------------------------
    def chunk_documents(self, documents):
        """
        Used ONLY by the controller before sending data to RabbitMQ.
        Input: List[langchain Document]
        Output: List[langchain Document] (chunks)
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)
        print(f"[Controller] Split {len(documents)} docs into {len(chunks)} chunks")
        return chunks

    # ------------------------
    # Preprocessing (Workers)
    # ------------------------
    def preprocess_text(self, text: str) -> str:
        """
        Example:
        'Product: iPhone 15 !!! Price $134'
        -> 'product iphone 15 price 134'
        """

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove symbols
        text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
        return text

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        return [self.preprocess_text(t) for t in texts]

    # ------------------------
    # Embeddings (Workers)
    # ------------------------
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Input: List[str] (plain text from RabbitMQ)
        Output: numpy array (N x D)
        """

        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        print(f"[Worker] Generating embeddings for {len(texts)} chunks")

        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings)

        print(f"[Worker] Embedding shape: {embeddings.shape}")
        return embeddings