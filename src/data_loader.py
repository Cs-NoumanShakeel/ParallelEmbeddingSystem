from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from typing import List


def load_docs(data_directory: str) -> List:
    """
    Loads PDFs and returns raw LangChain Documents.
    This is a CONTROLLER helper, not a pipeline stage.
    """

    data_path = Path(data_directory)
    print(f"[Controller] Loading PDFs from: {data_path}")

    all_documents = []
    pdf_files = list(data_path.glob("**/*.pdf"))

    print(f"[Controller] Found {len(pdf_files)} PDF(s)")

    for pdf_path in pdf_files:
        print(f"[Controller] Processing: {pdf_path.name}")

        try:
            loader = PyMuPDFLoader(str(pdf_path))
            documents = loader.load()

            # ✅ Attach dataset-level metadata
            for doc in documents:
                doc.metadata["source_pdf"] = pdf_path.name
                doc.metadata["source_path"] = str(pdf_path)
                doc.metadata["dataset_id"] = pdf_path.stem

            all_documents.extend(documents)
            print(f"[Controller] Loaded {len(documents)} pages")

        except Exception as e:
            print(f"[Controller] Failed to load {pdf_path.name}: {e}")

    print(f"[Controller] Total pages loaded: {len(all_documents)}")
    return all_documents
