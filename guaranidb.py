# -*- coding: utf-8 -*-
"""Create a vector database from a corpus in Guaraní."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Directory where this script is located.
BASE_DIR = Path(__file__).resolve().parent

# Input text file.
TXT_PATH = BASE_DIR / "data" / "guarani.txt"

# Local directory where the database will be stored.
PERSIST_DIR = BASE_DIR / "guarani_db"

COLLECTION_NAME = "guarani"

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# Sentence-Transformers model
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main():
    if not TXT_PATH.exists():
        raise FileNotFoundError(
            f"{TXT_PATH} does not exist. Place guarani.txt in the same directory "
            "as this script or adjust TXT_PATH."
        )

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    print("OK:", TXT_PATH)
    print("Will persist in:", PERSIST_DIR)

    loader = TextLoader(str(TXT_PATH), encoding="utf-8")
    docs = loader.load()

    print("Documents uploaded:", len(docs))
    print("Document characters:", len(docs[0].page_content))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    print("Chunks generated", len(chunks))
    print("Example chunk (first 20 characters):")
    print(chunks[0].page_content[:20])

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # Optional: sentence-transformers automatically uses the GPU if one is available.
        # Optional: normalizing embeddings may be useful depending on the metric or use case.
        # encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    print("Chroma created.")
    print("Collection:", COLLECTION_NAME)
    print("Persistent directory:", PERSIST_DIR)

    query = "Tercera persona del plural"
    results = vectorstore.similarity_search(query, k=6)

    for i, result in enumerate(results, 1):
        print(f"\n--- Results {i} ---")
        print(result.page_content[:3000])


if __name__ == "__main__":
    main()
