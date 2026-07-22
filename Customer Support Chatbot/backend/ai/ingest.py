import os
import logging
import shutil
import hashlib

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_ATOMIC_LENGTH = 1200


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"instruction", "response", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def _build_documents(df: pd.DataFrame) -> list[Document]:
    """Each Q&A pair becomes one atomic document.
    Question placed first so short user queries align better with embeddings."""
    documents = []
    skipped = 0

    for idx, row in df.iterrows():
        question = str(row.get("instruction", "")).strip()
        answer = str(row.get("response", "")).strip()
        if not question or not answer:
            skipped += 1
            continue

        content = f"Customer Question: {question}\n\nSupport Answer: {answer}"

        metadata = {
            "category": row.get("category", "general"),
            "intent": row.get("intent", "unknown"),
            "question": question[:500],
            "source_row": int(idx),
        }
        documents.append(Document(page_content=content, metadata=metadata))

    if skipped:
        logger.info("Skipped %d empty rows", skipped)
    return documents


def _smart_split(documents: list[Document]) -> list[Document]:
    """Keep short Q&A pairs atomic. Only split the ~7%% that exceed MAX_ATOMIC_LENGTH."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_ATOMIC_LENGTH,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    result = []
    split_count = 0
    for doc in documents:
        if len(doc.page_content) <= MAX_ATOMIC_LENGTH:
            result.append(doc)
        else:
            result.extend(splitter.split_documents([doc]))
            split_count += 1

    if split_count:
        logger.info("Split %d long documents (>%d chars)", split_count, MAX_ATOMIC_LENGTH)
    return result


def ingest_data(reset: bool = False):
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {PROCESSED_DATA_PATH}")

    if reset and os.path.exists(CHROMA_DB_DIR):
        logger.info("Resetting ChromaDB at %s", CHROMA_DB_DIR)
        shutil.rmtree(CHROMA_DB_DIR)

    logger.info("Loading processed data from %s", PROCESSED_DATA_PATH)
    df = _load_csv(PROCESSED_DATA_PATH)
    logger.info("Loaded %d rows", len(df))

    documents = _build_documents(df)
    logger.info("Built %d documents", len(documents))

    chunks = _smart_split(documents)
    atomic = sum(1 for d in documents if len(d.page_content) <= MAX_ATOMIC_LENGTH)
    logger.info("Final: %d chunks (%d kept atomic, %d were split)", len(chunks), atomic, len(documents) - atomic)

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        if i == 0 and not os.path.exists(CHROMA_DB_DIR):
            Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR,
            )
        else:
            store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            store.add_documents(batch)
        logger.info("Ingested batch %d-%d / %d", i + 1, min(i + BATCH_SIZE, len(chunks)), len(chunks))

    logger.info("Ingestion complete — %d chunks stored in %s", len(chunks), CHROMA_DB_DIR)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Ingest processed data into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Delete existing ChromaDB before ingesting")
    args = parser.parse_args()

    ingest_data(reset=args.reset)

def ingest_single_file(file_path: str):
    """Ingests a single text file into ChromaDB."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # تحويل النص لـ Document
    doc = Document(page_content=content, metadata={"source": os.path.basename(file_path)})
    
    # تقسيم النص
    chunks = _smart_split([doc])
    
    # تحميل الـ Embeddings وإضافتها للـ DB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    store.add_documents(chunks)
    logger.info("Successfully ingested %s", file_path)