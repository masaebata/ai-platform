import hashlib
import os
from pathlib import Path
 
from pypdf import PdfReader
from qdrant_client import models
 
from common import (
    QDRANT_COLLECTION,
    embed_passages,
    get_embedding_model,
    get_qdrant_client,
)
 
 
DOCUMENT_DIR = Path(
    os.getenv(
        "DOCUMENT_DIR",
        "/documents",
    )
)
 
CHUNK_SIZE = int(
    os.getenv(
        "RAG_CHUNK_SIZE",
        "1200",
    )
)
 
CHUNK_OVERLAP = int(
    os.getenv(
        "RAG_CHUNK_OVERLAP",
        "200",
    )
)
 
 
def load_pdf(
    path: Path,
) -> list[dict]:
 
    reader = PdfReader(
        str(path)
    )
 
    documents = []
 
    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
 
        text = page.extract_text()
 
        if not text:
            continue
 
        documents.append(
            {
                "source": path.name,
                "path": str(path),
                "page": page_number,
                "text": text,
            }
        )
 
    return documents
 
 
def load_text_file(
    path: Path,
) -> list[dict]:
 
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )
 
    return [
        {
            "source": path.name,
            "path": str(path),
            "page": None,
            "text": text,
        }
    ]
 
 
def load_documents() -> list[dict]:
 
    documents = []
 
    for path in sorted(
        DOCUMENT_DIR.rglob("*")
    ):
 
        if not path.is_file():
            continue
 
        suffix = path.suffix.lower()
 
        if suffix == ".pdf":
            documents.extend(
                load_pdf(path)
            )
 
        elif suffix in (
            ".txt",
            ".md",
            ".markdown",
        ):
            documents.extend(
                load_text_file(path)
            )
 
    return documents
 
 
def chunk_text(
    text: str,
) -> list[str]:
 
    cleaned = (
        text
        .replace("\x00", " ")
        .strip()
    )
 
    if not cleaned:
        return []
 
    chunks = []
 
    start = 0
 
    while start < len(cleaned):
 
        end = min(
            start + CHUNK_SIZE,
            len(cleaned),
        )
 
        chunk = cleaned[
            start:end
        ].strip()
 
        if chunk:
            chunks.append(chunk)
 
        if end >= len(cleaned):
            break
 
        start = max(
            0,
            end - CHUNK_OVERLAP,
        )
 
    return chunks
 
 
def make_point_id(
    source: str,
    page: int | None,
    chunk_index: int,
    text: str,
) -> str:
 
    raw = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{text}"
    )
 
    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()
 
 
def ensure_collection() -> None:
 
    client = get_qdrant_client()
    model = get_embedding_model()
 
    vector_size = (
        model
        .get_sentence_embedding_dimension()
    )
 
    if client.collection_exists(
        QDRANT_COLLECTION
    ):
        return
 
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
 
    print(
        f"Created collection: "
        f"{QDRANT_COLLECTION}"
    )
 
 
def ingest() -> None:
 
    if not DOCUMENT_DIR.exists():
        raise RuntimeError(
            f"Document directory does not exist: "
            f"{DOCUMENT_DIR}"
        )
 
    ensure_collection()
 
    documents = load_documents()
 
    if not documents:
        print(
            f"No supported documents found in "
            f"{DOCUMENT_DIR}"
        )
        return
 
    client = get_qdrant_client()
 
    total_chunks = 0
 
    for document in documents:
 
        chunks = chunk_text(
            document["text"]
        )
 
        if not chunks:
            continue
 
        vectors = embed_passages(
            chunks
        )
 
        points = []
 
        for index, (
            chunk,
            vector,
        ) in enumerate(
            zip(
                chunks,
                vectors,
            )
        ):
 
            point_id = make_point_id(
                document["source"],
                document["page"],
                index,
                chunk,
            )
 
            payload = {
                "source": document["source"],
                "path": document["path"],
                "page": document["page"],
                "chunk_index": index,
                "text": chunk,
            }
 
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
 
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
 
        total_chunks += len(
            points
        )
 
        print(
            f"Ingested: "
            f"{document['source']} "
            f"page={document['page']} "
            f"chunks={len(points)}"
        )
 
    print(
        f"Completed. "
        f"Total chunks: {total_chunks}"
    )
 
 
if __name__ == "__main__":
    ingest()