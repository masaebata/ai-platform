import io
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse
 
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from qdrant_client import models
 
from common import (
    QDRANT_COLLECTION,
    embed_passages,
    get_embedding_model,
    get_qdrant_client,
)
 
 
URL_LIST = Path(
    os.getenv(
        "URL_LIST",
        "/app/urls.txt",
    )
)
 
CHUNK_SIZE = int(
    os.getenv(
        "RAG_CHUNK_SIZE",
        "800",
    )
)
 
CHUNK_OVERLAP = int(
    os.getenv(
        "RAG_CHUNK_OVERLAP",
        "120",
    )
)
 
HTTP_TIMEOUT = float(
    os.getenv(
        "RAG_HTTP_TIMEOUT",
        "60",
    )
)
 
USER_AGENT = os.getenv(
    "RAG_USER_AGENT",
    "ai-platform-rag/1.0",
)
 
 
def load_urls() -> list[str]:
 
    if not URL_LIST.exists():
 
        raise RuntimeError(
            f"URL list does not exist: {URL_LIST}"
        )
 
    urls = []
 
    for line in URL_LIST.read_text(
        encoding="utf-8"
    ).splitlines():
 
        line = line.strip()
 
        if not line:
            continue
 
        if line.startswith("#"):
            continue
 
        if not (
            line.startswith("https://")
            or line.startswith("http://")
        ):
            print(
                f"Skip unsupported URL: {line}"
            )
            continue
 
        urls.append(line)
 
    return list(
        dict.fromkeys(urls)
    )
 
 
def clean_html(
    html: str,
    url: str,
) -> dict:
 
    soup = BeautifulSoup(
        html,
        "lxml",
    )
 
    title = None
 
    if soup.title:
        title = soup.title.get_text(
            " ",
            strip=True,
        )
 
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "aside",
        ]
    ):
        tag.decompose()
 
    content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )
 
    text = content.get_text(
        "\n",
        strip=True,
    )
 
    lines = []
 
    for line in text.splitlines():
 
        cleaned = " ".join(
            line.split()
        )
 
        if cleaned:
            lines.append(
                cleaned
            )
 
    return {
        "source_type": "html",
        "source_url": url,
        "title": title or url,
        "page": None,
        "text": "\n".join(lines),
    }
 
 
def load_pdf(
    content: bytes,
    url: str,
) -> list[dict]:
 
    reader = PdfReader(
        io.BytesIO(content)
    )
 
    title = None
 
    if reader.metadata:
        title = reader.metadata.title
 
    if not title:
 
        parsed = urlparse(url)
 
        title = (
            Path(parsed.path).name
            or url
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
                "source_type": "pdf",
                "source_url": url,
                "title": title,
                "page": page_number,
                "text": text.strip(),
            }
        )
 
    return documents
 
 
def fetch_url(
    url: str,
) -> list[dict]:
 
    print(
        f"Fetching: {url}"
    )
 
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,"
            "application/xhtml+xml,"
            "application/pdf,"
            "*/*"
        ),
    }
 
    with httpx.Client(
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
 
        response = client.get(
            url
        )
 
        response.raise_for_status()
 
    content_type = (
        response.headers
        .get(
            "content-type",
            "",
        )
        .lower()
    )
 
    final_url = str(
        response.url
    )
 
    parsed = urlparse(
        final_url
    )
 
    is_pdf = (
        "application/pdf"
        in content_type
        or parsed.path.lower().endswith(
            ".pdf"
        )
    )
 
    if is_pdf:
 
        return load_pdf(
            response.content,
            final_url,
        )
 
    return [
        clean_html(
            response.text,
            final_url,
        )
    ]
 
 
def chunk_text(
    text: str,
) -> list[str]:
 
    cleaned = (
        text
        .replace(
            "\x00",
            " ",
        )
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
 
            chunks.append(
                chunk
            )
 
        if end >= len(cleaned):
            break
 
        start = max(
            0,
            end - CHUNK_OVERLAP,
        )
 
    return chunks
 
 
def make_point_id(
    source_url: str,
    page: int | None,
    chunk_index: int,
    text: str,
) -> str:
 
    raw = (
        f"{source_url}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{text}"
    )
 
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            raw,
        )
    )
 
 
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
        "Created collection: "
        f"{QDRANT_COLLECTION}"
    )
 
 
def delete_existing_source(
    source_url: str,
) -> None:
 
    client = get_qdrant_client()
 
    source_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_url",
 
                match=models.MatchValue(
                    value=source_url
                ),
            )
        ]
    )
 
    client.delete(
        collection_name=QDRANT_COLLECTION,
 
        points_selector=models.FilterSelector(
            filter=source_filter
        ),
 
        wait=True,
    )
 
 
def ingest_document(
    document: dict,
) -> int:
 
    chunks = chunk_text(
        document["text"]
    )
 
    if not chunks:
        return 0
 
    vectors = embed_passages(
        chunks
    )
 
    client = get_qdrant_client()
 
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
            document["source_url"],
            document["page"],
            index,
            chunk,
        )
 
        payload = {
            "source_type":
                document["source_type"],
 
            "source_url":
                document["source_url"],
 
            "title":
                document["title"],
 
            "page":
                document["page"],
 
            "chunk_index":
                index,
 
            "text":
                chunk,
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
 
    return len(points)
 
 
def ingest() -> None:
 
    ensure_collection()
 
    urls = load_urls()
 
    if not urls:
 
        print(
            "No URLs found."
        )
 
        return
 
    total_chunks = 0
 
    for url in urls:
 
        try:
 
            documents = fetch_url(
                url
            )
 
            if not documents:
 
                print(
                    f"No content: {url}"
                )
 
                continue
 
            source_url = (
                documents[0][
                    "source_url"
                ]
            )
 
            delete_existing_source(
                source_url
            )
 
            url_chunks = 0
 
            for document in documents:
 
                count = ingest_document(
                    document
                )
 
                url_chunks += count
                total_chunks += count
 
                print(
                    "Indexed: "
                    f"{document['title']} "
                    f"page={document['page']} "
                    f"chunks={count}"
                )
 
            print(
                f"Completed URL: "
                f"{source_url} "
                f"chunks={url_chunks}"
            )
 
        except Exception as exc:
 
            print(
                f"FAILED: {url}"
            )
 
            print(
                f"Reason: {exc}"
            )
 
    print(
        "Ingestion completed. "
        f"Total chunks={total_chunks}"
    )
 
 
if __name__ == "__main__":
 
    ingest()