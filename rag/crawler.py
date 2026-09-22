import hashlib
import io
import json
import os
import time
import uuid
 
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser
 
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
 
 
SEED_FILE = Path(
    os.getenv(
        "CRAWL_SEED_FILE",
        "/app/seeds.txt",
    )
)
 
STATE_FILE = Path(
    os.getenv(
        "CRAWL_STATE_FILE",
        "/state/crawl_state.json",
    )
)
 
ALLOWED_DOMAIN = os.getenv(
    "CRAWL_ALLOWED_DOMAIN",
    "docs.paloaltonetworks.com",
)
 
MAX_DEPTH = int(
    os.getenv(
        "CRAWL_MAX_DEPTH",
        "3",
    )
)
 
MAX_PAGES = int(
    os.getenv(
        "CRAWL_MAX_PAGES",
        "300",
    )
)
 
CRAWL_DELAY = float(
    os.getenv(
        "CRAWL_DELAY",
        "1.0",
    )
)
 
HTTP_TIMEOUT = float(
    os.getenv(
        "RAG_HTTP_TIMEOUT",
        "60",
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
 
USER_AGENT = os.getenv(
    "RAG_USER_AGENT",
    "ai-platform-rag/1.0",
)
 
RESPECT_ROBOTS = (
    os.getenv(
        "CRAWL_RESPECT_ROBOTS",
        "true",
    ).lower()
    in (
        "true",
        "1",
        "yes",
    )
)
 
 
SKIP_PATH_PARTS = (
    "/search",
    "/login",
    "/logout",
    "/signin",
    "/register",
)
 
SKIP_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".css",
    ".js",
    ".ico",
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".exe",
    ".dmg",
    ".mp4",
    ".mp3",
    ".woff",
    ".woff2",
    ".ttf",
)
 
 
def load_seeds() -> list[str]:
 
    if not SEED_FILE.exists():
        raise RuntimeError(
            f"Seed file not found: {SEED_FILE}"
        )
 
    seeds = []
 
    for line in SEED_FILE.read_text(
        encoding="utf-8"
    ).splitlines():
 
        line = line.strip()
 
        if not line:
            continue
 
        if line.startswith("#"):
            continue
 
        seeds.append(
            normalize_url(line)
        )
 
    return list(
        dict.fromkeys(seeds)
    )
 
 
def load_state() -> dict:
 
    if not STATE_FILE.exists():
        return {}
 
    try:
        return json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )
 
    except Exception:
        return {}
 
 
def save_state(
    state: dict,
) -> None:
 
    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
 
    temporary = STATE_FILE.with_suffix(
        ".tmp"
    )
 
    temporary.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
 
    temporary.replace(
        STATE_FILE
    )
 
 
def normalize_url(
    url: str,
) -> str:
 
    parsed = urlparse(
        url
    )
 
    normalized = parsed._replace(
        fragment="",
    )
 
    result = urlunparse(
        normalized
    )
 
    if result.endswith("/"):
        result = result[:-1]
 
    return result
 
 
def is_allowed_url(
    url: str,
) -> bool:
 
    parsed = urlparse(
        url
    )
 
    if parsed.scheme not in (
        "http",
        "https",
    ):
        return False
 
    if (
        parsed.hostname
        != ALLOWED_DOMAIN
    ):
        return False
 
    lower_path = (
        parsed.path.lower()
    )
 
    if any(
        item in lower_path
        for item in SKIP_PATH_PARTS
    ):
        return False
 
    if lower_path.endswith(
        SKIP_EXTENSIONS
    ):
        return False
 
    return True
 
 
def create_robot_parser(
    client: httpx.Client,
) -> RobotFileParser | None:
 
    if not RESPECT_ROBOTS:
        return None
 
    robots_url = (
        f"https://{ALLOWED_DOMAIN}"
        "/robots.txt"
    )
 
    try:
 
        response = client.get(
            robots_url
        )
 
        if response.status_code != 200:
            print(
                "robots.txt unavailable; "
                "continuing conservatively."
            )
            return None
 
        parser = RobotFileParser()
 
        parser.set_url(
            robots_url
        )
 
        parser.parse(
            response.text.splitlines()
        )
 
        return parser
 
    except Exception as exc:
 
        print(
            f"robots.txt check failed: {exc}"
        )
 
        return None
 
 
def robots_allows(
    parser: RobotFileParser | None,
    url: str,
) -> bool:
 
    if parser is None:
        return True
 
    return parser.can_fetch(
        USER_AGENT,
        url,
    )
 
 
def extract_html(
    html: str,
    url: str,
) -> tuple[
    dict,
    list[str],
]:
 
    soup = BeautifulSoup(
        html,
        "lxml",
    )
 
    title = url
 
    if soup.title:
 
        title_text = (
            soup.title.get_text(
                " ",
                strip=True,
            )
        )
 
        if title_text:
            title = title_text
 
    links = []
 
    for anchor in soup.find_all(
        "a",
        href=True,
    ):
 
        href = anchor.get(
            "href"
        )
 
        if not href:
            continue
 
        absolute = normalize_url(
            urljoin(
                url,
                href,
            )
        )
 
        if is_allowed_url(
            absolute
        ):
 
            links.append(
                absolute
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
 
    document = {
        "source_type": "html",
        "source_url": url,
        "title": title,
        "page": None,
        "text": "\n".join(lines),
    }
 
    return (
        document,
        list(
            dict.fromkeys(
                links
            )
        ),
    )
 
 
def extract_pdf(
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
 
        title = (
            Path(
                urlparse(
                    url
                ).path
            ).name
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
 
        chunk = (
            cleaned[start:end]
            .strip()
        )
 
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
 
 
def content_hash(
    documents: list[dict],
) -> str:
 
    combined = "\n".join(
        document["text"]
        for document in documents
    )
 
    return hashlib.sha256(
        combined.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()
 
 
def ensure_collection() -> None:
 
    client = (
        get_qdrant_client()
    )
 
    model = (
        get_embedding_model()
    )
 
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
 
 
def delete_existing_source(
    source_url: str,
) -> None:
 
    client = (
        get_qdrant_client()
    )
 
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
 
 
def index_documents(
    documents: list[dict],
) -> int:
 
    client = (
        get_qdrant_client()
    )
 
    total = 0
 
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
 
            payload = {
                "source_type":
                    document[
                        "source_type"
                    ],
 
                "source_url":
                    document[
                        "source_url"
                    ],
 
                "title":
                    document[
                        "title"
                    ],
 
                "page":
                    document[
                        "page"
                    ],
 
                "chunk_index":
                    index,
 
                "text":
                    chunk,
            }
 
            points.append(
                models.PointStruct(
                    id=make_point_id(
                        document[
                            "source_url"
                        ],
                        document[
                            "page"
                        ],
                        index,
                        chunk,
                    ),
                    vector=vector,
                    payload=payload,
                )
            )
 
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )
 
        total += len(
            points
        )
 
    return total
 
 
def crawl() -> None:
 
    ensure_collection()
 
    seeds = load_seeds()
 
    state = load_state()
 
    queue = deque(
        (
            seed,
            0,
        )
        for seed in seeds
    )
 
    visited = set()
 
    processed = 0
    changed = 0
    skipped = 0
    failed = 0
 
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
 
        robot_parser = (
            create_robot_parser(
                client
            )
        )
 
        while (
            queue
            and processed < MAX_PAGES
        ):
 
            url, depth = (
                queue.popleft()
            )
 
            url = normalize_url(
                url
            )
 
            if url in visited:
                continue
 
            visited.add(
                url
            )
 
            if not is_allowed_url(
                url
            ):
                continue
 
            if depth > MAX_DEPTH:
                continue
 
            if not robots_allows(
                robot_parser,
                url,
            ):
 
                print(
                    f"ROBOTS SKIP: {url}"
                )
 
                continue
 
            print(
                f"[{processed + 1}/"
                f"{MAX_PAGES}] "
                f"depth={depth} "
                f"Fetching: {url}"
            )
 
            try:
 
                response = client.get(
                    url
                )
 
                response.raise_for_status()
 
                final_url = normalize_url(
                    str(
                        response.url
                    )
                )
 
                content_type = (
                    response.headers
                    .get(
                        "content-type",
                        "",
                    )
                    .lower()
                )
 
                is_pdf = (
                    "application/pdf"
                    in content_type
                    or urlparse(
                        final_url
                    ).path.lower().endswith(
                        ".pdf"
                    )
                )
 
                discovered_links = []
 
                if is_pdf:
 
                    documents = (
                        extract_pdf(
                            response.content,
                            final_url,
                        )
                    )
 
                else:
 
                    document, (
                        discovered_links
                    ) = extract_html(
                        response.text,
                        final_url,
                    )
 
                    documents = [
                        document
                    ]
 
                if (
                    not documents
                    or not any(
                        item["text"].strip()
                        for item
                        in documents
                    )
                ):
 
                    print(
                        f"EMPTY: {final_url}"
                    )
 
                    processed += 1
 
                    continue
 
                digest = content_hash(
                    documents
                )
 
                previous = state.get(
                    final_url,
                    {}
                )
 
                if (
                    previous.get(
                        "sha256"
                    )
                    == digest
                ):
 
                    print(
                        f"UNCHANGED: "
                        f"{final_url}"
                    )
 
                    skipped += 1
 
                else:
 
                    delete_existing_source(
                        final_url
                    )
 
                    chunk_count = (
                        index_documents(
                            documents
                        )
                    )
 
                    state[
                        final_url
                    ] = {
                        "sha256":
                            digest,
 
                        "title":
                            documents[
                                0
                            ][
                                "title"
                            ],
 
                        "updated_at":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
 
                        "chunks":
                            chunk_count,
                    }
 
                    save_state(
                        state
                    )
 
                    changed += 1
 
                    print(
                        "INDEXED: "
                        f"{final_url} "
                        f"chunks="
                        f"{chunk_count}"
                    )
 
                if depth < MAX_DEPTH:
 
                    for link in (
                        discovered_links
                    ):
 
                        if (
                            link
                            not in visited
                        ):
 
                            queue.append(
                                (
                                    link,
                                    depth + 1,
                                )
                            )
 
                processed += 1
 
                if CRAWL_DELAY > 0:
                    time.sleep(
                        CRAWL_DELAY
                    )
 
            except Exception as exc:
 
                failed += 1
                processed += 1
 
                print(
                    f"FAILED: {url}"
                )
 
                print(
                    f"Reason: {exc}"
                )
 
    save_state(
        state
    )
 
    print("")
    print(
        "=== Crawl completed ==="
    )
    print(
        f"Processed: {processed}"
    )
    print(
        f"Changed:   {changed}"
    )
    print(
        f"Unchanged: {skipped}"
    )
    print(
        f"Failed:    {failed}"
    )
 
 
if __name__ == "__main__":
    crawl()