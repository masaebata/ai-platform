import os

from functools import lru_cache
 
from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer
 
 
QDRANT_URL = os.getenv(

    "QDRANT_URL",

    "http://qdrant:6333",

)
 
QDRANT_COLLECTION = os.getenv(

    "QDRANT_COLLECTION",

    "paloalto-docs",

)
 
EMBEDDING_MODEL = os.getenv(

    "EMBEDDING_MODEL",

    "intfloat/multilingual-e5-small",

)
 
 
@lru_cache(maxsize=1)

def get_embedding_model() -> SentenceTransformer:

    return SentenceTransformer(

        EMBEDDING_MODEL

    )
 
 
@lru_cache(maxsize=1)

def get_qdrant_client() -> QdrantClient:

    return QdrantClient(

        url=QDRANT_URL

    )
 
 
def embed_passages(

    texts: list[str],

) -> list[list[float]]:
 
    model = get_embedding_model()
 
    passages = [

        f"passage: {text}"

        for text in texts

    ]
 
    vectors = model.encode(

        passages,

        normalize_embeddings=True,

        show_progress_bar=False,

    )
 
    return vectors.tolist()
 
 
def embed_query(

    query: str,

) -> list[float]:
 
    model = get_embedding_model()
 
    vector = model.encode(

        [f"query: {query}"],

        normalize_embeddings=True,

        show_progress_bar=False,

    )[0]
 
    return vector.tolist()
 