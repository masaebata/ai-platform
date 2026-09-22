from fastmcp import FastMCP
 
from common import (
    QDRANT_COLLECTION,
    embed_query,
    get_qdrant_client,
)
 
 
mcp = FastMCP(
    name="Documentation RAG MCP"
)
 
 
@mcp.tool
async def search_paloalto_docs(
    query: str,
    limit: int = 5,
) -> dict:
    """
    Search documentation stored in the
    local Qdrant RAG database.
 
    Use this tool for:
    - Palo Alto Networks documentation
    - PAN-OS configuration guidance
    - recommended configuration
    - technical specifications
    - implementation guidance
    - configuration meaning
    - troubleshooting documentation
 
    Args:
 
        query:
            Natural language search query.
 
        limit:
            Maximum number of results.
            Default is 5.
            Maximum is 10.
 
    Read-only operation.
    """
 
    if limit < 1:
        limit = 1
 
    if limit > 10:
        limit = 10
 
    client = get_qdrant_client()
 
    if not client.collection_exists(
        QDRANT_COLLECTION
    ):
 
        return {
            "count": 0,
            "results": [],
            "message": (
                "RAG collection does "
                "not exist."
            ),
        }
 
    vector = embed_query(
        query
    )
 
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    )
 
    results = []
 
    for point in response.points:
 
        payload = (
            point.payload
            or {}
        )
 
        results.append(
            {
                "score":
                    point.score,
 
                "title":
                    payload.get(
                        "title"
                    ),
 
                "url":
                    payload.get(
                        "source_url"
                    ),
 
                "source_type":
                    payload.get(
                        "source_type"
                    ),
 
                "page":
                    payload.get(
                        "page"
                    ),
 
                "chunk_index":
                    payload.get(
                        "chunk_index"
                    ),
 
                "text":
                    payload.get(
                        "text"
                    ),
            }
        )
 
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }
 
 
if __name__ == "__main__":
 
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )