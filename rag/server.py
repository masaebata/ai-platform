from fastmcp import FastMCP
 
from common import (
    QDRANT_COLLECTION,
    embed_query,
    get_qdrant_client,
)
 
 
mcp = FastMCP(
    name="Palo Alto Documentation RAG MCP"
)
 
 
@mcp.tool
async def search_paloalto_docs(
    query: str,
    limit: int = 5,
) -> dict:
    """
    Search Palo Alto Networks documentation
    stored in the local Qdrant RAG database.
 
    Use this tool when the user asks about:
    - PAN-OS configuration guidance
    - Palo Alto Networks documentation
    - recommended configuration
    - configuration meaning
    - implementation guidance
    - technical references
 
    Args:
        query:
            Natural language search query.
 
        limit:
            Maximum number of chunks to return.
            Default is 5.
 
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
                "RAG collection has not "
                "been created yet."
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
                "score": point.score,
                "source": payload.get(
                    "source"
                ),
                "page": payload.get(
                    "page"
                ),
                "chunk_index": payload.get(
                    "chunk_index"
                ),
                "text": payload.get(
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