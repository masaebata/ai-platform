from fastmcp import FastMCP
 
from panos_client import PanOSClient
 
 
mcp = FastMCP(
    name="Palo Alto Networks MCP Server"
)
 
 
@mcp.tool
async def get_device_info() -> dict:
    """
    Get basic system information from the configured
    Palo Alto Networks firewall.
 
    Read-only operation.
    """
 
    client = PanOSClient()
 
    return await client.get_device_info()
 
 
@mcp.tool
async def get_interfaces() -> dict:
    """
    Get interface information from the configured
    Palo Alto Networks firewall.
 
    Equivalent to:
    show interface all
 
    Read-only operation.
    """
 
    client = PanOSClient()
 
    return await client.get_interfaces()
 
 
@mcp.tool
async def get_routes() -> dict:
    """
    Get the runtime routing table from the configured
    Palo Alto Networks firewall.
 
    Equivalent to:
    show routing route
 
    Read-only operation.
    """
 
    client = PanOSClient()
 
    return await client.get_routes()
 
 
@mcp.tool
async def get_security_rules() -> dict:
    """
    Get configured Security Policy rules.
 
    Returns a normalized list including:
    name, zones, source, destination,
    application, service, action and logging settings.
 
    Read-only operation.
    """
 
    client = PanOSClient()
 
    return await client.get_security_rules()
 
 
@mcp.tool
async def get_threat_logs(
    nlogs: int = 20,
    query: str | None = None,
) -> dict:
    """
    Get recent PAN-OS Threat Logs.
 
    Args:
        nlogs:
            Number of logs to retrieve.
            Default is 20.
 
        query:
            Optional PAN-OS log query.
            Example:
            (severity eq high)
 
    Read-only operation.
    """
 
    client = PanOSClient()
 
    return await client.get_threat_logs(
        nlogs=nlogs,
        query=query,
    )
 
 
if __name__ == "__main__":
 
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )