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
    Get configured Security Policy rules from the
    configured Palo Alto Networks firewall.
 
    Uses the PAN-OS XML Configuration API.
 
    Read-only operation.
    """
 
    client = PanOSClient()
 
    return await client.get_security_rules()
 
 
if __name__ == "__main__":
 
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )