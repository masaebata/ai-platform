import asyncio
 
from fastmcp import Client
 
 
MCP_URL = "http://paloalto-mcp:8000/mcp"
 
 
async def main():

    client = Client(MCP_URL)
 
    async with client:

        print("=== Available MCP Tools ===")
 
        tools = await client.list_tools()
 
        for tool in tools:

            print(f"- {tool.name}")
 
        print()

        print("=== get_device_info ===")
 
        result = await client.call_tool(

            "get_device_info",

            {}

        )
 
        print(result)
 
 
if __name__ == "__main__":

    asyncio.run(main())
 