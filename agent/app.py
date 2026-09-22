import os
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
 
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
 
 
LITELLM_BASE_URL = os.getenv(
    "LITELLM_BASE_URL",
    "http://litellm:4000/v1",
)
 
LITELLM_API_KEY = os.environ["LITELLM_MASTER_KEY"]
 
AGENT_MODEL = os.getenv(
    "AGENT_MODEL",
    "local-general",
)
 
PALOALTO_MCP_URL = os.getenv(
    "PALOALTO_MCP_URL",
    "http://paloalto-mcp:8000/mcp",
)
 
 
SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援するAI Agentです。
 
Palo Alto Networksファイアウォールの現在状態を確認する必要がある場合は、
推測せず、利用可能なMCP Toolを使用してください。
 
現在のMCP ToolはRead Onlyです。
 
ルール:
- 実機情報について推測しないこと
- Palo Alto Networksの実機情報を聞かれた場合はMCP Toolを優先すること
- Toolから取得した情報と一般知識を区別すること
- APIキーや認証情報などのSecretを回答に含めないこと
- 現時点では設定変更を行わないこと
"""
 
 
class ChatRequest(BaseModel):
    message: str
 
 
class ChatResponse(BaseModel):
    answer: str
    model: str
    available_tools: list[str]
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
 
    mcp_client = MultiServerMCPClient(
        {
            "paloalto": {
                "transport": "http",
                "url": PALOALTO_MCP_URL,
            }
        }
    )
 
    tools = await mcp_client.get_tools()
 
    model = ChatOpenAI(
        model=AGENT_MODEL,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        temperature=0,
    )
 
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
 
    app.state.mcp_client = mcp_client
    app.state.tools = tools
    app.state.agent = agent
 
    yield
 
 
app = FastAPI(
    title="AI Network Agent",
    version="0.1.0",
    lifespan=lifespan,
)
 
 
@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "model": AGENT_MODEL,
        "tools": [
            tool.name
            for tool in request.app.state.tools
        ],
    }
 
 
@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
):
 
    try:
        result = await request.app.state.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": body.message,
                    }
                ]
            }
        )
 
        final_message = result["messages"][-1]
 
        content = final_message.content
 
        if not isinstance(content, str):
            content = str(content)
 
        return ChatResponse(
            answer=content,
            model=AGENT_MODEL,
            available_tools=[
                tool.name
                for tool in request.app.state.tools
            ],
        )
 
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc