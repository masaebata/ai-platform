import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
 
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
 
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
 
 
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)
 
logger = logging.getLogger(
    "ai-agent"
)
 
 
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://192.168.11.129:11434/v1",
)
 
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b",
)
 
AGENT_MODEL_ID = os.getenv(
    "AGENT_MODEL_ID",
    "paloalto-agent",
)
 
AGENT_API_KEY = os.getenv(
    "AGENT_API_KEY",
    "local-agent",
)
 
PALOALTO_MCP_URL = os.getenv(
    "PALOALTO_MCP_URL",
    "http://paloalto-mcp:8000/mcp",
)
 
RAG_MCP_URL = os.getenv(
    "RAG_MCP_URL",
    "http://rag-mcp:8000/mcp",
)
 
AGENT_TIMEOUT = int(
    os.getenv(
        "AGENT_TIMEOUT",
        "600",
    )
)
 
 
SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援するAI Agentです。
 
必ず日本語で回答してください。
 
利用可能な情報源は主に次の2種類です。
 
1. Palo Alto MCP
   実際のPA-VMから現在の設定や状態を取得します。
 
2. RAG MCP
   Qdrantに登録されたPalo Alto Networks関連文書を検索します。
 
ルール:
 
- 実機の現在状態について質問された場合は、
  推測せずPalo Alto MCPを使用してください。
 
- 製品仕様、設定方法、推奨事項、技術説明、
  ドキュメント根拠が必要な場合は、
  RAG MCPのsearch_paloalto_docsを使用してください。
 
- 実機設定とドキュメントの比較を求められた場合は、
  Palo Alto MCPとRAG MCPの両方を使用してください。
 
- Toolから取得した情報と一般知識を区別してください。
 
- RAG結果を使用した場合は、
  回答中に資料名とページ番号が取得できる場合は明示してください。
 
- APIキー、パスワード、認証情報などのSecretを
  回答に含めないでください。
 
- 現在のPalo Alto MCPはRead Onlyです。
  設定変更を行わないでください。
 
- 必ず日本語で簡潔かつ技術的に正確に回答してください。
"""
 
 
class ChatRequest(BaseModel):
    message: str
 
 
class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None
 
 
class OpenAIChatRequest(BaseModel):
    model: str = AGENT_MODEL_ID
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
 
 
def check_api_key(
    authorization: str | None,
) -> None:
 
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is required",
        )
 
    expected = (
        f"Bearer {AGENT_API_KEY}"
    )
 
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
 
 
def get_last_user_message(
    messages: list[OpenAIMessage],
) -> str:
 
    for message in reversed(
        messages
    ):
 
        if (
            message.role == "user"
            and message.content
        ):
            return message.content
 
    raise HTTPException(
        status_code=400,
        detail="No user message found",
    )
 
 
def extract_final_answer(
    result_messages: list[Any],
) -> str:
 
    for message in reversed(
        result_messages
    ):
 
        if (
            message.__class__.__name__
            != "AIMessage"
        ):
            continue
 
        content = getattr(
            message,
            "content",
            None,
        )
 
        if isinstance(
            content,
            str,
        ):
            content = content.strip()
 
            if content:
                return content
 
    for message in reversed(
        result_messages
    ):
 
        if (
            message.__class__.__name__
            != "ToolMessage"
        ):
            continue
 
        content = getattr(
            message,
            "content",
            None,
        )
 
        if content:
            return (
                "Toolから取得した情報です。\n\n"
                f"{content}"
            )
 
    return (
        "Tool処理は実行されましたが、"
        "回答本文を生成できませんでした。"
    )
 
 
async def run_agent(
    app: FastAPI,
    message: str,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
 
    logger.info(
        "Agent request started: %s",
        message,
    )
 
    try:
 
        result = await asyncio.wait_for(
            app.state.agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": message,
                        }
                    ]
                }
            ),
            timeout=AGENT_TIMEOUT,
        )
 
    except asyncio.TimeoutError as exc:
 
        logger.error(
            "Agent timed out after %s seconds",
            AGENT_TIMEOUT,
        )
 
        raise HTTPException(
            status_code=504,
            detail=(
                f"Agent timeout after "
                f"{AGENT_TIMEOUT} seconds"
            ),
        ) from exc
 
    debug_messages = []
 
    for msg in result[
        "messages"
    ]:
 
        debug_messages.append(
            {
                "type":
                    msg.__class__.__name__,
 
                "content":
                    getattr(
                        msg,
                        "content",
                        None,
                    ),
 
                "tool_calls":
                    getattr(
                        msg,
                        "tool_calls",
                        None,
                    ),
 
                "name":
                    getattr(
                        msg,
                        "name",
                        None,
                    ),
            }
        )
 
    logger.info(
        "Agent messages: %s",
        debug_messages,
    )
 
    answer = extract_final_answer(
        result["messages"]
    )
 
    logger.info(
        "Agent request completed"
    )
 
    return (
        answer,
        debug_messages,
    )
 
 
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
 
    logger.info(
        "Connecting Palo Alto MCP: %s",
        PALOALTO_MCP_URL,
    )
 
    logger.info(
        "Connecting RAG MCP: %s",
        RAG_MCP_URL,
    )
 
    mcp_client = MultiServerMCPClient(
        {
            "paloalto": {
                "transport": "http",
                "url": PALOALTO_MCP_URL,
            },
 
            "rag": {
                "transport": "http",
                "url": RAG_MCP_URL,
            },
        }
    )
 
    tools = await (
        mcp_client.get_tools()
    )
 
    logger.info(
        "MCP tools loaded: %s",
        [
            tool.name
            for tool in tools
        ],
    )
 
    model = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0,
    )
 
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
 
    app.state.mcp_client = (
        mcp_client
    )
 
    app.state.tools = tools
    app.state.agent = agent
 
    yield
 
 
app = FastAPI(
    title="AI Network Agent",
    version="0.5.0",
    lifespan=lifespan,
)
 
 
@app.get("/health")
async def health(
    request: Request,
):
 
    return {
        "status": "ok",
        "agent_model":
            AGENT_MODEL_ID,
 
        "llm_model":
            OLLAMA_MODEL,
 
        "llm_endpoint":
            OLLAMA_BASE_URL,
 
        "tools": [
            tool.name
            for tool in request.app.state.tools
        ],
    }
 
 
@app.get("/v1/models")
async def models(
    authorization: str | None = Header(
        default=None
    ),
):
 
    check_api_key(
        authorization
    )
 
    return {
        "object": "list",
        "data": [
            {
                "id":
                    AGENT_MODEL_ID,
 
                "object":
                    "model",
 
                "created":
                    int(
                        time.time()
                    ),
 
                "owned_by":
                    "local-ai-platform",
            }
        ],
    }
 
 
@app.post(
    "/v1/chat/completions"
)
async def openai_chat_completions(
    body: OpenAIChatRequest,
    request: Request,
    authorization: str | None = Header(
        default=None
    ),
):
 
    check_api_key(
        authorization
    )
 
    if (
        body.model
        != AGENT_MODEL_ID
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown model: "
                f"{body.model}"
            ),
        )
 
    user_message = (
        get_last_user_message(
            body.messages
        )
    )
 
    completion_id = (
        f"chatcmpl-"
        f"{uuid.uuid4().hex}"
    )
 
    created = int(
        time.time()
    )
 
    if body.stream:
 
        async def event_stream():
 
            first_chunk = {
                "id": completion_id,
                "object":
                    "chat.completion.chunk",
                "created": created,
                "model":
                    AGENT_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role":
                                "assistant"
                        },
                        "finish_reason":
                            None,
                    }
                ],
            }
 
            yield (
                "data: "
                + json.dumps(
                    first_chunk,
                    ensure_ascii=False,
                )
                + "\n\n"
            )
 
            try:
 
                answer, _ = (
                    await run_agent(
                        request.app,
                        user_message,
                    )
                )
 
                content_chunk = {
                    "id":
                        completion_id,
 
                    "object":
                        "chat.completion.chunk",
 
                    "created":
                        created,
 
                    "model":
                        AGENT_MODEL_ID,
 
                    "choices": [
                        {
                            "index": 0,
 
                            "delta": {
                                "content":
                                    answer
                            },
 
                            "finish_reason":
                                None,
                        }
                    ],
                }
 
                yield (
                    "data: "
                    + json.dumps(
                        content_chunk,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
 
                final_chunk = {
                    "id":
                        completion_id,
 
                    "object":
                        "chat.completion.chunk",
 
                    "created":
                        created,
 
                    "model":
                        AGENT_MODEL_ID,
 
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason":
                                "stop",
                        }
                    ],
                }
 
                yield (
                    "data: "
                    + json.dumps(
                        final_chunk,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
 
                yield (
                    "data: [DONE]\n\n"
                )
 
            except Exception as exc:
 
                logger.exception(
                    "Streaming request failed"
                )
 
                error_chunk = {
                    "error": {
                        "message":
                            str(exc),
 
                        "type":
                            "agent_error",
                    }
                }
 
                yield (
                    "data: "
                    + json.dumps(
                        error_chunk,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
 
                yield (
                    "data: [DONE]\n\n"
                )
 
        return StreamingResponse(
            event_stream(),
            media_type=(
                "text/event-stream"
            ),
        )
 
    answer, _ = await run_agent(
        request.app,
        user_message,
    )
 
    return {
        "id": completion_id,
        "object":
            "chat.completion",
        "created": created,
        "model":
            AGENT_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role":
                        "assistant",
 
                    "content":
                        answer,
                },
                "finish_reason":
                    "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
 
 
@app.post("/chat")
async def debug_chat(
    body: ChatRequest,
    request: Request,
):
 
    answer, debug_messages = (
        await run_agent(
            request.app,
            body.message,
        )
    )
 
    return {
        "answer": answer,
 
        "agent_model":
            AGENT_MODEL_ID,
 
        "available_tools": [
            tool.name
            for tool in request.app.state.tools
        ],
 
        "messages":
            debug_messages,
    }