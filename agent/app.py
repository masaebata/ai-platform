import asyncio
import json
import logging
import os
import time
import uuid
 
from contextlib import asynccontextmanager
from typing import Any
 
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
)
 
from fastapi.responses import (
    StreamingResponse,
)
 
from pydantic import BaseModel
 
from langchain.agents import (
    create_agent,
)
 
from langchain_openai import (
    ChatOpenAI,
)
 
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)
 
 
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
 
BASE_AGENT_MODEL_ID = (
    "paloalto-agent"
)
 
RAG_AGENT_MODEL_ID = (
    "paloalto-rag-agent"
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
 
 
BASE_SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援する
Palo Alto Networks AI Agentです。
 
必ず日本語で回答してください。
 
このモデルは実機確認専用です。
 
Palo Alto MCPを使ってPA-VMの現在状態を確認できます。
 
ルール:
 
- 実機情報について推測しないこと。
- 実機状態に関する質問ではPalo Alto MCPを使用すること。
- RAG検索は使用しないこと。
- APIキーや認証情報を回答に含めないこと。
- 現在のMCP ToolはRead Onlyです。
- 設定変更を行わないこと。
"""
 
 
RAG_SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援する
Palo Alto Networks AI Agentです。
 
必ず日本語で回答してください。
 
このモデルでは次の2種類の情報源を利用できます。
 
1. Palo Alto MCP
   PA-VMの現在の設定・状態・ログを取得します。
 
2. Documentation RAG MCP
   Qdrantに保存されたPalo Alto Networks TechDocsを検索します。
 
ルール:
 
- 実機の現在状態はPalo Alto MCPで確認すること。
 
- 製品仕様、設定方法、推奨事項、
  技術的な根拠が必要な場合は
  search_paloalto_docsを使用すること。
 
- 実機設定とTechDocsの比較を求められた場合は
  両方のToolを使用すること。
 
- TechDocs検索結果を使った場合は、
  可能な限り資料タイトル、URL、
  PDFの場合はページ番号を回答に示すこと。
 
- 一般知識だけでTechDocsの内容を推測しないこと。
 
- APIキーや認証情報を回答に含めないこと。
 
- Palo Alto MCPは現在Read Onlyです。
  設定変更を行わないこと。
"""
 
 
class ChatRequest(BaseModel):
    message: str
    model: str = BASE_AGENT_MODEL_ID
 
 
class OpenAIMessage(BaseModel):
    role: str
    content: str | None = None
 
 
class OpenAIChatRequest(BaseModel):
    model: str = BASE_AGENT_MODEL_ID
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
 
    if authorization != (
        f"Bearer {AGENT_API_KEY}"
    ):
 
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
 
 
def get_agent_for_model(
    app: FastAPI,
    model_id: str,
):
 
    if (
        model_id
        == BASE_AGENT_MODEL_ID
    ):
 
        return (
            app.state.base_agent,
            app.state.base_tools,
        )
 
    if (
        model_id
        == RAG_AGENT_MODEL_ID
    ):
 
        return (
            app.state.rag_agent,
            app.state.rag_tools,
        )
 
    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown model: "
            f"{model_id}"
        ),
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
    model_id: str,
):
 
    agent, tools = (
        get_agent_for_model(
            app,
            model_id,
        )
    )
 
    logger.info(
        "Agent request started "
        "model=%s message=%s",
        model_id,
        message,
    )
 
    try:
 
        result = await asyncio.wait_for(
            agent.ainvoke(
                {
                    "messages": [
                        {
                            "role":
                                "user",
 
                            "content":
                                message,
                        }
                    ]
                }
            ),
            timeout=AGENT_TIMEOUT,
        )
 
    except asyncio.TimeoutError as exc:
 
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
 
    answer = extract_final_answer(
        result["messages"]
    )
 
    return (
        answer,
        debug_messages,
        tools,
    )
 
 
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
 
    mcp_client = (
        MultiServerMCPClient(
            {
                "paloalto": {
                    "transport":
                        "http",
 
                    "url":
                        PALOALTO_MCP_URL,
                },
 
                "rag": {
                    "transport":
                        "http",
 
                    "url":
                        RAG_MCP_URL,
                },
            }
        )
    )
 
    all_tools = await (
        mcp_client.get_tools()
    )
 
    base_tools = [
        tool
        for tool in all_tools
        if tool.name
        != "search_paloalto_docs"
    ]
 
    rag_tools = (
        all_tools
    )
 
    model = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=0,
    )
 
    base_agent = create_agent(
        model=model,
        tools=base_tools,
        system_prompt=(
            BASE_SYSTEM_PROMPT
        ),
    )
 
    rag_agent = create_agent(
        model=model,
        tools=rag_tools,
        system_prompt=(
            RAG_SYSTEM_PROMPT
        ),
    )
 
    app.state.mcp_client = (
        mcp_client
    )
 
    app.state.base_tools = (
        base_tools
    )
 
    app.state.rag_tools = (
        rag_tools
    )
 
    app.state.base_agent = (
        base_agent
    )
 
    app.state.rag_agent = (
        rag_agent
    )
 
    logger.info(
        "Base tools: %s",
        [
            tool.name
            for tool in base_tools
        ],
    )
 
    logger.info(
        "RAG tools: %s",
        [
            tool.name
            for tool in rag_tools
        ],
    )
 
    yield
 
 
app = FastAPI(
    title="AI Network Agent",
    version="0.6.0",
    lifespan=lifespan,
)
 
 
@app.get("/health")
async def health(
    request: Request,
):
 
    return {
        "status": "ok",
 
        "models": {
            BASE_AGENT_MODEL_ID: [
                tool.name
                for tool
                in request.app.state.base_tools
            ],
 
            RAG_AGENT_MODEL_ID: [
                tool.name
                for tool
                in request.app.state.rag_tools
            ],
        },
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
 
    created = int(
        time.time()
    )
 
    return {
        "object": "list",
 
        "data": [
            {
                "id":
                    BASE_AGENT_MODEL_ID,
 
                "object":
                    "model",
 
                "created":
                    created,
 
                "owned_by":
                    "local-ai-platform",
            },
 
            {
                "id":
                    RAG_AGENT_MODEL_ID,
 
                "object":
                    "model",
 
                "created":
                    created,
 
                "owned_by":
                    "local-ai-platform",
            },
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
                "id":
                    completion_id,
 
                "object":
                    "chat.completion.chunk",
 
                "created":
                    created,
 
                "model":
                    body.model,
 
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
 
                answer, _, _ = (
                    await run_agent(
                        request.app,
                        user_message,
                        body.model,
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
                        body.model,
 
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
                        body.model,
 
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
 
    answer, _, _ = await run_agent(
        request.app,
        user_message,
        body.model,
    )
 
    return {
        "id":
            completion_id,
 
        "object":
            "chat.completion",
 
        "created":
            created,
 
        "model":
            body.model,
 
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
 
    (
        answer,
        debug_messages,
        tools,
    ) = await run_agent(
        request.app,
        body.message,
        body.model,
    )
 
    return {
        "answer":
            answer,
 
        "model":
            body.model,
 
        "available_tools": [
            tool.name
            for tool in tools
        ],
 
        "messages":
            debug_messages,
    }