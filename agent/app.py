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
from pydantic import BaseModel, Field
 
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
 
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
 
logger = logging.getLogger("ai-agent")
 
 
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
 
AGENT_TIMEOUT = int(
    os.getenv(
        "AGENT_TIMEOUT",
        "180",
    )
)
 
 
SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援するAI Agentです。
 
必ず日本語で回答してください。
 
Palo Alto Networksファイアウォールの現在状態を確認する必要がある場合は、
推測せず、利用可能なMCP Toolを使用してください。
 
現在のMCP ToolはRead Onlyです。
 
ルール:
- 必ず日本語で回答すること
- 実機情報について推測しないこと
- Palo Alto Networksの実機情報を聞かれた場合はMCP Toolを優先すること
- Toolから取得した情報と一般知識を明確に区別すること
- APIキー、パスワード、認証情報などのSecretを回答に含めないこと
- 現時点では設定変更を行わないこと
- Toolの実行結果を取得した場合は、日本語で簡潔に整理して回答すること
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
 
    expected = f"Bearer {AGENT_API_KEY}"
 
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
 
 
def get_last_user_message(
    messages: list[OpenAIMessage],
) -> str:
    for message in reversed(messages):
        if (
            message.role == "user"
            and message.content
        ):
            return message.content
 
    raise HTTPException(
        status_code=400,
        detail="No user message found",
    )
 
 
def tool_result_fallback(
    result_messages: list[Any],
) -> str:
    """
    LLMがTool実行後の最終回答を空で返した場合、
    ToolMessageの内容をユーザー向けに返す。
    """
 
    for message in reversed(result_messages):
 
        if message.__class__.__name__ != "ToolMessage":
            continue
 
        content = getattr(
            message,
            "content",
            None,
        )
 
        if not content:
            continue
 
        if not isinstance(content, str):
            return (
                "PA-VMから取得した情報です。\n\n"
                f"{content}"
            )
 
        try:
            data = json.loads(content)
 
            if isinstance(data, dict):
                labels = {
                    "hostname": "ホスト名",
                    "ip_address": "管理IP",
                    "netmask": "ネットマスク",
                    "default_gateway": "デフォルトゲートウェイ",
                    "model": "モデル",
                    "serial": "シリアル番号",
                    "sw_version": "PAN-OS",
                    "app_version": "Applications and Threats",
                    "av_version": "Antivirus",
                    "wildfire_version": "WildFire",
                    "uptime": "稼働時間",
                    "family": "ファミリー",
                }
 
                lines = [
                    "PA-VMから取得した機器情報です。",
                    "",
                ]
 
                for key, value in data.items():
                    if value is None:
                        continue
 
                    label = labels.get(
                        key,
                        key,
                    )
 
                    lines.append(
                        f"- {label}: {value}"
                    )
 
                return "\n".join(lines)
 
        except json.JSONDecodeError:
            pass
 
        return (
            "PA-VMから取得した情報です。\n\n"
            f"{content}"
        )
 
    return (
        "MCP Toolの実行は行われましたが、"
        "回答本文を生成できませんでした。"
    )
 
 
def extract_final_answer(
    result_messages: list[Any],
) -> str:
 
    for message in reversed(result_messages):
 
        if message.__class__.__name__ != "AIMessage":
            continue
 
        content = getattr(
            message,
            "content",
            None,
        )
 
        if isinstance(content, str):
            content = content.strip()
 
            if content:
                return content
 
    return tool_result_fallback(
        result_messages
    )
 
 
async def run_agent(
    app: FastAPI,
    message: str,
) -> tuple[str, list[dict[str, Any]]]:
 
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
 
    for msg in result["messages"]:
 
        debug_messages.append(
            {
                "type": msg.__class__.__name__,
                "content": getattr(
                    msg,
                    "content",
                    None,
                ),
                "tool_calls": getattr(
                    msg,
                    "tool_calls",
                    None,
                ),
                "name": getattr(
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
 
    return answer, debug_messages
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
 
    logger.info(
        "Connecting to Palo Alto MCP: %s",
        PALOALTO_MCP_URL,
    )
 
    mcp_client = MultiServerMCPClient(
        {
            "paloalto": {
                "transport": "http",
                "url": PALOALTO_MCP_URL,
            }
        }
    )
 
    tools = await mcp_client.get_tools()
 
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
 
    app.state.mcp_client = mcp_client
    app.state.tools = tools
    app.state.agent = agent
 
    yield
 
 
app = FastAPI(
    title="AI Network Agent",
    version="0.4.0",
    lifespan=lifespan,
)
 
 
@app.get("/health")
async def health(
    request: Request,
):
    return {
        "status": "ok",
        "agent_model": AGENT_MODEL_ID,
        "llm_model": OLLAMA_MODEL,
        "llm_endpoint": OLLAMA_BASE_URL,
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
                "id": AGENT_MODEL_ID,
                "object": "model",
                "created": int(
                    time.time()
                ),
                "owned_by": "local-ai-platform",
            }
        ],
    }
 
 
@app.post("/v1/chat/completions")
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
 
    if body.model != AGENT_MODEL_ID:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown model: "
                f"{body.model}"
            ),
        )
 
    user_message = get_last_user_message(
        body.messages
    )
 
    completion_id = (
        f"chatcmpl-{uuid.uuid4().hex}"
    )
 
    created = int(
        time.time()
    )
 
    if body.stream:
 
        async def event_stream():
 
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": AGENT_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None,
                    }
                ],
            }
 
            yield (
                f"data: "
                f"{json.dumps(first_chunk, ensure_ascii=False)}"
                f"\n\n"
            )
 
            try:
                answer, _ = await run_agent(
                    request.app,
                    user_message,
                )
 
                content_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": AGENT_MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": answer
                            },
                            "finish_reason": None,
                        }
                    ],
                }
 
                yield (
                    f"data: "
                    f"{json.dumps(content_chunk, ensure_ascii=False)}"
                    f"\n\n"
                )
 
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": AGENT_MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
 
                yield (
                    f"data: "
                    f"{json.dumps(final_chunk, ensure_ascii=False)}"
                    f"\n\n"
                )
 
                yield "data: [DONE]\n\n"
 
            except Exception as exc:
                logger.exception(
                    "Streaming request failed"
                )
 
                error_chunk = {
                    "error": {
                        "message": str(exc),
                        "type": "agent_error",
                    }
                }
 
                yield (
                    f"data: "
                    f"{json.dumps(error_chunk, ensure_ascii=False)}"
                    f"\n\n"
                )
 
                yield "data: [DONE]\n\n"
 
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )
 
    answer, _ = await run_agent(
        request.app,
        user_message,
    )
 
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": AGENT_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
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
 
    answer, debug_messages = await run_agent(
        request.app,
        body.message,
    )
 
    return {
        "answer": answer,
        "agent_model": AGENT_MODEL_ID,
        "available_tools": [
            tool.name
            for tool in request.app.state.tools
        ],
        "messages": debug_messages,
    }