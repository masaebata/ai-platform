import asyncio
import json
import logging
import os
import time
import uuid

from contextlib import asynccontextmanager
from typing import Any

import asyncpg

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

from langchain_openai import (
    ChatOpenAI,
)

from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
)

from agent_graph import (
    build_agent_graph,
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


# =========================================================
# LLM
# =========================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://192.168.11.129:11434/v1",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3-4b-wiz",
)


# =========================================================
# Existing Agent IDs
# =========================================================

BASE_AGENT_MODEL_ID = (
    "paloalto-agent"
)

RAG_AGENT_MODEL_ID = (
    "paloalto-rag-agent"
)


# =========================================================
# API / MCP
# =========================================================

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


# =========================================================
# PostgreSQL
# =========================================================

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "postgres",
)

POSTGRES_PORT = int(
    os.getenv(
        "POSTGRES_PORT",
        "5432",
    )
)

POSTGRES_USER = os.environ[
    "POSTGRES_USER"
]

POSTGRES_PASSWORD = os.environ[
    "POSTGRES_PASSWORD"
]

POSTGRES_DB = os.environ[
    "POSTGRES_DB"
]


# =========================================================
# System Prompts
# =========================================================

BASE_SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援する
Palo Alto Networks AI Agentです。

あなたのAgent IDは paloalto-agent です。

必ず日本語で回答してください。

このAgentはPalo Alto Networks実機確認専用です。

Palo Alto MCPを利用して、
PA-VMの現在の設定・状態・ログを確認できます。

ルール:

- 実機情報について推測しないこと。
- 実機状態に関する質問ではPalo Alto MCPを使用すること。
- RAG検索は使用しないこと。
- Toolから取得した結果を根拠に回答すること。
- APIキーや認証情報を回答に含めないこと。
- 現在利用可能なMCP ToolはRead Onlyです。
- 設定変更を行わないこと。
"""


RAG_SYSTEM_PROMPT = """
あなたは企業ネットワーク運用を支援する
Palo Alto Networks AI Agentです。

あなたのAgent IDは paloalto-rag-agent です。

必ず日本語で回答してください。

このAgentでは次の2種類の情報源を利用できます。

1. Palo Alto MCP

PA-VMの現在の設定・状態・ログを取得します。

2. Documentation RAG MCP

Qdrantに保存された
Palo Alto Networks TechDocsを検索します。

ルール:

- 実機の現在状態はPalo Alto MCPで確認すること。

- 製品仕様、設定方法、推奨事項、
  技術的な根拠が必要な場合は
  search_paloalto_docsを使用すること。

- 実機設定とTechDocsの比較を求められた場合は
  Palo Alto MCPとRAG MCPの両方を使用すること。

- TechDocs検索結果を使用した場合は、
  可能な限り資料タイトル、URL、
  PDFの場合はページ番号を回答に示すこと。

- Toolから取得した結果と一般知識を区別すること。

- 一般知識だけでTechDocsの内容を推測しないこと。

- APIキーや認証情報を回答に含めないこと。

- 現在利用可能なPalo Alto MCPはRead Onlyです。
  設定変更を行わないこと。
"""


# =========================================================
# Request Models
# =========================================================

class ChatRequest(
    BaseModel
):
    message: str
    model: str = (
        BASE_AGENT_MODEL_ID
    )


class OpenAIMessage(
    BaseModel
):
    role: str
    content: str | None = None


class OpenAIChatRequest(
    BaseModel
):
    model: str = (
        BASE_AGENT_MODEL_ID
    )

    messages: list[
        OpenAIMessage
    ]

    stream: bool = False

    temperature: (
        float
        | None
    ) = None

    max_tokens: (
        int
        | None
    ) = None


# =========================================================
# Authentication
# =========================================================

def check_api_key(
    authorization: str | None,
) -> None:

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail=(
                "Authorization header "
                "is required"
            ),
        )

    if authorization != (
        f"Bearer {AGENT_API_KEY}"
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# =========================================================
# Request Helpers
# =========================================================

def get_last_user_message(
    messages: list[
        OpenAIMessage
    ],
) -> str:

    for message in reversed(
        messages
    ):

        if (
            message.role
            == "user"
            and message.content
        ):

            return (
                message.content
            )

    raise HTTPException(
        status_code=400,
        detail=(
            "No user message found"
        ),
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


# =========================================================
# Message Helpers
# =========================================================

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

            content = (
                content.strip()
            )

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


def serialize_content(
    value: Any,
) -> Any:

    try:

        json.dumps(
            value,
            ensure_ascii=False,
        )

        return value

    except Exception:

        return str(
            value
        )


# =========================================================
# Audit Helpers
# =========================================================

def extract_tool_audit(
    result_messages: list[Any],
) -> tuple[
    list[str],
    list[dict[str, Any]],
]:

    tools_used = []
    tool_results = []

    for message in (
        result_messages
    ):

        message_type = (
            message
            .__class__
            .__name__
        )

        if (
            message_type
            == "AIMessage"
        ):

            tool_calls = getattr(
                message,
                "tool_calls",
                None,
            )

            if not tool_calls:
                continue

            for tool_call in (
                tool_calls
            ):

                tool_name = None

                if isinstance(
                    tool_call,
                    dict,
                ):

                    tool_name = (
                        tool_call.get(
                            "name"
                        )
                    )

                if (
                    tool_name
                    and tool_name
                    not in tools_used
                ):

                    tools_used.append(
                        tool_name
                    )

        elif (
            message_type
            == "ToolMessage"
        ):

            tool_results.append(
                {
                    "name":
                        getattr(
                            message,
                            "name",
                            None,
                        ),

                    "content":
                        serialize_content(
                            getattr(
                                message,
                                "content",
                                None,
                            )
                        ),
                }
            )

    return (
        tools_used,
        tool_results,
    )


# =========================================================
# PostgreSQL
# =========================================================

async def init_database(
    pool: asyncpg.Pool,
) -> None:

    async with (
        pool.acquire()
        as connection
    ):

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_audit_logs (
                id BIGSERIAL PRIMARY KEY,

                request_id UUID NOT NULL UNIQUE,

                created_at TIMESTAMPTZ
                    NOT NULL
                    DEFAULT NOW(),

                model VARCHAR(100)
                    NOT NULL,

                user_message TEXT
                    NOT NULL,

                tools_used JSONB
                    NOT NULL
                    DEFAULT '[]'::jsonb,

                tool_results JSONB
                    NOT NULL
                    DEFAULT '[]'::jsonb,

                answer TEXT,

                duration_ms BIGINT,

                status VARCHAR(20)
                    NOT NULL,

                error_message TEXT
            );
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_agent_audit_logs_created_at
            ON agent_audit_logs (
                created_at DESC
            );
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_agent_audit_logs_model
            ON agent_audit_logs (
                model
            );
            """
        )

        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_agent_audit_logs_status
            ON agent_audit_logs (
                status
            );
            """
        )


async def write_audit_log(
    app: FastAPI,
    request_id: str,
    model: str,
    user_message: str,
    tools_used: list[str],
    tool_results: list[
        dict[str, Any]
    ],
    answer: str | None,
    duration_ms: int,
    status: str,
    error_message: (
        str
        | None
    ) = None,
) -> None:

    try:

        async with (
            app.state.db_pool.acquire()
            as connection
        ):

            await connection.execute(
                """
                INSERT INTO agent_audit_logs (
                    request_id,
                    model,
                    user_message,
                    tools_used,
                    tool_results,
                    answer,
                    duration_ms,
                    status,
                    error_message
                )
                VALUES (
                    $1::uuid,
                    $2,
                    $3,
                    $4::jsonb,
                    $5::jsonb,
                    $6,
                    $7,
                    $8,
                    $9
                )
                """,
                request_id,
                model,
                user_message,
                json.dumps(
                    tools_used,
                    ensure_ascii=False,
                ),
                json.dumps(
                    tool_results,
                    ensure_ascii=False,
                ),
                answer,
                duration_ms,
                status,
                error_message,
            )

    except Exception:

        logger.exception(
            "Failed to write "
            "audit log"
        )


# =========================================================
# Agent Runtime
# =========================================================

async def run_agent(
    app: FastAPI,
    message: str,
    model_id: str,
):

    request_id = str(
        uuid.uuid4()
    )

    started = (
        time.perf_counter()
    )

    agent, tools = (
        get_agent_for_model(
            app,
            model_id,
        )
    )

    logger.info(
        "Agent request started "
        "request_id=%s "
        "model=%s",
        request_id,
        model_id,
    )

    try:

        result = (
            await asyncio.wait_for(
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
                timeout=(
                    AGENT_TIMEOUT
                ),
            )
        )

        result_messages = (
            result["messages"]
        )

        answer = (
            extract_final_answer(
                result_messages
            )
        )

        (
            tools_used,
            tool_results,
        ) = extract_tool_audit(
            result_messages
        )

        duration_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        await write_audit_log(
            app=app,
            request_id=request_id,
            model=model_id,
            user_message=message,
            tools_used=tools_used,
            tool_results=tool_results,
            answer=answer,
            duration_ms=duration_ms,
            status="success",
        )

        logger.info(
            "Agent request completed "
            "request_id=%s "
            "duration_ms=%s "
            "tools=%s",
            request_id,
            duration_ms,
            tools_used,
        )

        debug_messages = []

        for msg in (
            result_messages
        ):

            debug_messages.append(
                {
                    "type":
                        msg
                        .__class__
                        .__name__,

                    "content":
                        serialize_content(
                            getattr(
                                msg,
                                "content",
                                None,
                            )
                        ),

                    "tool_calls":
                        serialize_content(
                            getattr(
                                msg,
                                "tool_calls",
                                None,
                            )
                        ),

                    "name":
                        getattr(
                            msg,
                            "name",
                            None,
                        ),
                }
            )

        return (
            answer,
            debug_messages,
            tools,
            request_id,
        )

    except asyncio.TimeoutError as exc:

        duration_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        await write_audit_log(
            app=app,
            request_id=request_id,
            model=model_id,
            user_message=message,
            tools_used=[],
            tool_results=[],
            answer=None,
            duration_ms=duration_ms,
            status="timeout",
            error_message=(
                f"Agent timeout after "
                f"{AGENT_TIMEOUT} seconds"
            ),
        )

        raise HTTPException(
            status_code=504,
            detail=(
                f"Agent timeout after "
                f"{AGENT_TIMEOUT} seconds"
            ),
        ) from exc

    except Exception as exc:

        duration_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        await write_audit_log(
            app=app,
            request_id=request_id,
            model=model_id,
            user_message=message,
            tools_used=[],
            tool_results=[],
            answer=None,
            duration_ms=duration_ms,
            status="error",
            error_message=str(
                exc
            ),
        )

        logger.exception(
            "Agent request failed "
            "request_id=%s",
            request_id,
        )

        raise


# =========================================================
# FastAPI Lifespan
# =========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    logger.info(
        "Connecting PostgreSQL: "
        "%s:%s/%s",
        POSTGRES_HOST,
        POSTGRES_PORT,
        POSTGRES_DB,
    )

    db_pool = (
        await asyncpg.create_pool(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            min_size=1,
            max_size=5,
        )
    )

    await init_database(
        db_pool
    )

    app.state.db_pool = (
        db_pool
    )

    logger.info(
        "Connecting MCP servers"
    )

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
        if (
            tool.name
            != "search_paloalto_docs"
        )
    ]

    rag_tools = list(
        all_tools
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

    model = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=(
            OLLAMA_BASE_URL
        ),
        api_key="ollama",
        temperature=0,
    )

    # =====================================================
    # Existing Agent: paloalto-agent
    # Now actually implemented by LangGraph StateGraph
    # =====================================================

    base_agent = (
        build_agent_graph(
            agent_name=(
                BASE_AGENT_MODEL_ID
            ),
            model=model,
            tools=base_tools,
            system_prompt=(
                BASE_SYSTEM_PROMPT
            ),
        )
    )

    # =====================================================
    # Existing Agent: paloalto-rag-agent
    # Now actually implemented by LangGraph StateGraph
    # =====================================================

    rag_agent = (
        build_agent_graph(
            agent_name=(
                RAG_AGENT_MODEL_ID
            ),
            model=model,
            tools=rag_tools,
            system_prompt=(
                RAG_SYSTEM_PROMPT
            ),
        )
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
        "LangGraph agents initialized: "
        "%s, %s",
        BASE_AGENT_MODEL_ID,
        RAG_AGENT_MODEL_ID,
    )

    yield

    await db_pool.close()


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="AI Network Agent",
    version="0.8.0",
    lifespan=lifespan,
)


# =========================================================
# Health
# =========================================================

@app.get("/health")
async def health(
    request: Request,
):

    return {
        "status":
            "ok",

        "llm_model":
            OLLAMA_MODEL,

        "llm_endpoint":
            OLLAMA_BASE_URL,

        "agent_framework":
            "langgraph",

        "database":
            "connected",

        "models": {

            BASE_AGENT_MODEL_ID: {
                "framework":
                    "langgraph",

                "graph":
                    (
                        "START -> agent -> "
                        "tools -> agent / END"
                    ),

                "tools": [
                    tool.name
                    for tool
                    in request.app.state.base_tools
                ],
            },

            RAG_AGENT_MODEL_ID: {
                "framework":
                    "langgraph",

                "graph":
                    (
                        "START -> agent -> "
                        "tools -> agent / END"
                    ),

                "tools": [
                    tool.name
                    for tool
                    in request.app.state.rag_tools
                ],
            },
        },
    }


# =========================================================
# OpenAI Models API
# =========================================================

@app.get("/v1/models")
async def models(
    authorization: (
        str
        | None
    ) = Header(
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
        "object":
            "list",

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


# =========================================================
# OpenAI Chat Completions API
# =========================================================

@app.post(
    "/v1/chat/completions"
)
async def openai_chat_completions(
    body: OpenAIChatRequest,
    request: Request,
    authorization: (
        str
        | None
    ) = Header(
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

                (
                    answer,
                    _,
                    _,
                    _,
                ) = await run_agent(
                    request.app,
                    user_message,
                    body.model,
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
                            "index":
                                0,

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
                            "index":
                                0,

                            "delta":
                                {},

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

    (
        answer,
        _,
        _,
        request_id,
    ) = await run_agent(
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

        "audit_request_id":
            request_id,

        "choices": [
            {
                "index":
                    0,

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
            "prompt_tokens":
                0,

            "completion_tokens":
                0,

            "total_tokens":
                0,
        },
    }


# =========================================================
# Debug API
# =========================================================

@app.post("/chat")
async def debug_chat(
    body: ChatRequest,
    request: Request,
):

    (
        answer,
        debug_messages,
        tools,
        request_id,
    ) = await run_agent(
        request.app,
        body.message,
        body.model,
    )

    return {
        "request_id":
            request_id,

        "answer":
            answer,

        "model":
            body.model,

        "framework":
            "langgraph",

        "available_tools": [
            tool.name
            for tool in tools
        ],

        "messages":
            debug_messages,
    }