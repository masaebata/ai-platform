import os
import secrets
from contextlib import asynccontextmanager

import asyncpg

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)

from fastapi.responses import HTMLResponse
from fastapi.security import (
    HTTPBasic,
    HTTPBasicCredentials,
)


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


ADMIN_USERNAME = os.getenv(
    "ADMIN_USERNAME",
    "admin",
)

ADMIN_PASSWORD = os.environ[
    "ADMIN_PASSWORD"
]


security = HTTPBasic()


def authenticate(
    credentials: HTTPBasicCredentials = Depends(
        security
    ),
) -> str:

    username_ok = secrets.compare_digest(
        credentials.username,
        ADMIN_USERNAME,
    )

    password_ok = secrets.compare_digest(
        credentials.password,
        ADMIN_PASSWORD,
    )

    if not (
        username_ok
        and password_ok
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={
                "WWW-Authenticate": "Basic"
            },
        )

    return credentials.username


async def init_database(
    pool: asyncpg.Pool,
) -> None:

    async with pool.acquire() as connection:

        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_audit_logs (
                id BIGSERIAL PRIMARY KEY,
                request_id UUID NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

                model VARCHAR(100) NOT NULL,
                user_message TEXT NOT NULL,

                tools_used JSONB NOT NULL DEFAULT '[]'::jsonb,
                tool_results JSONB NOT NULL DEFAULT '[]'::jsonb,

                answer TEXT,

                duration_ms BIGINT,

                status VARCHAR(20) NOT NULL,

                error_message TEXT
            );
            """
        )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
        min_size=1,
        max_size=5,
    )

    await init_database(
        pool
    )

    app.state.db_pool = pool

    yield

    await pool.close()


app = FastAPI(
    title="AI Platform Admin",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get(
    "/health"
)
async def health():

    return {
        "status": "ok"
    }


@app.get(
    "/api/summary"
)
async def summary(
    request: Request,
    _: str = Depends(
        authenticate
    ),
):

    async with (
        request.app.state.db_pool.acquire()
        as connection
    ):

        row = await connection.fetchrow(
            """
            SELECT
                COUNT(*) AS total_requests,

                COUNT(*) FILTER (
                    WHERE status = 'success'
                ) AS successful_requests,

                COUNT(*) FILTER (
                    WHERE status = 'error'
                ) AS error_requests,

                COUNT(*) FILTER (
                    WHERE status = 'timeout'
                ) AS timeout_requests,

                COALESCE(
                    AVG(duration_ms),
                    0
                ) AS average_duration_ms,

                COUNT(*) FILTER (
                    WHERE model = 'paloalto-rag-agent'
                ) AS rag_requests,

                COUNT(*) FILTER (
                    WHERE tools_used @> '["search_paloalto_docs"]'::jsonb
                ) AS rag_tool_calls

            FROM agent_audit_logs;
            """
        )

    total = (
        row["total_requests"]
        or 0
    )

    success = (
        row["successful_requests"]
        or 0
    )

    rag_requests = (
        row["rag_requests"]
        or 0
    )

    success_rate = (
        (success / total) * 100
        if total
        else 0
    )

    rag_rate = (
        (rag_requests / total) * 100
        if total
        else 0
    )

    return {
        "total_requests":
            total,

        "successful_requests":
            success,

        "error_requests":
            row["error_requests"]
            or 0,

        "timeout_requests":
            row["timeout_requests"]
            or 0,

        "success_rate":
            round(
                success_rate,
                1,
            ),

        "average_duration_ms":
            int(
                row[
                    "average_duration_ms"
                ]
                or 0
            ),

        "rag_requests":
            rag_requests,

        "rag_rate":
            round(
                rag_rate,
                1,
            ),

        "rag_tool_calls":
            row["rag_tool_calls"]
            or 0,
    }


@app.get(
    "/api/model-usage"
)
async def model_usage(
    request: Request,
    _: str = Depends(
        authenticate
    ),
):

    async with (
        request.app.state.db_pool.acquire()
        as connection
    ):

        rows = await connection.fetch(
            """
            SELECT
                model,
                COUNT(*) AS request_count,
                COALESCE(
                    AVG(duration_ms),
                    0
                ) AS average_duration_ms
            FROM agent_audit_logs
            GROUP BY model
            ORDER BY request_count DESC;
            """
        )

    return [
        {
            "model":
                row["model"],

            "request_count":
                row["request_count"],

            "average_duration_ms":
                int(
                    row[
                        "average_duration_ms"
                    ]
                    or 0
                ),
        }

        for row in rows
    ]


@app.get(
    "/api/tool-usage"
)
async def tool_usage(
    request: Request,
    _: str = Depends(
        authenticate
    ),
):

    async with (
        request.app.state.db_pool.acquire()
        as connection
    ):

        rows = await connection.fetch(
            """
            SELECT
                tool_name,
                COUNT(*) AS usage_count

            FROM agent_audit_logs,

            LATERAL
                jsonb_array_elements_text(
                    tools_used
                ) AS tool_name

            GROUP BY tool_name
            ORDER BY usage_count DESC;
            """
        )

    return [
        {
            "tool":
                row["tool_name"],

            "count":
                row["usage_count"],
        }

        for row in rows
    ]


@app.get(
    "/api/recent"
)
async def recent(
    request: Request,
    _: str = Depends(
        authenticate
    ),
):

    async with (
        request.app.state.db_pool.acquire()
        as connection
    ):

        rows = await connection.fetch(
            """
            SELECT
                request_id,
                created_at,
                model,
                user_message,
                tools_used,
                duration_ms,
                status

            FROM agent_audit_logs

            ORDER BY created_at DESC

            LIMIT 30;
            """
        )

    return [
        {
            "request_id":
                str(
                    row[
                        "request_id"
                    ]
                ),

            "created_at":
                row[
                    "created_at"
                ].isoformat(),

            "model":
                row["model"],

            "user_message":
                row[
                    "user_message"
                ],

            "tools_used":
                row[
                    "tools_used"
                ],

            "duration_ms":
                row[
                    "duration_ms"
                ],

            "status":
                row["status"],
        }

        for row in rows
    ]


@app.get(
    "/api/errors"
)
async def errors(
    request: Request,
    _: str = Depends(
        authenticate
    ),
):

    async with (
        request.app.state.db_pool.acquire()
        as connection
    ):

        rows = await connection.fetch(
            """
            SELECT
                request_id,
                created_at,
                model,
                user_message,
                duration_ms,
                status,
                error_message

            FROM agent_audit_logs

            WHERE status <> 'success'

            ORDER BY created_at DESC

            LIMIT 20;
            """
        )

    return [
        {
            "request_id":
                str(
                    row[
                        "request_id"
                    ]
                ),

            "created_at":
                row[
                    "created_at"
                ].isoformat(),

            "model":
                row["model"],

            "user_message":
                row[
                    "user_message"
                ],

            "duration_ms":
                row[
                    "duration_ms"
                ],

            "status":
                row["status"],

            "error":
                row[
                    "error_message"
                ],
        }

        for row in rows
    ]


@app.get(
    "/api/request/{request_id}"
)
async def request_detail(
    request_id: str,
    request: Request,
    _: str = Depends(
        authenticate
    ),
):

    async with (
        request.app.state.db_pool.acquire()
        as connection
    ):

        row = await connection.fetchrow(
            """
            SELECT
                request_id,
                created_at,
                model,
                user_message,
                tools_used,
                tool_results,
                answer,
                duration_ms,
                status,
                error_message

            FROM agent_audit_logs

            WHERE request_id = $1::uuid;
            """,
            request_id,
        )

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Request not found",
        )

    return {
        "request_id":
            str(
                row[
                    "request_id"
                ]
            ),

        "created_at":
            row[
                "created_at"
            ].isoformat(),

        "model":
            row["model"],

        "user_message":
            row[
                "user_message"
            ],

        "tools_used":
            row[
                "tools_used"
            ],

        "tool_results":
            row[
                "tool_results"
            ],

        "answer":
            row["answer"],

        "duration_ms":
            row[
                "duration_ms"
            ],

        "status":
            row["status"],

        "error":
            row[
                "error_message"
            ],
    }


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ja">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
AI Platform Admin
</title>

<style>

body {
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 0;

    background:
        #f4f6f8;

    color:
        #1f2937;
}

header {
    background:
        #111827;

    color:
        white;

    padding:
        20px 30px;
}

header h1 {
    margin: 0;
}

header p {
    margin:
        5px 0 0 0;

    color:
        #cbd5e1;
}

main {
    padding:
        30px;

    max-width:
        1600px;

    margin:
        auto;
}

.cards {
    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                180px,
                1fr
            )
        );

    gap:
        16px;

    margin-bottom:
        30px;
}

.card {
    background:
        white;

    border-radius:
        10px;

    padding:
        20px;

    box-shadow:
        0 1px 4px
        rgba(
            0,
            0,
            0,
            0.10
        );
}

.card-title {
    font-size:
        13px;

    color:
        #6b7280;

    margin-bottom:
        10px;
}

.card-value {
    font-size:
        30px;

    font-weight:
        bold;
}

.panel {
    background:
        white;

    border-radius:
        10px;

    margin-bottom:
        25px;

    padding:
        20px;

    box-shadow:
        0 1px 4px
        rgba(
            0,
            0,
            0,
            0.10
        );
}

.panel h2 {
    margin-top:
        0;
}

table {
    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        14px;
}

th,
td {
    text-align:
        left;

    padding:
        10px;

    border-bottom:
        1px solid
        #e5e7eb;

    vertical-align:
        top;
}

th {
    background:
        #f9fafb;
}

.status-success {
    font-weight:
        bold;

    color:
        #15803d;
}

.status-error {
    font-weight:
        bold;

    color:
        #b91c1c;
}

.status-timeout {
    font-weight:
        bold;

    color:
        #b45309;
}

.message {
    max-width:
        500px;

    overflow-wrap:
        anywhere;
}

.tool {
    display:
        inline-block;

    background:
        #e5e7eb;

    padding:
        3px 7px;

    margin:
        2px;

    border-radius:
        5px;

    font-size:
        12px;
}

button {
    border:
        none;

    background:
        #2563eb;

    color:
        white;

    border-radius:
        6px;

    padding:
        8px 12px;

    cursor:
        pointer;
}

pre {
    white-space:
        pre-wrap;

    overflow-wrap:
        anywhere;

    background:
        #111827;

    color:
        #e5e7eb;

    padding:
        15px;

    border-radius:
        8px;
}

</style>

</head>


<body>

<header>

<h1>
AI Platform Admin
</h1>

<p>
Agent Audit & Operations Dashboard
</p>

</header>


<main>

<div
    id="cards"
    class="cards"
>
</div>


<div class="panel">

<h2>
Model Usage
</h2>

<div id="modelUsage">
Loading...
</div>

</div>


<div class="panel">

<h2>
Tool Usage
</h2>

<div id="toolUsage">
Loading...
</div>

</div>


<div class="panel">

<h2>
Recent Agent Activity
</h2>

<div id="recent">
Loading...
</div>

</div>


<div class="panel">

<h2>
Recent Errors
</h2>

<div id="errors">
Loading...
</div>

</div>


<div class="panel">

<h2>
Request Detail
</h2>

<div id="detail">
Click a request to inspect it.
</div>

</div>


</main>


<script>

function formatMs(ms) {

    if (
        ms === null
        || ms === undefined
    ) {
        return "-";
    }

    if (ms < 1000) {
        return ms + " ms";
    }

    return (
        (ms / 1000)
        .toFixed(1)
        + " sec"
    );
}


function escapeHtml(value) {

    if (
        value === null
        || value === undefined
    ) {
        return "";
    }

    return String(value)
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        );
}


async function api(path) {

    const response = await fetch(
        path
    );

    if (!response.ok) {

        throw new Error(
            "API error: "
            + response.status
        );
    }

    return await response.json();
}


async function loadSummary() {

    const data = await api(
        "/api/summary"
    );

    const cards = [
        [
            "Total Requests",
            data.total_requests
        ],

        [
            "Success Rate",
            data.success_rate + "%"
        ],

        [
            "Average Response",
            formatMs(
                data.average_duration_ms
            )
        ],

        [
            "RAG Usage",
            data.rag_rate + "%"
        ],

        [
            "Errors",
            data.error_requests
        ],

        [
            "Timeouts",
            data.timeout_requests
        ]
    ];

    document.getElementById(
        "cards"
    ).innerHTML = cards.map(
        item => `
            <div class="card">

                <div class="card-title">
                    ${escapeHtml(item[0])}
                </div>

                <div class="card-value">
                    ${escapeHtml(item[1])}
                </div>

            </div>
        `
    ).join("");
}


async function loadModelUsage() {

    const data = await api(
        "/api/model-usage"
    );

    document.getElementById(
        "modelUsage"
    ).innerHTML = `
        <table>

            <thead>
                <tr>
                    <th>Model</th>
                    <th>Requests</th>
                    <th>Average Response</th>
                </tr>
            </thead>

            <tbody>

                ${
                    data.map(
                        row => `
                            <tr>

                                <td>
                                    ${escapeHtml(row.model)}
                                </td>

                                <td>
                                    ${row.request_count}
                                </td>

                                <td>
                                    ${formatMs(
                                        row.average_duration_ms
                                    )}
                                </td>

                            </tr>
                        `
                    ).join("")
                }

            </tbody>

        </table>
    `;
}


async function loadToolUsage() {

    const data = await api(
        "/api/tool-usage"
    );

    document.getElementById(
        "toolUsage"
    ).innerHTML = `
        <table>

            <thead>
                <tr>
                    <th>Tool</th>
                    <th>Calls</th>
                </tr>
            </thead>

            <tbody>

                ${
                    data.map(
                        row => `
                            <tr>

                                <td>
                                    ${escapeHtml(row.tool)}
                                </td>

                                <td>
                                    ${row.count}
                                </td>

                            </tr>
                        `
                    ).join("")
                }

            </tbody>

        </table>
    `;
}


function statusClass(
    status
) {

    return (
        "status-"
        + status
    );
}


async function loadRecent() {

    const data = await api(
        "/api/recent"
    );

    document.getElementById(
        "recent"
    ).innerHTML = `
        <table>

            <thead>
                <tr>
                    <th>Time</th>
                    <th>Model</th>
                    <th>Message</th>
                    <th>Tools</th>
                    <th>Duration</th>
                    <th>Status</th>
                    <th></th>
                </tr>
            </thead>

            <tbody>

                ${
                    data.map(
                        row => {

                            const tools = (
                                row.tools_used
                                || []
                            ).map(
                                tool =>
                                    `<span class="tool">
                                        ${escapeHtml(tool)}
                                    </span>`
                            ).join("");

                            return `
                                <tr>

                                    <td>
                                        ${escapeHtml(
                                            new Date(
                                                row.created_at
                                            ).toLocaleString()
                                        )}
                                    </td>

                                    <td>
                                        ${escapeHtml(row.model)}
                                    </td>

                                    <td class="message">
                                        ${escapeHtml(
                                            row.user_message
                                        )}
                                    </td>

                                    <td>
                                        ${tools}
                                    </td>

                                    <td>
                                        ${formatMs(
                                            row.duration_ms
                                        )}
                                    </td>

                                    <td
                                        class="${
                                            statusClass(
                                                row.status
                                            )
                                        }"
                                    >
                                        ${escapeHtml(row.status)}
                                    </td>

                                    <td>

                                        <button
                                            onclick="
                                                loadDetail(
                                                    '${row.request_id}'
                                                )
                                            "
                                        >
                                            Detail
                                        </button>

                                    </td>

                                </tr>
                            `;
                        }
                    ).join("")
                }

            </tbody>

        </table>
    `;
}


async function loadErrors() {

    const data = await api(
        "/api/errors"
    );

    if (
        data.length === 0
    ) {

        document.getElementById(
            "errors"
        ).innerHTML =
            "No recent errors.";

        return;
    }

    document.getElementById(
        "errors"
    ).innerHTML = `
        <table>

            <thead>
                <tr>
                    <th>Time</th>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Error</th>
                </tr>
            </thead>

            <tbody>

                ${
                    data.map(
                        row => `
                            <tr>

                                <td>
                                    ${escapeHtml(
                                        new Date(
                                            row.created_at
                                        ).toLocaleString()
                                    )}
                                </td>

                                <td>
                                    ${escapeHtml(row.model)}
                                </td>

                                <td
                                    class="${
                                        statusClass(
                                            row.status
                                        )
                                    }"
                                >
                                    ${escapeHtml(row.status)}
                                </td>

                                <td class="message">
                                    ${escapeHtml(
                                        row.error
                                    )}
                                </td>

                            </tr>
                        `
                    ).join("")
                }

            </tbody>

        </table>
    `;
}


async function loadDetail(
    requestId
) {

    const data = await api(
        "/api/request/"
        + requestId
    );

    document.getElementById(
        "detail"
    ).innerHTML = `
        <pre>${
            escapeHtml(
                JSON.stringify(
                    data,
                    null,
                    2
                )
            )
        }</pre>
    `;
}


async function refresh() {

    try {

        await Promise.all(
            [
                loadSummary(),
                loadModelUsage(),
                loadToolUsage(),
                loadRecent(),
                loadErrors()
            ]
        );

    } catch (error) {

        console.error(
            error
        );
    }
}


refresh();

setInterval(
    refresh,
    30000
);

</script>

</body>

</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse,
)
async def dashboard(
    _: str = Depends(
        authenticate
    ),
):

    return HTMLResponse(
        DASHBOARD_HTML
    )