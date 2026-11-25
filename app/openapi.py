from typing import Any, Dict, Type

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from app.schemas.query import V1QueryRequest
from app.schemas.socket import (
    SocketErrorEvent,
    SocketJoinRoomPayload,
    SocketQuestionPayload,
    SocketResponseEvent,
    SocketUpdateEvent,
)
from app.schemas.threads import (
    RecordThreadRunRequest,
    RenameThreadRequest,
    ThreadContextUpdateRequest,
)

API_TITLE = "Rejuve.Bio AI Assistant API"
API_VERSION = "1.0.0"

API_DESCRIPTION = """
Backend HTTP and real-time Socket.IO API for the Rejuve.Bio AI Assistant.

## Authentication
All endpoints (except `GET /`) require a valid HS256 JWT in the `Authorization` header:
`Authorization: Bearer <token>` (raw `<token>` without the `Bearer` prefix is also accepted by legacy clients).
Missing or invalid tokens return HTTP `403` with `{"text": "Token is missing!"}` or `{"text": "Token is invalid!"}`.

## Rate Limiting
SlowAPI applies a default limit of **200 requests/minute** per client IP (returns HTTP `429` when exceeded).

## Real-Time Socket.IO Protocol (`/socket.io/`)
In addition to HTTP routes, the server exposes a Socket.IO ASGI endpoint mounted at `/socket.io/` (`app/socket_manager.py`).

### Connection & Authentication
- **Path:** `/socket.io/`
- **Header:** `Authorization: Bearer <token>` during the Socket.IO handshake. Invalid or missing tokens reject the connection (`connect` returns `false`).
- On connect, the client is automatically joined to its personal room `{user_id}` and receives a `response` event: `{"message": "Connected successfully"}`.

### Client -> Server Events
| Event | Payload Schema | Behavior |
| --- | --- | --- |
| `connect` | Handshake `Authorization` header | Authenticates JWT, saves session, auto-joins `{user_id}`, emits `response`. |
| `disconnect` | — | Leaves `{user_id}` room and cleans up session. |
| `join_room` | `SocketJoinRoomPayload` (`{"thread_id": "..."}`) | Joins room `{user_id}:{thread_id}` (or `{user_id}` if omitted) and emits `response`. |
| `question` | `SocketQuestionPayload` (`{"question": "...", "thread_id": "..."}`) | Acquires per-thread Redis lock, invokes the orchestrator, and emits progress `update` events followed by `response` (`status: "completed"`) to the room. |

### Server -> Client Events
| Event | Payload Schema | When Emitted |
| --- | --- | --- |
| `response` | `SocketResponseEvent` | On `connect`, `join_room`, busy rejection (`status: "busy"`), or final `question` completion (`status: "completed"`). |
| `update` | `SocketUpdateEvent` | Intermediate progress/status updates streamed while processing a `question`. |
| `error` | `SocketErrorEvent` | Validation errors (missing/empty `question`) or unhandled processing exceptions. |
""".strip()

TAGS_METADATA = [
    {
        "name": "Threads v1",
        "description": "Thread lifecycle, message history, context configuration, and run-recording endpoints.",
    },
    {
        "name": "Query & Files v1",
        "description": "Threads v1 file upload and proposal-oriented query endpoints.",
    },
    {
        "name": "Legacy",
        "description": "Deprecated pre-v1 endpoints retained for backward compatibility.",
    },
    {
        "name": "Health",
        "description": "Service liveness check.",
    },
]

_EXTRA_SCHEMAS: list[Type[BaseModel]] = [
    V1QueryRequest,
    ThreadContextUpdateRequest,
    RecordThreadRunRequest,
    RenameThreadRequest,
    SocketJoinRoomPayload,
    SocketQuestionPayload,
    SocketUpdateEvent,
    SocketResponseEvent,
    SocketErrorEvent,
]

_PUBLIC_OPERATIONS = {("/", "get")}


def schema_ref(model: Type[BaseModel]) -> Dict[str, str]:
    """Returns an OpenAPI $ref pointing to a model registered in components.schemas."""
    return {"$ref": f"#/components/schemas/{model.__name__}"}


def json_request_body(
    model: Type[BaseModel],
    description: str,
    required: bool = True,
) -> Dict[str, Any]:
    """Builds an OpenAPI requestBody block for handlers reading raw `await request.json()`."""
    return {
        "requestBody": {
            "required": required,
            "description": description,
            "content": {
                "application/json": {
                    "schema": schema_ref(model),
                }
            },
        }
    }


def _strip_authorization_header_param(operation: Dict[str, Any]) -> None:
    params = operation.get("parameters")
    if not params:
        return
    filtered = [
        p for p in params
        if not (p.get("in") == "header" and p.get("name", "").lower() == "authorization")
    ]
    if filtered:
        operation["parameters"] = filtered
    else:
        operation.pop("parameters", None)


def _register_extra_schemas(openapi_schema: Dict[str, Any]) -> None:
    components = openapi_schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    _, generated = models_json_schema(
        [(m, "validation") for m in _EXTRA_SCHEMAS],
        ref_template="#/components/schemas/{model}",
    )
    for name, definition in generated.get("$defs", {}).items():
        schemas.setdefault(name, definition)


def _socketio_extension() -> Dict[str, Any]:
    return {
        "path": "/socket.io/",
        "auth": {
            "header": "Authorization",
            "format": "Bearer <jwt>",
        },
        "rooms": ["{user_id}", "{user_id}:{thread_id}"],
        "client_events": {
            "connect": {"description": "Authenticate via HTTP Authorization header and join {user_id} room."},
            "disconnect": {"description": "Leave user room on disconnect."},
            "join_room": {"payload": schema_ref(SocketJoinRoomPayload)},
            "question": {"payload": schema_ref(SocketQuestionPayload)},
        },
        "server_events": {
            "response": {"payload": schema_ref(SocketResponseEvent)},
            "update": {"payload": schema_ref(SocketUpdateEvent)},
            "error": {"payload": schema_ref(SocketErrorEvent)},
        },
    }


def configure_openapi(app: FastAPI) -> None:
    """Attaches a customized OpenAPI schema generator with BearerAuth, extra schemas, and Socket.IO metadata."""

    def custom_openapi() -> Dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=API_TITLE,
            version=API_VERSION,
            description=API_DESCRIPTION,
            tags=TAGS_METADATA,
            routes=app.routes,
        )

        _register_extra_schemas(openapi_schema)

        components = openapi_schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "HS256 JWT containing `user_id`. Passed in the `Authorization` header.",
        }

        for path, path_item in openapi_schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if not isinstance(operation, dict):
                    continue
                _strip_authorization_header_param(operation)
                # OpenAPI 3.x forbids requestBody on GET; strip any emitted by legacy Form(...) params.
                if method.lower() == "get":
                    operation.pop("requestBody", None)
                # Deprecated DELETE routes also use Form(None) for legacy params (user_id, content_id, etc.)
                # that are actually resolved from the JWT — strip the misleading requestBody.
                if method.lower() == "delete" and operation.get("deprecated"):
                    operation.pop("requestBody", None)
                if (path, method.lower()) in _PUBLIC_OPERATIONS:
                    operation["security"] = []
                else:
                    operation["security"] = [{"BearerAuth": []}]

        openapi_schema["x-socketio"] = _socketio_extension()

        # Remove unreferenced Body_* schemas auto-generated by FastAPI for Form() params
        # whose requestBody was stripped above (GET and deprecated DELETE routes).
        import json as _json
        _spec_text = _json.dumps(openapi_schema)
        _schemas = openapi_schema.get("components", {}).get("schemas", {})
        for _name in list(_schemas):
            if _name.startswith("Body_") and f'"#/components/schemas/{_name}"' not in _spec_text:
                del _schemas[_name]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

