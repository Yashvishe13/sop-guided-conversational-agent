"""FastAPI application: HTTP contract, cookies/CSRF, error envelope, static UI.

No SOP logic lives here. Handlers validate input, resolve the session
credential, and delegate to :class:`ConversationService`.

Run with ``uvicorn insurance_claims.web.app:create_app --factory`` (or
``python -m insurance_claims``).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from insurance_claims import __version__
from insurance_claims.agent.loop import ClaimsAgent
from insurance_claims.agent.prompts import PromptSet, load_prompts
from insurance_claims.claims.fixtures import FixtureBundle, load_fixtures
from insurance_claims.config import APP_ROOT, ConfigError, Settings
from insurance_claims.llm.base import ModelTransport
from insurance_claims.mail.sender import (
    DisabledTransport,
    EmailDispatcher,
    EmailTransport,
    OutboxTransport,
    SmtpTransport,
)
from insurance_claims.observability import tracing
from insurance_claims.observability.redaction import build_fixture_redactor
from insurance_claims.persistence.store import SessionStore, load_cipher
from insurance_claims.web.service import ConversationService, ServiceError

log = logging.getLogger("insurance_claims.web.app")

COOKIE_NAME = "ic_session"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16_384
SESSION_ID_PATTERN = r"^[A-Za-z0-9_\-]{16,64}$"
TURN_ID_PATTERN = r"^[A-Za-z0-9_\-]{8,64}$"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_turn_id: str = Field(pattern=TURN_ID_PATTERN)
    text: str | None = Field(default=None, max_length=8000)
    action: Literal["email_send", "email_skip"] | None = None

    @model_validator(mode="after")
    def _one_of(self) -> TurnRequest:
        has_text = self.text is not None and self.text.strip() != ""
        if has_text == (self.action is not None):
            raise ValueError("send exactly one of text or action")
        return self


def default_settings() -> Settings:
    """Settings from the process environment plus the first ``.env`` found (process env wins)."""
    import os

    candidates = [Path(p) for p in [os.environ.get("DOTENV_PATH", "")] if p]
    candidates += [APP_ROOT / ".env", APP_ROOT.parents[1] / ".env"]
    dotenv = next((p for p in candidates if p.is_file()), None)
    return Settings.from_env(dotenv=dotenv)


def build_model(settings: Settings) -> ModelTransport:
    if settings.model_provider == "fake":
        from insurance_claims.llm.fake import OfflineFakeModel

        return OfflineFakeModel(today=settings.frozen_today)
    from insurance_claims.agent.budget import remaining_seconds
    from insurance_claims.llm.openai_transport import OpenAITransport
    from insurance_claims.llm.resilient import ResilientTransport

    return ResilientTransport(
        OpenAITransport(settings),
        max_retries=settings.model_max_retries,
        backoff_base_s=settings.model_backoff_base_s,
        deadline=remaining_seconds,
    )


def build_email_transport(settings: Settings) -> EmailTransport:
    if settings.email_transport == "smtp":
        return SmtpTransport(
            host=settings.smtp_host or "",
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            starttls=settings.smtp_starttls,
        )
    if settings.email_transport == "disabled":
        return DisabledTransport()
    return OutboxTransport(settings.outbox_dir)


class Runtime:
    """Everything the app needs, built once at startup (fails fast on bad config)."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: ModelTransport | None = None,
        email_transport: EmailTransport | None = None,
        clock: Any = None,
    ) -> None:
        settings.validate_runtime()
        self.settings = settings
        self.fixtures: FixtureBundle = load_fixtures(settings.fixtures_dir)
        self.prompts: PromptSet = load_prompts(settings.prompts_path)
        secrets_list = [s for s in (settings.openai_api_key, settings.smtp_password, settings.state_encryption_key) if s]
        tracing.configure(
            trace_dir=settings.trace_dir,
            enabled=settings.traces_enabled,
            redactor=build_fixture_redactor(self.fixtures, secrets=secrets_list),
        )
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            settings.data_dir.chmod(0o700)
        except OSError:
            pass
        self.clock = clock or (lambda: datetime.now(UTC))
        cipher = load_cipher(settings)
        self.store = SessionStore(settings.db_path, cipher, clock=self.clock)
        self.store.init_schema()
        self.model = model or build_model(settings)
        self.email_transport = email_transport or build_email_transport(settings)
        self.agent = ClaimsAgent(
            settings=settings,
            fixtures=self.fixtures,
            prompts=self.prompts,
            model=self.model,
            clock=self.clock,
            failure_ledger=self.store,
        )
        self.dispatcher = EmailDispatcher(
            self.email_transport,
            self.store,
            from_address=settings.email_from,
            timeout=settings.email_timeout_s,
            clock=self.clock,
        )
        csrf_key = hashlib.sha256(b"csrf:" + cipher_key_material(settings)).digest()
        self.service = ConversationService(
            settings=settings,
            store=self.store,
            agent=self.agent,
            dispatcher=self.dispatcher,
            csrf_key=csrf_key,
            clock=self.clock,
        )
        try:
            self.service.purge()
        except Exception:
            log.warning("startup purge failed")

    def health(self) -> dict[str, Any]:
        base = self.service.health()
        checks = dict(base["checks"])
        checks["fixtures"] = {
            "policyholders": len(self.fixtures.policyholders),
            "claims": len(self.fixtures.claims),
            "quarantined_rows": len(self.fixtures.issues),
        }
        checks["prompts"] = {"version": self.prompts.version, "sha256": self.prompts.sha256[:12]}
        checks["model"] = {
            "provider": self.settings.model_provider,
            "model": self.settings.openai_model if self.settings.model_provider == "openai" else "fake",
            "configured": self.settings.model_provider == "fake" or bool(self.settings.openai_api_key),
        }
        checks["email_transport"] = self.email_transport.name
        status = base["status"]
        if status == "ok" and self.fixtures.issues:
            status = "degraded"
        return {"status": status, "checks": checks, "version": __version__}


def cipher_key_material(settings: Settings) -> bytes:
    if settings.state_encryption_key:
        return settings.state_encryption_key.encode()
    try:
        return settings.key_file.read_bytes()
    except OSError:
        return b"ephemeral"


def _error(status: int, code: str, message: str, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse({"error": {"code": code, "message": message, "request_id": request_id}}, status_code=status)


def _ascii_header(request: Request, name: str) -> str | None:
    """Header value, or ``""`` when it is not ASCII (credential tokens are hex; never pass bytes we cannot compare)."""
    value = request.headers.get(name)
    if value is None:
        return None
    return value if value.isascii() else ""


def _source_matches_host(source: str, headers: Headers) -> bool:
    """True when the ``Origin``/``Referer`` URL names the host this request was sent to.

    ``Host`` is compared first; ``X-Forwarded-Host`` is accepted too so a reverse proxy that rewrites
    ``Host`` keeps working. A cross-site page cannot set either header on a form post or a simple
    request, so neither weakens the check. ``Origin: null`` and non-HTTP origins never match.
    """
    try:
        parts = urlsplit(source.strip())
        port = parts.port
    except ValueError:
        return False
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        return False
    default_port = _DEFAULT_PORTS[parts.scheme]
    want = (parts.hostname.lower(), port or default_port)
    forwarded = (headers.get("x-forwarded-host") or "").split(",")[0]
    for candidate in (headers.get("host"), forwarded):
        if not candidate or not candidate.strip():
            continue
        try:
            host = urlsplit("//" + candidate.strip())
            host_port = host.port
        except ValueError:
            continue
        if host.hostname and (host.hostname.lower(), host_port or default_port) == want:
            return True
    return False


def is_cross_site_write(method: str, path: str, headers: Headers) -> bool:
    """A state-changing ``/api`` request that a browser sent from another site.

    Browsers always attach ``Origin`` to POST/DELETE (``null`` when the page hides it); older ones may
    send only ``Referer``. ``Sec-Fetch-Site`` is checked too when present. Requests carrying none of
    these headers come from non-browser clients, which cannot ride on a victim's cookie, so they pass.
    """
    if method.upper() in SAFE_METHODS or not path.startswith("/api/"):
        return False
    fetch_site = (headers.get("sec-fetch-site") or "").strip().lower()
    if fetch_site in {"cross-site", "same-site"}:
        return True
    source = headers.get("origin")
    if source is None:
        source = headers.get("referer")
    if source is None:
        return False
    return not _source_matches_host(source, headers)


class _BodyTooLarge(StarletteHTTPException):
    def __init__(self) -> None:
        super().__init__(413)


class Envelope:
    """Outermost ASGI middleware: request id, body-size limit, same-origin writes, headers, last-resort 500.

    It is a pure ASGI middleware (not ``@app.middleware("http")``) so it can count streamed request
    bytes (chunked bodies carry no ``Content-Length``) and so every response it sees, including the
    500 it builds for an unhandled exception, gets the security headers, ``Cache-Control: no-store``
    on API paths, and ``X-Request-ID``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id
        request = Request(scope)
        path: str = scope.get("path", "")
        no_store = path.startswith("/api/") or path == "/health"
        started = time.monotonic()
        status: int | None = None
        received = 0

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                headers = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)
                if no_store:
                    headers["Cache-Control"] = "no-store"
                headers["X-Request-ID"] = request_id
            await send(message)

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    raise _BodyTooLarge()
            return message

        try:
            length = request.headers.get("content-length")
            if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
                await _error(413, "payload_too_large", "The message is too large.", request)(scope, receive, send_with_headers)
            elif is_cross_site_write(request.method, path, request.headers):
                await _error(403, "cross_site_request", "This request must come from the claims support page.", request)(
                    scope, receive, send_with_headers
                )
            else:
                await self.app(scope, limited_receive, send_with_headers)
        except Exception as exc:
            if status is None:
                if isinstance(exc, StarletteHTTPException) and exc.status_code == 413:
                    response = _error(413, "payload_too_large", "The message is too large.", request)
                else:
                    log.error("unhandled %s on %s", exc.__class__.__name__, path)
                    response = _error(500, "internal_error", "Something went wrong on our side. Please try again.", request)
                await response(scope, receive, send_with_headers)
            else:
                log.error("unhandled %s on %s after the response started", exc.__class__.__name__, path)
        finally:
            route = scope.get("route")
            log.info("%s %s %s %.0fms", request.method, getattr(route, "path", "-"), status, (time.monotonic() - started) * 1000)


def create_app(
    settings: Settings | None = None,
    *,
    model: ModelTransport | None = None,
    email_transport: EmailTransport | None = None,
    clock: Any = None,
) -> FastAPI:
    settings = settings or default_settings()
    try:
        runtime = Runtime(settings, model=model, email_transport=email_transport, clock=clock)
    except ConfigError:
        raise
    app = FastAPI(title="Insurance claims support agent", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runtime = runtime
    service = runtime.service

    # Outermost user middleware: it wraps the router and the exception handlers below, so their
    # responses (and the 500 it builds itself) all get the same headers and request id.
    app.add_middleware(Envelope)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) or "body" for e in exc.errors()})
        return _error(400, "invalid_request", f"Invalid request ({', '.join(fields)[:120]}).", request)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 413:  # raised while a streamed (chunked) body is being read
            return _error(413, "payload_too_large", "The message is too large.", request)
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, "Not found." if exc.status_code == 404 else "Request not allowed.", request)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled %s on %s", exc.__class__.__name__, request.url.path)
        return _error(500, "internal_error", "Something went wrong on our side. Please try again.", request)

    def _check_session_id(session_id: str) -> None:
        import re

        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            raise ServiceError(404, "session_not_found", "This conversation was not found. Please start a new one.")

    def _set_cookie(response: Response, secret: str) -> None:
        response.set_cookie(
            COOKIE_NAME,
            secret,
            max_age=settings.session_max_age_hours * 3600,
            path="/api",
            httponly=True,
            samesite="strict",
            secure=settings.cookie_secure,
        )

    @app.post("/api/sessions", status_code=201)
    def create_session(response: Response) -> dict[str, Any]:
        _session_id, secret, payload = service.create_session()
        _set_cookie(response, secret)
        return payload

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, request: Request) -> dict[str, Any]:
        _check_session_id(session_id)
        return service.get_session(session_id, request.cookies.get(COOKIE_NAME))

    @app.post("/api/sessions/{session_id}/messages")
    def post_message(session_id: str, body: TurnRequest, request: Request) -> dict[str, Any]:
        _check_session_id(session_id)
        if body.text is not None and len(body.text) > settings.max_message_chars:
            raise ServiceError(413, "message_too_long", f"Please keep messages under {settings.max_message_chars} characters.")
        return service.post_turn(
            session_id,
            request.cookies.get(COOKIE_NAME),
            _ascii_header(request, "x-csrf-token"),
            client_turn_id=body.client_turn_id,
            text=body.text if body.action is None else None,
            action=body.action,
        )

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, request: Request) -> Response:
        _check_session_id(session_id)
        service.delete_session(session_id, request.cookies.get(COOKIE_NAME), _ascii_header(request, "x-csrf-token"))
        response = Response(status_code=204)
        response.delete_cookie(COOKIE_NAME, path="/api")
        return response

    @app.get("/health")
    def health() -> JSONResponse:
        data = runtime.health()
        return JSONResponse(data, status_code=200 if data["status"] != "error" else 503)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
