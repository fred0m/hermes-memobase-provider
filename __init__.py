# -*- coding: utf-8 -*-
"""Memobase memory provider plugin for Hermes Agent.

Connects Hermes's MemoryProvider lifecycle (prefetch / queue_prefetch /
sync_turn / tools) to a self-hosted Memobase server
(FastAPI + PostgreSQL + pgvector). Memobase is an open-source user-profile
memory backend; this plugin gives the agent *automatic* per-turn recall
instead of model-discretion search.

Recalls:   GET /users/context/{user_id}  (profile + event timeline assembly)
Writes:    POST /blobs/insert/{user_id}?wait_process=false  (async buffer write)
Flush:     POST /users/buffer/{user_id}/chat?wait_process=true  (session end)
Profiles:  GET /users/profile/{user_id}

Design notes
------------
* **prefetch keeps the hot path fast**: the actual recall runs in a
  background thread (``queue_prefetch``, fired after each completed turn) and
  ``prefetch()`` consumes the cached result, waiting at most
  ``_DEFAULT_PREFETCH_WAIT_SECS``. A slow backend degrades to "no injection"
  and the ``memobase_search`` tool remains the backstop — same pattern as
  Mem0's official plugin.
* **sync_turn is skipped for non-primary agents** (subagent / cron / flush):
  their system prompts would corrupt the user profile.
* **single in-flight writer**: sync turns reuse one thread and skip instead of
  piling up when the backend is slow (mirrors mem0).
* **fail-open**: any exception is logged and swallowed so the agent turn never
  breaks because memory died.
* **circuit breaker**: after 5 consecutive failures the provider pauses for
  60 s instead of hammering a down server; a success resets the counter.

Configuration
-------------
Secrets live in ``$HERMES_HOME/.env`` (or the environment):

    MEMOBASE_BASE_URL=http://<your-memobase-host>:8019/api/v1
    MEMOBASE_API_KEY=<your-memobase-token>
    MEMOBASE_USER_NAME=user            # optional; defaults to "user"

`MEMOBASE_USER_NAME` is mapped to a stable UUID via Memobase's own
``string_to_uuid`` scheme (uuid5(NAMESPACE_DNS, name + "memobase_client")),
so the provider reads/writes the SAME store as the memobase MCP server.

Behavioral settings (optional, ``$HERMES_HOME/memobase.json``):
    prefetch_max_tokens   context size returned per turn (default 2000)
    prefetch_wait_secs    how long prefetch blocks on a cold backend (5)
    fill_window_events    always include the recent event window (true)
    time_range_days       event window depth (default 180)
"""  # noqa: E501

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from agent.memory_provider import MemoryProvider, RecallStatus

logger = logging.getLogger(__name__)

_PKG = "memobase"

# Memobase's own id scheme: uuid5(NAMESPACE_DNS, name + "memobase_client").
_DEFAULT_USER_NAME = "user"

# Circuit breaker: after this many consecutive failures, pause API calls for
# _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_MAX_FAILS = 5
_BREAKER_COOLDOWN_SECS = 60.0

# Cold-backend block budget in prefetch() — long enough for a healthy local
# Memobase (hot path < 100ms) but short enough not to delay the LLM call.
_DEFAULT_PREFETCH_WAIT_SECS = 5.0
_DEFAULT_PREFETCH_MAX_TOKENS = 2000

# Guard against absurdly long single-turn prompts turning into HTTP 414 on the
# chats_str query parameter.
_MAX_QUERY_CHARS = 1500


def _load_json_config(hermes_home: str) -> dict:
    """Read optional $HERMES_HOME/memobase.json behavioral settings."""
    p = Path(hermes_home) / "memobase.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("[memobase] ignoring unparseable %s", p)
        return {}


class MemobaseMemoryProvider(MemoryProvider):
    """MemoryProvider implementation backed by a self-hosted Memobase server."""

    def __init__(self) -> None:
        self._base_url: Optional[str] = None
        self._api_key: Optional[str] = None
        self._user_id: Optional[str] = None
        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()
        self._prefetch_result: Optional[str] = None
        self._prefetch_for_query: Optional[str] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._writes_enabled = True
        self._fail_count = 0
        self._breaker_open_until = 0.0
        self._prefetch_max_tokens = _DEFAULT_PREFETCH_MAX_TOKENS
        self._prefetch_wait_secs = _DEFAULT_PREFETCH_WAIT_SECS
        self._fill_window_events = True
        self._time_range_days = 180
        self._sync_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Identity & config
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return _PKG

    def is_available(self) -> bool:
        """Ready only when the user has configured a Memobase base URL."""
        return bool(os.environ.get("MEMOBASE_BASE_URL", "").strip())

    def unavailable_reason(self) -> str:
        return (
            "Set MEMOBASE_BASE_URL (and optionally MEMOBASE_API_KEY) in "
            "~/.hermes/.env to enable the Memobase memory provider."
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = str(kwargs.get("hermes_home") or os.path.expanduser("~/.hermes"))
        agent_context = kwargs.get("agent_context") or "primary"
        if agent_context not in ("primary", "cli"):
            # subagent / cron / flush system prompts would corrupt the
            # user profile — keep recalling, stop writing.
            self._writes_enabled = False

        cfg = _load_json_config(hermes_home)
        self._prefetch_max_tokens = int(
            cfg.get(
                "prefetch_max_tokens",
                os.environ.get("MEMOBASE_PREFETCH_MAX_TOKENS", _DEFAULT_PREFETCH_MAX_TOKENS),
            )
        )
        self._prefetch_wait_secs = float(cfg.get("prefetch_wait_secs", _DEFAULT_PREFETCH_WAIT_SECS))
        self._fill_window_events = bool(cfg.get("fill_window_events", True))
        self._time_range_days = int(cfg.get("time_range_days", 180))

        self._base_url = os.environ.get("MEMOBASE_BASE_URL", "").rstrip("/")
        self._api_key = os.environ.get("MEMOBASE_API_KEY") or None
        user_name = os.environ.get("MEMOBASE_USER_NAME", _DEFAULT_USER_NAME)
        self._user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_name + "memobase_client"))

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=8.0,
        )
        logger.info(
            "[memobase] initialized base=%s user=%s writes=%s",
            self._base_url, self._user_id, self._writes_enabled,
        )

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # ------------------------------------------------------------------ #
    # System prompt & recall plumbing
    # ------------------------------------------------------------------ #
    def system_prompt_block(self) -> str:
        return (
            "You have access to a Memobase long-term memory backend. "
            "Relevant memories are injected automatically each turn inside the "
            "`<memory-context>` block. You can also search on demand with the "
            "`memobase_search` tool, or list the full user profile with "
            "`memobase_profile`."
        )

    def _breaker_open(self) -> bool:
        with self._lock:
            if self._fail_count >= _BREAKER_MAX_FAILS and time.time() < self._breaker_open_until:
                return True
            if time.time() >= self._breaker_open_until:
                self._fail_count = 0
            return False

    def _record_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if self._fail_count == _BREAKER_MAX_FAILS:
                self._breaker_open_until = time.time() + _BREAKER_COOLDOWN_SECS
                logger.warning(
                    "[memobase] circuit breaker opened for %.0fs", _BREAKER_COOLDOWN_SECS
                )

    def _record_success(self) -> None:
        with self._lock:
            self._fail_count = 0

    def _call_context(self, query: str) -> str:
        """GET /users/context/{user_id} — the same recall path as the MCP tools."""
        if self._client is None or self._user_id is None:
            return ""
        query = query[:_MAX_QUERY_CHARS]

        chats = json.dumps([{"role": "user", "content": query}], ensure_ascii=False)
        params = {
            "max_token_size": str(self._prefetch_max_tokens),
            "chats_str": chats,
            "time_range_in_days": str(self._time_range_days),
        }
        if self._fill_window_events:
            params["fill_window_with_events"] = "true"
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        r = self._client.get(f"/users/context/{self._user_id}?{qs}")
        r.raise_for_status()
        data = r.json()
        # Memobase can return HTTP 200 with a business error body
        # ({"errno": <nonzero>, "errmsg": ...}) — treat that as failure.
        errno = data.get("errno", 0)
        if errno:
            errmsg = data.get("errmsg") or f"errno={errno}"
            raise RuntimeError(f"Memobase context error: {errmsg}")
        ctx = (data.get("data") or {}).get("context") or ""
        self._record_success()
        return ctx.strip()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        if self._breaker_open():
            return ""
        with self._lock:
            cached = self._prefetch_result if self._prefetch_for_query == query else None
        if cached is not None:
            return cached or ""
        self.queue_prefetch(query, session_id=session_id)
        with self._lock:
            thread = self._prefetch_thread if self._prefetch_for_query == query else None
        if thread:
            thread.join(timeout=self._prefetch_wait_secs)
            with self._lock:
                cached = self._prefetch_result if self._prefetch_for_query == query else None
        if cached is not None:
            return cached or ""
        # Slow/unreachable backend: skip injection; memobase_search tool stays
        # available as the backstop.
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background recall whose result is consumed by the next prefetch."""
        if self._breaker_open():
            return
        with self._lock:
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                return
            self._prefetch_for_query = query
            self._prefetch_result = None

        def _run() -> None:
            try:
                ctx = self._call_context(query)
                with self._lock:
                    if self._prefetch_for_query == query:
                        self._prefetch_result = ctx
            except Exception as exc:
                logger.warning("[memobase] prefetch failed: %s", exc)
                self._record_failure()

        t = threading.Thread(target=_run, daemon=True, name="memobase-prefetch")
        with self._lock:
            self._prefetch_thread = t
        t.start()

    def recall_status(self) -> Optional[RecallStatus]:
        """Report what the last prefetch injected (deterministic, model-free)."""
        with self._lock:
            has = self._prefetch_result is not None
        if not has:
            return None
        return RecallStatus(provider_label="memobase", count=0, glyph="🧠")

    # ------------------------------------------------------------------ #
    # Turn sync (writes)
    # ------------------------------------------------------------------ #
    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Send the completed turn to Memobase for background profile extraction.

        Uses the same insert path as the memobase MCP client
        (POST /blobs/insert/{user_id}, wait_process=false). We deliberately do
        NOT force a flush here: Memobase auto-flushes the buffer at its token
        threshold (1024 by default), and on_session_end() flushes explicitly.

        One in-flight writer at a time (mirrors the official mem0 plugin):
        a busy writer is awaited up to 5s; if still busy, the turn is skipped
        instead of piling up writes.
        """
        if not self._writes_enabled or self._breaker_open():
            return
        if not user_content or not assistant_content:
            return
        if self._client is None or self._user_id is None:
            return

        def _sync() -> None:
            try:
                body = {
                    "blob_type": "chat",
                    "blob_data": {
                        "messages": [
                            {"role": "user", "content": user_content[:_MAX_QUERY_CHARS * 2]},
                            {"role": "assistant", "content": assistant_content[:_MAX_QUERY_CHARS * 2]},
                        ]
                    },
                }
                r = self._client.post(
                    f"/blobs/insert/{self._user_id}?wait_process=false",
                    json=body,
                )
                r.raise_for_status()
                self._record_success()
            except Exception as exc:
                logger.warning("[memobase] sync_turn failed: %s", exc)
                self._record_failure()

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            if self._sync_thread and self._sync_thread.is_alive():
                return  # still busy — skip to avoid ingestion pileup
            self._sync_thread = threading.Thread(
                target=_sync, daemon=True, name="memobase-sync"
            )
            self._sync_thread.start()

    # ------------------------------------------------------------------ #
    # Tools (on-demand backstop)
    # ------------------------------------------------------------------ #
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memobase_search",
                "description": (
                    "Search the user's long-term memory (Memobase) for context "
                    "relevant to a question. Returns the user profile plus recent "
                    "event-timeline memories."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Max context characters (default 2000).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memobase_profile",
                "description": (
                    "Return the full user profile (topics and sub-topics) from "
                    "the Memobase backend."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name == "memobase_search":
            query = (args or {}).get("query", "")
            max_len = int((args or {}).get("max_length", 2000))
            if self._breaker_open():
                return json.dumps({"error": "Memobase temporarily unavailable"}, ensure_ascii=False)
            try:
                ctx = self._call_context(query)
                if max_len and len(ctx) > max_len:
                    ctx = ctx[:max_len]
                if not ctx:
                    return json.dumps({"note": "No relevant memories found."}, ensure_ascii=False)
                return json.dumps({"context": ctx}, ensure_ascii=False)
            except Exception as exc:
                logger.warning("[memobase] memobase_search failed: %s", exc)
                self._record_failure()
                return json.dumps({"error": str(exc)}, ensure_ascii=False)
        if tool_name == "memobase_profile":
            if self._breaker_open():
                return json.dumps({"error": "Memobase temporarily unavailable"}, ensure_ascii=False)
            try:
                r = self._client.get(f"/users/profile/{self._user_id}")
                r.raise_for_status()
                data = r.json()
                return json.dumps(data, ensure_ascii=False)
            except Exception as exc:
                logger.warning("[memobase] memobase_profile failed: %s", exc)
                self._record_failure()
                return json.dumps({"error": str(exc)}, ensure_ascii=False)
        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )

    # ------------------------------------------------------------------ #
    # Optional hooks
    # ------------------------------------------------------------------ #
    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session flush: the buffered conversation gets profile-extracted."""
        if not self._writes_enabled or self._client is None or self._user_id is None:
            return
        try:
            r = self._client.post(
                f"/users/buffer/{self._user_id}/chat?wait_process=true",
                timeout=30.0,
            )
            r.raise_for_status()
            self._record_success()
        except Exception as exc:
            logger.warning("[memobase] on_session_end flush failed: %s", exc)
            self._record_failure()


def register(ctx: Any) -> None:
    """Register Memobase as a memory provider plugin."""
    ctx.register_memory_provider(MemobaseMemoryProvider())