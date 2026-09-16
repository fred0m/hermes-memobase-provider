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
    prefetch_max_tokens        context size returned per turn (default 2000)
    prefetch_wait_secs         how long prefetch blocks on a cold backend (5)
    fill_window_events         always include the recent event window (true)
    time_range_days            event window depth (default 180)
    temporal_boost_enabled     enable temporal boost on time-window queries (default true)
    temporal_gain              in-window score multiplier (default 1.6)
    temporal_half_life_days    exponential decay half life for out-of-window events (default 30)
    temporal_floor             decay floor multiplier (default 0.6)
    timezone                   local timezone for date-diff computation (default "Asia/Shanghai")
    entity_boost_enabled       enable entity boost via inverted index RRF leg (default true)
    anchor_backfill_enabled    enable anchor backfill injection on pure semantic queries (default true)
    anchor_topk                number of anchor dates to extract from semantic summaries (default 2)
    anchor_events_per_day      max events to inject per missing anchor date (default 2)
    anchor_budget_tokens       max budget cap for anchor backfill lines (default 200)
    cjk_known_names            extra CJK proper nouns that always count as entities
                               (e.g. ["Alice", "Bob"]-style names); default [] — the
                               public build ships no personal names, deployments add
                               their own here
    rerank_enabled             allow the optional cross-encoder rerank pass (default true)
    rerank_mode                auto | always | never (default auto)
    rerank_base_url            OpenAI-compatible endpoint serving the reranker;
                               NO default — rerank runs only when both this and
                               rerank_api_key are set
    rerank_api_key             key for that endpoint (falls back to MEMOBASE_RERANK_API_KEY)
    recall_label               text shown by the recall indicator (default "memobase")
    recall_glyph               emoji shown by the recall indicator (default "🧠")
"""  # noqa: E501

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from agent.memory_provider import MemoryProvider, RecallStatus

from .hybrid_retriever import (
    _CJK_RE,
    _LATIN_RE,
    BM25Index,
    EventStore,
    extract_entities,
    get_timezone,
    load_cjk_known_names,
    low_discrimination,
    parse_created_at,
    parse_time_window,
    rerank,
    rrf_fusion,
    rrf_fusion_scored,
    set_cjk_known_names,
    temporal_factor,
)

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


def _as_bool(value: Any, default: bool = False) -> bool:
    """Safe bool parsing — a JSON/config string "false" must not become True."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_int(value: Any, default: int = 0) -> int:
    """Safe int parsing — None/empty string fallback to default."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
        self._hybrid_enabled = True
        self._hybrid_vector_topk = 20
        self._hybrid_bm25_topk = 20
        self._hybrid_keep = 10
        self._hybrid_event_budget_tokens = 900
        # Summary layer config (FR-010)
        self._summary_enabled = False
        self._summary_budget_tokens = 350
        self._event_budget_tokens = 850
        self._summary_topk = 2
        self._summary_semantic_topk = 3
        # Anchor backfill config (FR-019)
        self._anchor_backfill_enabled = True
        self._anchor_topk = 2
        self._anchor_events_per_day = 2
        self._anchor_budget_tokens = 200
        self._event_cache_ttl = 300.0
        self._temporal_boost_enabled = True
        self._temporal_gain = 1.6
        self._temporal_half_life_days = 30.0
        self._temporal_floor = 0.6
        self._timezone = "Asia/Shanghai"
        self._tz = get_timezone("Asia/Shanghai")
        self._entity_boost_enabled = True
        self._rerank_enabled = True
        self._rerank_mode = "auto"  # auto | always | never
        self._rerank_model = "Qwen/Qwen3-Reranker-4B"
        self._rerank_base_url = ""
        self._rerank_api_key = ""
        self._rerank_topk = 15
        self._rerank_keep = 8
        self._rerank_ratio_threshold = 1.01
        self._rerank_overlap_threshold = 0.3
        self._rerank_sim_threshold = 0.3
        self._recall_label = "memobase"
        self._recall_glyph = "🧠"
        self._store: Optional[EventStore] = None
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

        # Deployment-specific CJK proper nouns (memobase.json: cjk_known_names).
        # Kept out of the public source tree; empty list = upstream default.
        set_cjk_known_names(load_cjk_known_names(hermes_home))

        # Recall indicator label. The public default is the provider name;
        # a deployment can brand it (memobase.json: recall_label / recall_glyph).
        self._recall_label = str(cfg.get("recall_label") or "").strip() or "memobase"
        self._recall_glyph = str(cfg.get("recall_glyph") or "").strip() or "🧠"
        self._prefetch_max_tokens = int(
            cfg.get(
                "prefetch_max_tokens",
                os.environ.get("MEMOBASE_PREFETCH_MAX_TOKENS", _DEFAULT_PREFETCH_MAX_TOKENS),
            )
        )
        self._prefetch_wait_secs = float(cfg.get("prefetch_wait_secs", _DEFAULT_PREFETCH_WAIT_SECS))
        self._fill_window_events = bool(cfg.get("fill_window_events", True))
        self._time_range_days = int(cfg.get("time_range_days", 180))
        self._hybrid_enabled = _as_bool(
            cfg.get("hybrid_enabled", os.environ.get("MEMOBASE_HYBRID_ENABLED", True)), True
        )
        self._hybrid_vector_topk = int(
            cfg.get("hybrid_vector_topk", os.environ.get("MEMOBASE_HYBRID_VECTOR_TOPK", 20))
        )
        self._hybrid_bm25_topk = int(
            cfg.get("hybrid_bm25_topk", os.environ.get("MEMOBASE_HYBRID_BM25_TOPK", 20))
        )
        self._hybrid_keep = int(
            cfg.get("hybrid_keep", os.environ.get("MEMOBASE_HYBRID_KEEP", 10))
        )
        # Summary layer configs (FR-010 / FR-014)
        self._summary_enabled = _as_bool(
            cfg.get("summary_enabled", os.environ.get("MEMOBASE_SUMMARY_ENABLED", False)), False
        )
        self._summary_budget_tokens = _cfg_int(
            cfg.get("summary_budget_tokens", os.environ.get("MEMOBASE_SUMMARY_BUDGET_TOKENS", 350)), 350
        )
        self._event_budget_tokens = _cfg_int(
            cfg.get(
                "event_budget_tokens",
                os.environ.get(
                    "MEMOBASE_EVENT_BUDGET_TOKENS",
                    cfg.get("hybrid_event_budget_tokens", os.environ.get("MEMOBASE_HYBRID_EVENT_BUDGET", 850)),
                ),
            ),
            850,
        )
        self._hybrid_event_budget_tokens = self._event_budget_tokens
        self._summary_topk = _cfg_int(
            cfg.get("summary_topk", os.environ.get("MEMOBASE_SUMMARY_TOPK", 2)), 2
        )
        self._summary_semantic_topk = _cfg_int(
            cfg.get("summary_semantic_topk", os.environ.get("MEMOBASE_SUMMARY_SEMANTIC_TOPK", 3)), 3
        )
        self._summary_single_day_budget = _cfg_int(
            cfg.get("summary_single_day_budget", os.environ.get("MEMOBASE_SUMMARY_SINGLE_DAY_BUDGET", 700)), 700
        )
        # Anchor backfill config (FR-019)
        self._anchor_backfill_enabled = _as_bool(
            cfg.get("anchor_backfill_enabled", os.environ.get("MEMOBASE_ANCHOR_ENABLED", True)), True
        )
        self._anchor_topk = _cfg_int(
            cfg.get("anchor_topk", os.environ.get("MEMOBASE_ANCHOR_TOPK", 2)), 2
        )
        self._anchor_events_per_day = _cfg_int(
            cfg.get("anchor_events_per_day", os.environ.get("MEMOBASE_ANCHOR_EVENTS_PER_DAY", 2)), 2
        )
        self._anchor_budget_tokens = _cfg_int(
            cfg.get("anchor_budget_tokens", os.environ.get("MEMOBASE_ANCHOR_BUDGET_TOKENS", 200)), 200
        )
        self._event_cache_ttl = float(cfg.get("event_cache_ttl", os.environ.get("MEMOBASE_EVENT_CACHE_TTL", 300)))

        # Temporal boost config (P0-A)
        self._temporal_boost_enabled = _as_bool(
            cfg.get("temporal_boost_enabled", os.environ.get("MEMOBASE_TEMPORAL_ENABLED", True)), True
        )
        try:
            self._temporal_gain = float(
                cfg.get("temporal_gain", os.environ.get("MEMOBASE_TEMPORAL_GAIN", 1.6))
            )
        except (TypeError, ValueError):
            self._temporal_gain = 1.6

        try:
            self._temporal_half_life_days = float(
                cfg.get("temporal_half_life_days", os.environ.get("MEMOBASE_TEMPORAL_HALF_LIFE", 30.0))
            )
        except (TypeError, ValueError):
            self._temporal_half_life_days = 30.0

        try:
            self._temporal_floor = float(
                cfg.get("temporal_floor", os.environ.get("MEMOBASE_TEMPORAL_FLOOR", 0.6))
            )
        except (TypeError, ValueError):
            self._temporal_floor = 0.6

        # agy P2-5：空串回退 Asia/Shanghai
        tz_str = (cfg.get("timezone") or os.environ.get("MEMOBASE_TIMEZONE") or "Asia/Shanghai").strip()
        self._timezone = tz_str or "Asia/Shanghai"
        self._tz = get_timezone(self._timezone)

        # Entity boost config (P0-B)
        self._entity_boost_enabled = _as_bool(
            cfg.get("entity_boost_enabled", os.environ.get("MEMOBASE_ENTITY_ENABLED", True)), True
        )

        self._rerank_enabled = _as_bool(
            cfg.get("rerank_enabled", os.environ.get("MEMOBASE_RERANK_ENABLED", True)), True
        )
        self._rerank_mode = str(
            cfg.get("rerank_mode", os.environ.get("MEMOBASE_RERANK_MODE", "auto"))
        ).strip().lower()
        if self._rerank_mode not in ("auto", "always", "never"):
            self._rerank_mode = "auto"
        self._rerank_model = str(
            cfg.get("rerank_model") or os.environ.get("MEMOBASE_RERANK_MODEL") or "Qwen/Qwen3-Reranker-4B"
        )
        # No default endpoint: rerank stays off unless the deployment names
        # BOTH an endpoint and a key. This plugin makes no outbound call to any
        # host the user did not explicitly configure.
        self._rerank_base_url = str(
            cfg.get("rerank_base_url") or os.environ.get("MEMOBASE_RERANK_BASE_URL") or ""
        ).strip()
        self._rerank_api_key = str(
            cfg.get("rerank_api_key") or os.environ.get("MEMOBASE_RERANK_API_KEY") or ""
        ).strip()
        self._rerank_ratio_threshold = float(
            cfg.get("rerank_ratio_threshold", os.environ.get("MEMOBASE_RERANK_RATIO", 1.01))
        )
        # P1-3 防御：None/空串不崩溃，int 解析失败回退默认
        self._rerank_topk = _cfg_int(cfg.get("rerank_topk", os.environ.get("MEMOBASE_RERANK_TOPK", 15)), 15)
        self._rerank_keep = _cfg_int(cfg.get("rerank_keep", os.environ.get("MEMOBASE_RERANK_KEEP", 8)), 8)
        self._rerank_overlap_threshold = float(
            cfg.get("rerank_overlap_threshold", os.environ.get("MEMOBASE_RERANK_OVERLAP", 0.3))
        )
        self._rerank_sim_threshold = float(
            cfg.get("rerank_sim_threshold", os.environ.get("MEMOBASE_RERANK_SIM", 0.3))
        )

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
        if self._hybrid_enabled:
            self._store = EventStore(
                self._client, self._user_id, ttl_secs=self._event_cache_ttl
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
            "`<memory-context>` block. "
            "Recent summaries come from daily memory compression; if you need specific event details, use `memobase_search` with the `date` parameter (YYYY-MM-DD). "
            "You can also search on demand with the `memobase_search` tool, or list the full user profile with "
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

    def _call_context_vector(self, query: str, _with_events: bool = True) -> str:
        """GET /users/context/{user_id} — the vector recall path (base behavior)."""
        if self._client is None or self._user_id is None:
            return ""
        query = query[:_MAX_QUERY_CHARS]

        chats = json.dumps([{"role": "user", "content": query}], ensure_ascii=False)
        params = {
            "max_token_size": str(self._prefetch_max_tokens),
            "chats_str": chats,
            "time_range_in_days": str(self._time_range_days),
        }
        if self._fill_window_events and _with_events:
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

    def _call_context(self, query: str) -> str:
        """Hybrid recall (BM25 + vector RRF) with automatic vector fallback.

        The vector path (context API) is still the source of truth for the
        profile section; the event section is rebuilt from RRF-fused ids.
        Any hybrid error degrades to the plain vector path (fail-open).
        """
        if not self._hybrid_enabled or self._store is None:
            return self._call_context_vector(query)
        try:
            return self._call_context_hybrid(query)
        except Exception as exc:
            logger.warning("[memobase] hybrid recall failed, fallback to vector: %s", exc)
            return self._call_context_vector(query)

    def _fetch_summaries_dual_path(
        self, query: str, time_window: Optional[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """FR-011: 双路检索概要文档（语义路 + 时间直通路），去重合并后返回最多 summary_topk 个。"""
        if self._client is None or self._user_id is None:
            return []

        direct_summaries: List[Dict[str, Any]] = []
        semantic_summaries: List[Dict[str, Any]] = []

        # 1. 直通路 (b): parse_time_window 换算为日期范围
        if time_window is not None:
            try:
                min_age, max_age = time_window
                now_date = datetime.now(self._tz).date()
                end_d = now_date - timedelta(days=min_age)
                start_d = now_date - timedelta(days=max_age)
                if start_d > end_d:
                    start_d, end_d = end_d, start_d
                r_direct = self._client.get(
                    f"/users/summary/{self._user_id}",
                    params={
                        "start_date": start_d.strftime("%Y-%m-%d"),
                        "end_date": end_d.strftime("%Y-%m-%d"),
                        "kind": "daily",
                    },
                )
                r_direct.raise_for_status()
                d_data = r_direct.json()
                if not d_data.get("errno", 0):
                    s_list = (d_data.get("data") or {}).get("summaries") or []
                    # 按日期倒序排（最近日期优先）
                    direct_summaries = sorted(
                        s_list,
                        key=lambda s: str(s.get("summary_date", "")),
                        reverse=True,
                    )
            except Exception as exc:
                logger.warning("[memobase] direct summary retrieval error: %s", exc)
                direct_summaries = []

        # 2. 语义路 (a): GET /users/summary/search/{uid}
        try:
            r_sem = self._client.get(
                f"/users/summary/search/{self._user_id}",
                params={
                    "query": query,
                    "topk": str(self._summary_semantic_topk),
                    "kind": "daily",
                },
            )
            r_sem.raise_for_status()
            sem_data = r_sem.json()
            if not sem_data.get("errno", 0):
                semantic_summaries = (sem_data.get("data") or {}).get("summaries") or []
        except Exception as exc:
            logger.warning("[memobase] semantic summary search error: %s", exc)
            semantic_summaries = []

        # 3. 按 summary_date 去重合并：直通路优先保时间命中，再补语义路命中
        merged: List[Dict[str, Any]] = []
        seen_dates: Set[str] = set()

        for s in direct_summaries:
            s_date = str(s.get("summary_date", "")).strip()
            if s_date and s_date not in seen_dates:
                seen_dates.add(s_date)
                merged.append(s)

        for s in semantic_summaries:
            s_date = str(s.get("summary_date", "")).strip()
            if s_date and s_date not in seen_dates:
                seen_dates.add(s_date)
                merged.append(s)

        return merged[: self._summary_topk]

    def _call_context_hybrid(self, query: str) -> str:
        """Run BM25 + vector legs, fuse with RRF, rebuild the event section.

        Both legs operate on the SAME id space (event id): the vector leg is
        event-level semantic search (`/users/event/search`), the BM25 leg is
        the local event index — so RRF fusion is meaningful and text/dedupe
        stay consistent. Any error propagates to the caller (fail-open there).
        """
        if self._client is None or self._user_id is None:
            return ""
        query = query[:_MAX_QUERY_CHARS]

        # FR-012: 一次性解析时间窗，直通路与事件腿 temporal boost 共用
        time_window = parse_time_window(query)

        # 共享跨段行级去重集合 (FR-015)
        seen_lines: Set[str] = set()
        summary_lines: List[str] = []
        used_summary = 0
        summaries: List[Dict[str, Any]] = []

        # FR-011 / FR-013 / FR-016: 文档段组装（flag 开启时执行，异常 fail-open）
        if self._summary_enabled:
            try:
                summaries = self._fetch_summaries_dual_path(query, time_window)
                # 单日窗预算放宽（Q5-1：直通只命中 1 天时 350 明显不够，30 事件 ≈1500+ 字符）
                single_day = time_window is not None and (time_window[1] - time_window[0]) <= 1
                summary_budget = self._summary_single_day_budget if single_day else self._summary_budget_tokens
                for doc in summaries:
                    d_str = str(doc.get("summary_date", "")).strip()
                    if not d_str:
                        continue
                    try:
                        dt = datetime.strptime(d_str, "%Y-%m-%d").date()
                        m_d = f"{dt.month}/{dt.day}"
                    except Exception:
                        m_d = d_str

                    eids = doc.get("event_ids") or []
                    n_events = len(eids)
                    header_line = f"【{m_d} 概要】（当日 {n_events} 事件，细节可查 memobase_search date={d_str}）"

                    if used_summary + len(header_line) > summary_budget and summary_lines:
                        break
                    summary_lines.append(header_line)
                    used_summary += len(header_line)

                    # 概要 content 行级渲染 (FR-013) + 行级 BM25 排序（Q5-1：#21 截断正解）
                    content = doc.get("content") or ""
                    strip_chars = "-*• \t"
                    raw_lines = []
                    for ln in content.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        disp = f"- {ln.lstrip(strip_chars)}"
                        if disp in seen_lines:
                            continue
                        raw_lines.append(disp)
                    if len(raw_lines) > 1:
                        try:
                            bm = BM25Index()
                            bm.build(raw_lines)
                            scored = bm.score(query)
                            if scored:
                                raw_lines = [raw_lines[i] for i, _ in sorted(scored, key=lambda kv: -kv[1])]
                            # scored 为空（查询词与概要行无 bigram 重叠）保持原序，防内容行全丢
                        except Exception:
                            pass  # BM25 排序失败保持原序（fail-open）
                    for disp in raw_lines:
                        seen_lines.add(disp)
                        if used_summary + len(disp) > summary_budget and summary_lines:
                            break
                        summary_lines.append(disp)
                        used_summary += len(disp)
            except Exception as exc:
                logger.warning("[memobase] summary segment build failed, fail-open: %s", exc)
                summary_lines = []
                used_summary = 0
                summaries = []

        # FR-019: 锚点提取（纯语义查询才启用；有时间词 → 直通路已覆盖，跳过保证幂等）
        anchor_dates: List[str] = []
        if self._anchor_backfill_enabled and time_window is None and summaries:
            anchor_dates = [
                str(s["summary_date"]).strip()
                for s in summaries
                if str(s.get("summary_date", "")).strip()
            ][: self._anchor_topk]
            # P2-3：锚点提取处加 debug 日志，区分「锚点日均已覆盖」和「代码没执行」
            logger.debug(
                "[memobase] anchor extraction: %d dates (anchors=%s, time_window=%s)",
                len(anchor_dates),
                anchor_dates,
                time_window,
            )

        # FR-014 / FR-016: 事件段预算与让渡逻辑
        if self._summary_enabled:
            if summary_lines:
                event_budget = self._event_budget_tokens
            else:
                # 概要段无命中或异常，预算让渡给事件段
                event_budget = self._event_budget_tokens + self._summary_budget_tokens
        else:
            event_budget = self._event_budget_tokens

        # --- vector leg: event-level semantic search (event ids) ---
        vec_hits: List[Tuple[str, str]] = []
        vec_sims: List[Optional[float]] = []
        r = self._client.get(
            f"/users/event/search/{self._user_id}",
            params={
                "query": query,
                "topk": str(self._hybrid_vector_topk),
                "similarity_threshold": "0.05",
                "time_range_in_days": "3650",
            },
        )
        r.raise_for_status()
        data = r.json()
        errno = data.get("errno", 0)
        if errno:
            raise RuntimeError(f"Memobase event search error: {data.get('errmsg', errno)}")
        for ev in (data.get("data") or {}).get("events") or []:
            tip = (ev.get("event_data") or {}).get("event_tip")
            if tip:
                vec_hits.append((ev["id"], tip))
                sim = ev.get("similarity")
                vec_sims.append(sim if isinstance(sim, (int, float)) else None)
        vec_ids = [eid for eid, _ in vec_hits]

        # --- BM25 leg: local index over the (cached) full event list ---
        self._store.refresh()
        bm25_ids = self._store.bm25_top(query, topk=self._hybrid_bm25_topk)

        # --- Entity leg: local inverted index over entities (P0-B) ---
        entity_ids: List[str] = []
        if self._entity_boost_enabled and self._store is not None:
            try:
                entity_ids = self._store.entity_leg(query, topk=self._hybrid_bm25_topk)
                if entity_ids:
                    logger.debug("[memobase] entity leg hits: %d docs", len(entity_ids))
            except Exception as exc:
                logger.warning("[memobase] entity leg error, fallback to empty: %s", exc)
                entity_ids = []

        # --- RRF fusion (all legs share the event id space) ---
        fused_scored = rrf_fusion_scored([vec_ids, bm25_ids, entity_ids])
        raw_fused_scored = fused_scored

        # --- Temporal boost (P0-A, FR-012: 共用 time_window) ---
        if self._temporal_boost_enabled and self._store is not None and time_window is not None:
            try:
                now_date = datetime.now(self._tz).date()
                boosted: List[Tuple[str, float]] = []
                for eid, s in fused_scored:
                    t = self._store.time_for(eid)
                    if t is not None:
                        try:
                            ev_date = datetime.fromtimestamp(t, tz=self._tz).date()
                            age_days = float((now_date - ev_date).days)
                            f = temporal_factor(
                                age_days,
                                time_window,
                                gain=self._temporal_gain,
                                half_life_days=self._temporal_half_life_days,
                                floor=self._temporal_floor,
                            )
                        except Exception:
                            f = 1.0
                    else:
                        f = 1.0
                    boosted.append((eid, s * f))
                boosted.sort(key=lambda kv: kv[1], reverse=True)
                fused_scored = boosted
                logger.info(
                    "[memobase] temporal boost applied (window=%s, gain=%.2f, candidates=%d)",
                    time_window, self._temporal_gain, len(boosted),
                )
            except Exception as exc:
                logger.warning("[memobase] temporal boost error, skip: %s", exc)

        fused = [eid for eid, _ in fused_scored]
        text_map = {eid: txt for eid, txt in vec_hits}

        # --- rerank gating: only pay when the coarse ranking is suspect ---
        use_rerank = False
        if self._rerank_enabled and self._rerank_api_key and self._rerank_base_url:
            if self._rerank_mode == "always":
                use_rerank = True
            elif self._rerank_mode == "auto":
                # P1: short query check — CJK 按字符数/2 折算（连续中文块算多个词），
                # 防纯中文长句被误判为 1 个 term 而旁路门控（agy P0-1）
                latin_count = len(_LATIN_RE.findall(query))
                cjk_terms = sum(max(1, len(chunk) // 2) for chunk in _CJK_RE.findall(query))
                effective_terms = latin_count + cjk_terms
                if effective_terms < 3:
                    use_rerank = True
                    rr_metrics = {"short_query": float(effective_terms)}
                    logger.info("[memobase] rerank auto-triggered (short_query=%d)", effective_terms)
                else:
                    use_rerank, rr_metrics = low_discrimination(
                        vec_ids,
                        bm25_ids,
                        raw_fused_scored,
                        vec_sims,
                        ratio_threshold=self._rerank_ratio_threshold,
                        overlap_threshold=self._rerank_overlap_threshold,
                        sim_threshold=self._rerank_sim_threshold,
                    )
                    if use_rerank:
                        logger.info("[memobase] rerank auto-triggered (metrics=%s)", rr_metrics)
                    else:
                        logger.debug("[memobase] rerank skipped (metrics=%s)", rr_metrics)
        if use_rerank:
            cand_ids = fused[: self._rerank_topk]
            cand_pairs = []
            for eid in cand_ids:
                txt = text_map.get(eid) or self._store.text_for(eid)
                if txt and txt.strip():
                    cand_pairs.append((eid, txt.strip()[:500]))
            if cand_pairs:
                try:
                    ranked = rerank(
                        self._rerank_base_url,
                        self._rerank_api_key,
                        self._rerank_model,
                        query,
                        [t for _, t in cand_pairs],
                        top_k=self._rerank_keep,
                    )
                    reranked_ids = [
                        cand_pairs[i][0] for i, _ in ranked if 0 <= i < len(cand_pairs)
                    ]
                    reranked_set = set(reranked_ids)
                    fused = reranked_ids + [eid for eid in fused if eid not in reranked_set]
                    logger.info(
                        "[memobase] rerank applied (%s, %d docs) → %d kept",
                        self._rerank_model, len(cand_pairs), len(reranked_ids),
                    )
                except Exception as exc:
                    logger.warning("[memobase] rerank failed, keep RRF order: %s", exc)

        rows: List[Tuple[str, str]] = []
        seen: set = set()
        for eid in fused[: self._hybrid_keep * 2]:
            txt = text_map.get(eid) or self._store.text_for(eid)
            if not txt:
                continue
            key = txt.strip()
            if key in seen:
                continue
            seen.add(key)
            rows.append((eid, txt))
            if len(rows) >= self._hybrid_keep:
                break
        if not rows:
            # P1-5：事件腿全空时锚点仍要兜底（纯语义查询 + 概要命中时恰是锚点最该介入的场景）
            if anchor_dates and self._summary_enabled and self._anchor_backfill_enabled and self._store is not None:
                lines, used, seen_lines = self._anchor_backfill(
                    anchor_dates, [], fused, [], 0, seen_lines, event_budget
                )
                if lines:
                    return self._finalize_hybrid(query, lines, summary_lines if self._summary_enabled else None)
            if summary_lines:
                return self._finalize_hybrid(query, [], summary_lines)
            return self._call_context_vector(query)

        # --- rebuild event section under the token budget (line-level dedupe) ---
        lines: List[str] = []
        used = 0
        for _eid, txt in rows:
            budget_exceeded = False
            for ln in txt.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                strip_chars = "-*• \t"
                disp = f"- {ln.lstrip(strip_chars)}"
                if disp in seen_lines:
                    continue
                seen_lines.add(disp)
                if used + len(disp) > event_budget and lines:
                    budget_exceeded = True
                    break
                lines.append(disp)
                used += len(disp)
            if budget_exceeded:
                break

        # FR-019: 锚点保底注入（纯语义查询且存在缺失锚点日事件时追加）
        if anchor_dates and self._summary_enabled and self._anchor_backfill_enabled and self._store is not None:
            lines, used, seen_lines = self._anchor_backfill(
                anchor_dates, rows, fused, lines, used, seen_lines, event_budget
            )

        return self._finalize_hybrid(query, lines, summary_lines if self._summary_enabled else None)

    def _anchor_backfill(
        self,
        anchor_dates: List[str],
        rows: List[Tuple[str, str]],
        fused: List[str],
        lines: List[str],
        used: int,
        seen_lines: Set[str],
        event_budget: int,
    ) -> Tuple[List[str], int, Set[str]]:
        """FR-019 锚点保底注入：锚点日事件缺场时用剩余预算补 top2/日。

        纯追加不改排名；任何异常回滚到调用前状态（fail-open）。
        P1-5：抽成独立方法，rows 为空（事件腿全空）时也能兜底。
        """
        orig_lines = list(lines)
        orig_used = used
        orig_seen_lines = set(seen_lines)
        try:
            injected_dates = {
                self._store.date_for(eid, tz=self._tz)
                for eid, _ in rows
            }
            injected_dates.discard(None)
            missing = [d for d in anchor_dates if d not in injected_dates]
            if missing:
                cand = []
                seen_cand = {eid for eid, _ in rows}
                rank_map = {eid: i for i, eid in enumerate(fused)}
                for d in missing:
                    day_evs = self._store.events_for_date(d, tz=self._tz)
                    day_evs.sort(
                        key=lambda e: (0, rank_map[e[0]]) if e[0] in rank_map else (1, -(self._store.time_for(e[0]) or 0.0))
                    )
                    added_for_day = 0
                    for eid, _ in day_evs:
                        if eid in seen_cand:
                            continue
                        seen_cand.add(eid)
                        cand.append(eid)
                        added_for_day += 1
                        if added_for_day >= self._anchor_events_per_day:
                            break
                # FR-019 fix（2026-09-03）：锚点预算独立，不再被事件腿挤占
                # 原逻辑 budget_left = min(event_budget - used, anchor_budget) 在事件腿
                # 吃满预算（used≈850）时只剩个位数 token，锚点全被挡掉（79% 0 appended）。
                # 锚点是「时间敏感查询的缺失日保底」，不该跟日常检索抢同一池子；
                # 独立预算最多 +200 字符/次，换取缺失日事件真正可补。
                budget_left = self._anchor_budget_tokens
                strip_chars = "-*• \t"
                n_appended = 0
                for eid in cand:
                    txt = self._store.text_for(eid)
                    if not txt:
                        continue
                    event_added = False
                    for ln in txt.splitlines():
                        ln = ln.strip()
                        if not ln:
                            continue
                        disp = f"- {ln.lstrip(strip_chars)}"
                        if disp in seen_lines:
                            continue
                        if budget_left - len(disp) < 0:
                            break
                        seen_lines.add(disp)
                        lines.append(disp)
                        budget_left -= len(disp)
                        used += len(disp)
                        event_added = True
                    if event_added:
                        n_appended += 1
                    if budget_left <= 0:
                        break
                logger.info(
                    "[memobase] anchor backfill: %d days missing, %d events appended (anchors=%s)",
                    len(missing),
                    n_appended,
                    anchor_dates,
                )
        except Exception as exc:
            logger.warning("[memobase] anchor backfill failed, fail-open: %s", exc)
            lines = orig_lines
            used = orig_used
            seen_lines = orig_seen_lines
        return lines, used, seen_lines

    _EVENT_HEADER_RE = re.compile(r"## (?:过去事件|Past Events|Events)[：:]?\s*")

    def _finalize_hybrid(
        self,
        query: str,
        lines: List[str],
        summary_lines: Optional[List[str]] = None,
    ) -> str:
        """Attach the mixed event section to the vector context's profile part."""
        logger.info(
            "[memobase] hybrid recall OK: %d summary lines, %d event lines (query=%s...)",
            len(summary_lines or []),
            len(lines),
            query[:30],
        )
        base = self._call_context_vector(query, _with_events=False)
        head = self._EVENT_HEADER_RE.split(base, maxsplit=1)[0]
        summary_part = ""
        if summary_lines:
            summary_part = "## 近期概要：\n" + "\n".join(summary_lines) + "\n\n"
        event_part = ""
        if lines:
            event_part = "## 过去事件：\n" + "\n".join(lines)
        new_ctx = f"{head}{summary_part}{event_part}"
        self._record_success()
        return new_ctx.strip()

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
        return RecallStatus(
            provider_label=self._recall_label, count=0, glyph=self._recall_glyph
        )

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
                        "date": {
                            "type": "string",
                            "description": "Optional date in YYYY-MM-DD format to retrieve all raw event details for that day.",
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
            date_arg = (args or {}).get("date")
            max_len = int((args or {}).get("max_length", 2000))
            if self._breaker_open():
                return json.dumps({"error": "Memobase temporarily unavailable"}, ensure_ascii=False)

            # FR-017: 带 date 参数时从 EventStore 本地按日过滤
            if date_arg:
                try:
                    target_date = datetime.strptime(str(date_arg).strip(), "%Y-%m-%d").date()
                    if self._store is None and self._client is not None and self._user_id is not None:
                        self._store = EventStore(self._client, self._user_id, ttl_secs=self._event_cache_ttl)

                    if self._store is not None:
                        self._store.refresh()
                        events_for_date = self._store.events_for_date(target_date, tz=self._tz)
                    else:
                        events_for_date = []

                    if not events_for_date:
                        return json.dumps(
                            {"note": f"No events found for date {date_arg}."},
                            ensure_ascii=False,
                        )

                    lines = []
                    strip_chars = "-*• \t"
                    for _, txt in events_for_date:
                        for ln in txt.splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            disp = f"- {ln.lstrip(strip_chars)}"
                            lines.append(disp)

                    ctx = f"## {date_arg} 事件细节（共 {len(events_for_date)} 条事件）：\n" + "\n".join(lines)
                    if max_len and len(ctx) > max_len:
                        ctx = ctx[:max_len]
                    return json.dumps({"context": ctx}, ensure_ascii=False)
                except Exception as exc:
                    logger.warning("[memobase] memobase_search with date failed: %s", exc)
                    return json.dumps({"error": str(exc)}, ensure_ascii=False)

            # 不带 date 参数时行为与 v3 逐字节一致
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