# -*- coding: utf-8 -*-
"""Tests for the Memobase memory provider plugin.

Runs standalone: no Hermes Agent import required (a minimal stub for
``agent.memory_provider`` is installed only when the real package is
unavailable). Zero extra dependencies — stdlib ``unittest`` + ``httpx``
(httpx ships with Hermes but is also a normal PyPI dependency of this test).

Run from the repo root:

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import time
import types
import unittest
import uuid
from unittest import mock

import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# --- Agent dependency stub (used only when hermes-agent is not importable) ----
try:  # pragma: no cover - environment dependent
    from agent.memory_provider import MemoryProvider, RecallStatus  # noqa: F401
    _HAS_AGENT = True
except ImportError:  # pragma: no cover
    _HAS_AGENT = False
    _mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal behavioral stub
        def name(self):
            return "?"

        def is_available(self):
            return False

        def initialize(self, *args, **kwargs):
            pass

        def get_tool_schemas(self):
            return []

    class RecallStatus:
        def __init__(self, provider_label="", count=0, glyph="🧠"):
            self.provider_label = provider_label
            self.count = count
            self.glyph = glyph

    _mp.MemoryProvider = MemoryProvider
    _mp.RecallStatus = RecallStatus
    _agent = types.ModuleType("agent")
    _agent.memory_provider = _mp
    sys.modules["agent"] = _agent
    sys.modules["agent.memory_provider"] = _mp

# Load the plugin from the repo-root __init__.py (same layout as a dropped-in
# $HERMES_HOME/plugins/<name>/ directory).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "memobase_plugin", os.path.join(REPO_ROOT, "__init__.py")
)
_plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plugin)
MemobaseMemoryProvider = _plugin.MemobaseMemoryProvider

BASE_URL = "http://memobase.test:8019/api/v1"
USER_NAME = "user"
EXPECTED_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, USER_NAME + "memobase_client"))


def _make_provider(handler=None, base_url=BASE_URL, **init_kwargs):
    """Provider with a mocked HTTP transport; skips initialize() internals."""
    transport = httpx.MockTransport(handler or (lambda r: httpx.Response(404)))
    client = httpx.Client(transport=transport, base_url=base_url)
    p = MemobaseMemoryProvider()
    p._client = client
    p._base_url = base_url
    p._user_id = EXPECTED_USER_ID
    p._api_key = "test-token"
    if "writes_enabled" in init_kwargs:
        p._writes_enabled = init_kwargs.pop("writes_enabled")
    return p, transport


class TestNameAndAvailability(unittest.TestCase):
    def test_name_is_property_string(self):
        p = MemobaseMemoryProvider()
        self.assertEqual(p.name, "memobase")  # property access, not method

    def test_is_available_requires_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMOBASE_BASE_URL", None)
            self.assertFalse(MemobaseMemoryProvider().is_available())

        with mock.patch.dict(os.environ, {"MEMOBASE_BASE_URL": BASE_URL}, clear=False):
            self.assertTrue(MemobaseMemoryProvider().is_available())

    def test_unavailable_reason_hints_env(self):
        self.assertIn("MEMOBASE_BASE_URL", MemobaseMemoryProvider().unavailable_reason())

    def test_system_prompt_block_names_tools(self):
        block = MemobaseMemoryProvider().system_prompt_block()
        self.assertIn("memobase_search", block)
        self.assertIn("memobase_profile", block)


class TestInitialize(unittest.TestCase):
    def test_initialize_derives_user_id_from_name(self):
        with mock.patch.dict(
            os.environ,
            {"MEMOBASE_BASE_URL": BASE_URL, "MEMOBASE_USER_NAME": USER_NAME},
            clear=False,
        ):
            with mock.patch("httpx.Client") as client_cls:
                p = MemobaseMemoryProvider()
                p.initialize("sess", hermes_home="/tmp", agent_context="primary")
        self.assertEqual(p._user_id, EXPECTED_USER_ID)
        client_cls.assert_called_once()
        kwargs = client_cls.call_args.kwargs
        self.assertEqual(kwargs["base_url"], BASE_URL)

    def test_initialize_auth_header_only_when_key_present(self):
        with mock.patch.dict(
            os.environ, {"MEMOBASE_BASE_URL": BASE_URL}, clear=False,
        ):
            os.environ.pop("MEMOBASE_API_KEY", None)
            with mock.patch("httpx.Client") as client_cls:
                p = MemobaseMemoryProvider()
                p.initialize("sess", hermes_home="/tmp", agent_context="primary")
        headers = client_cls.call_args.kwargs.get("headers", {})
        self.assertEqual(headers, {})  # no key → no auth header

        with mock.patch.dict(
            os.environ,
            {"MEMOBASE_BASE_URL": BASE_URL, "MEMOBASE_API_KEY": "k"},
            clear=False,
        ):
            with mock.patch("httpx.Client") as client_cls:
                p = MemobaseMemoryProvider()
                p.initialize("sess", hermes_home="/tmp", agent_context="primary")
        self.assertEqual(
            client_cls.call_args.kwargs.get("headers", {}).get("Authorization"),
            "Bearer k",
        )

    def test_initialize_disables_writes_for_non_primary(self):
        with mock.patch.dict(
            os.environ,
            {"MEMOBASE_BASE_URL": BASE_URL, "MEMOBASE_API_KEY": "k"},
            clear=False,
        ):
            with mock.patch("httpx.Client"):
                p = MemobaseMemoryProvider()
                p.initialize("sess", hermes_home="/tmp", agent_context="cron")
        self.assertFalse(p._writes_enabled)


class TestPrefetch(unittest.TestCase):
    CTX = "---\n# 记忆\n## 用户当前状态：\n- name: momo\n## 过去事件：\n- bought coffee yesterday"

    def _ctx_handler(self, request):
        self.assertIn("/users/context/", request.url.path)
        self.assertIn("chats_str=", request.url.query.decode())
        self.assertIn("fill_window_with_events=true", request.url.query.decode())
        return httpx.Response(200, json={"errno": 0, "data": {"context": self.CTX}})

    def test_prefetch_returns_context(self):
        p, _ = _make_provider(self._ctx_handler)
        out = p.prefetch("what does momo like", session_id="s")
        self.assertEqual(out, self.CTX)
        self.assertEqual(p.recall_status().provider_label, "memobase")

    def test_prefetch_fail_open_on_http_error(self):
        def boom(request):
            raise httpx.ConnectError("down")
        p, _ = _make_provider(boom)
        self.assertEqual(p.prefetch("q", session_id="s"), "")

    def test_prefetch_fail_open_on_404(self):
        p, _ = _make_provider(lambda r: httpx.Response(404))
        self.assertEqual(p.prefetch("q", session_id="s"), "")

    def test_prefetch_treats_business_error_as_failure(self):
        def err(request):
            return httpx.Response(200, json={"errno": 1001, "errmsg": "user not found"})
        p, _ = _make_provider(err)
        self.assertEqual(p.prefetch("q", session_id="s"), "")

    def test_queue_prefetch_caches_for_next_turn(self):
        p, _ = _make_provider(self._ctx_handler)
        p.queue_prefetch("q1", session_id="s")
        time.sleep(0.5)
        self.assertEqual(p.prefetch("q1", session_id="s"), self.CTX)
        # different query → cache miss, still resolves
        self.assertEqual(p.prefetch("q2", session_id="s"), self.CTX)

    def test_query_truncation(self):
        seen = {}

        def handler(request):
            import urllib.parse as up
            qs = dict(up.parse_qsl(request.url.query.decode()))
            seen["chats"] = qs.get("chats_str", "")
            return httpx.Response(200, json={"errno": 0, "data": {"context": "x"}})

        p, _ = _make_provider(handler)
        p._call_context("长" * 5000)
        self.assertLessEqual(len(seen["chats"]), 1600)  # 1500 chars + JSON overhead


class TestSyncTurn(unittest.TestCase):
    def test_sync_turn_posts_blob(self):
        captured = {}

        def handler(request):
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"errno": 0, "data": {"id": "1"}})

        p, _ = _make_provider(handler)
        p.sync_turn("user said x", "assistant said y", session_id="s")
        time.sleep(0.8)
        self.assertEqual(captured.get("method"), "POST")
        self.assertIn("/blobs/insert/", captured.get("path", ""))
        self.assertEqual(captured["body"]["blob_type"], "chat")
        messages = captured["body"]["blob_data"]["messages"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["content"], "assistant said y")

    def test_sync_turn_skipped_when_writes_disabled(self):
        sent = []

        def handler(request):
            sent.append(request.url.path)
            return httpx.Response(200, json={"errno": 0, "data": {}})

        p, _ = _make_provider(handler, writes_enabled=False)
        p.sync_turn("u", "a", session_id="s")
        time.sleep(0.4)
        self.assertEqual(sent, [])


class TestTools(unittest.TestCase):
    def test_schema_names_reserved_free(self):
        p = MemobaseMemoryProvider()
        names = [s["name"] for s in p.get_tool_schemas()]
        self.assertEqual(names, ["memobase_search", "memobase_profile"])

    def test_search_returns_json_context(self):
        def handler(request):
            return httpx.Response(
                200, json={"errno": 0, "data": {"context": "profile data"}}
            )
        p, _ = _make_provider(handler)
        out = json.loads(p.handle_tool_call("memobase_search", {"query": "q"}))
        self.assertEqual(out["context"], "profile data")

    def test_search_error_returns_json_error(self):
        p, _ = _make_provider(lambda r: httpx.Response(500))
        out = json.loads(p.handle_tool_call("memobase_search", {"query": "q"}))
        self.assertIn("error", out)

    def test_profile_returns_json(self):
        def handler(request):
            return httpx.Response(
                200, json={"errno": 0, "data": {"profiles": [{"content": "x"}]}}
            )
        p, _ = _make_provider(handler)
        out = json.loads(p.handle_tool_call("memobase_profile", {}))
        self.assertIn("data", out)

    def test_unknown_tool_raises(self):
        p = MemobaseMemoryProvider()
        with self.assertRaises(NotImplementedError):
            p.handle_tool_call("nope", {})


class TestCircuitBreaker(unittest.TestCase):
    def test_opens_after_5_failures_and_resets_on_success(self):
        p, _ = _make_provider(lambda r: httpx.Response(500))
        for _ in range(5):
            p._record_failure()
        self.assertTrue(p._breaker_open())
        # Even a prefetch is now skipped while open
        self.assertEqual(p.prefetch("q", session_id="s"), "")
        # Reset breaker window so success can recover
        p._breaker_open_until = 0.0
        self.assertFalse(p._breaker_open())
        p._record_success()
        self.assertEqual(p._fail_count, 0)

    def test_success_keeps_breaker_closed(self):
        p, _ = _make_provider(lambda r: httpx.Response(200, json={"errno": 0, "data": {"context": "c"}}))
        p._record_success()
        p._record_success()
        self.assertFalse(p._breaker_open())


class TestShutdown(unittest.TestCase):
    def test_shutdown_closes_client(self):
        p, _ = _make_provider(lambda r: httpx.Response(404))
        with mock.patch.object(p._client, "close") as close_mock:
            p.shutdown()
        close_mock.assert_called_once()
        self.assertIsNone(p._client)


if __name__ == "__main__":
    unittest.main()