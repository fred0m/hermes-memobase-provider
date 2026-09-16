# Memobase Memory Provider for Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory provider
plugin that connects the `MemoryProvider` lifecycle to a **self-hosted
[Memobase](https://github.com/memodb-io/memobase) server**
(FastAPI + PostgreSQL + pgvector). Memobase stores a user profile
(ontology-based topics) plus an append-only event timeline, extracted offline
by an LLM when the buffer flushes.

This plugin gives the agent **automatic per-turn recall** instead of
model-discretion search: every turn, relevant memories are prefetched and
injected into context, and the completed turn is written back for background
profile extraction.

## How it works

When the provider is active, Hermes automatically:

1. **`prefetch(query)`** — recalls the user profile + recent event timeline for
   the current user message and injects it into context (Memobase's
   `/users/context` assembly; hot path < 100 ms).
2. **`sync_turn(user, asst)`** — inserts the completed turn into Memobase's
   buffer for background LLM profile extraction (auto-flush at the buffer
   threshold).
3. Exposes **`memobase_search`** / **`memobase_profile`** tools as an
   on-demand backstop.

## Hybrid retrieval (v3)

Recall is not a single vector lookup. The context endpoint is queried with a
fixed budget, and the plugin runs its own **three-leg hybrid retrieval**
locally over the returned event window:

| Leg | Signal | Purpose |
|-----|--------|---------|
| `bm25` | Okapi BM25 over CJK-bigram + latin tokens | exact term / keyword match |
| `vec` | Memobase's own embeddings | semantic match |
| `entity` | inverted index over extracted entities | proper nouns, file paths, IPs, domains, IDs, quoted strings, book titles |

The three ranked lists are fused with **Reciprocal Rank Fusion (RRF)**, then
refined by two optional passes:

- **Temporal boost** — when the query contains a time expression
  (`今天` / `昨天` / `上周` / `3天前` / …), events inside the parsed window are
  multiplied by `temporal_gain`; events outside decay exponentially with
  `temporal_half_life_days`, floored at `temporal_floor`.
- **Conditional rerank (opt-in)** — a cross-encoder rerank pass fires only
  when the fused score distribution is *low-discrimination* (i.e. the ranking
  is too flat to trust), so a clear-cut ranking never pays the latency cost.
  **It runs only when you configure both `rerank_base_url` and
  `rerank_api_key`.** There is no default endpoint, so an unconfigured install
  makes no outbound request to any host but your own Memobase server.

Tunable through `$HERMES_HOME/memobase.json` — see the module docstring in
`__init__.py` for the full list (`hybrid_enabled`, `hybrid_keep`,
`temporal_*`, `entity_boost_enabled`, `rerank_*`, `timezone`, …).

### CJK proper nouns

The entity leg recognizes structured entities (paths, IPs, domains, camel
case, quoted text) out of the box. Personal names are deployment-specific, so
the public build ships **no** name list. Add your own:

```json
{
  "cjk_known_names": ["Alice", "Bob"]
}
```

Empty (the default) is fine — everything else still works.

## Requirements

- A running Memobase server (>= 0.0.42) with a bearer token.
- Hermes Agent with the memory-provider plugin system (any recent build).

## Install

Drop this repository's files into `~/.hermes/plugins/memobase/`
(user-level memory provider discovery path), or clone it there:

```bash
mkdir -p ~/.hermes/plugins/memobase
git clone https://github.com/fred0m/hermes-memobase-provider.git ~/.hermes/plugins/memobase
```

Then configure:

```bash
# ~/.hermes/.env
MEMOBASE_BASE_URL=http://<your-memobase-host>:8019/api/v1
MEMOBASE_API_KEY=<your-memobase-token>
# optional: MEMOBASE_USER_NAME=user   (maps to uuid5(NAMESPACE_DNS, name+"memobase_client"))
```

Enable the provider:

```bash
hermes config set memory.provider memobase
hermes gateway restart
hermes memory status   # → Provider: memobase / Plugin: installed / Status: available
```

Behavioral settings may live in `~/.hermes/memobase.json`:

```json
{
  "prefetch_max_tokens": 2000,
  "prefetch_wait_secs": 5,
  "fill_window_events": true,
  "time_range_days": 180,
  "hybrid_enabled": true,
  "hybrid_keep": 10,
  "temporal_boost_enabled": true,
  "temporal_gain": 1.6,
  "temporal_half_life_days": 30,
  "temporal_floor": 0.6,
  "timezone": "Asia/Shanghai",
  "cjk_known_names": [],
  "rerank_base_url": "",
  "rerank_api_key": ""
}
```

## API surface used

| Lifecycle | Endpoint |
|-----------|----------|
| prefetch / memobase_search | `GET /users/context/{user_id}?max_token_size=…&chats_str=…` |
| sync_turn (write) | `POST /blobs/insert/{user_id}?wait_process=false` |
| on_session_end (flush) | `POST /users/buffer/{user_id}/chat?wait_process=true` |
| memobase_profile | `GET /users/profile/{user_id}` |

## Safety & reliability

- **Non-primary agents skip writes** (subagent / cron / flush contexts cannot
  corrupt the user profile); recall still works.
- **Fail-open**: any backend error is logged and swallowed — the agent turn
  never breaks because memory died.
- **Circuit breaker**: after 5 consecutive failures the provider pauses for
  60 s instead of hammering a down server; a success resets the counter.
- **Fast hot path**: `prefetch()` consumes a background thread's cached result
  (wait ≤ `prefetch_wait_secs`); a slow backend degrades to "no injection" and
  the search tool remains available.
- **Business errors are caught**: Memobase may return HTTP 200 with a
  non-zero `errno` body; those are treated as failures, not successes.
- **One outbound host by default**: your Memobase server. Vector recall and
  the entity leg are computed locally from the context payload. The reranker
  is the only other path, and it stays off until you set both an endpoint and
  a key for it.

## Development

Run the test suite (stdlib `unittest` + `httpx.MockTransport`, no extra deps):

```bash
python -m unittest discover -s tests -v
```

## License

MIT
