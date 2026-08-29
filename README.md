# Memobase Memory Provider for Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory provider
plugin that connects the `MemoryProvider` lifecycle to a **self-hosted
[Memobase](https://github.com/memobase-ai/memobase) server**
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
  "time_range_days": 180
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

## Development

Run the test suite (stdlib `unittest` + `httpx.MockTransport`, no extra deps):

```bash
python -m unittest discover -s tests -v
```

## License

MIT