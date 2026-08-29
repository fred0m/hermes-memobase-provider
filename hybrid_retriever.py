# -*- coding: utf-8 -*-
"""Hybrid retriever for the memobase memory provider — BM25 + vector RRF fusion.

Pure stdlib (no jieba / rank_bm25 / numpy): Chinese text is tokenized as
CJK unigram+bigram (per contiguous chunk) + latin/number tokens, scored with
standard Okapi BM25 (k1=1.5, b=0.75), then fused with the vector hit list via
Reciprocal Rank Fusion (RRF, k=60).

Design goals:
* zero dependency — runs inside the Hermes venv as-is
* fail-open friendly — every method returns plain data and raises on
  transport errors; the caller decides what to do
* thread-safe cache — prefetch runs on a background thread but the manual
  memobase_search tool may hit the store at the same time; the slow HTTP
  fetch happens OUTSIDE the lock, only the final assignment is atomic.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

# Latin/number word: allow inner dots/hyphens/underscores but never trailing.
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
# Contiguous CJK runs (ideographs only — no CJK punctuation).
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")


def tokenize(text: str) -> List[str]:
    """CJK unigram+bigram (per contiguous chunk) + latin words. Lowercased."""
    if not text:
        return []
    tokens: List[str] = []
    t = text.lower()
    for m in _LATIN_RE.finditer(t):
        w = m.group(0).rstrip("._-")
        if len(w) >= 2:
            tokens.append(w)
    for chunk in _CJK_RE.findall(t):
        n = len(chunk)
        if n == 1:
            tokens.append(chunk)
            continue
        # unigrams + bigrams
        tokens.extend(chunk)
        for i in range(n - 1):
            tokens.append(chunk[i : i + 2])
    return tokens


# --------------------------------------------------------------------------- #
# BM25 index (Okapi)
# --------------------------------------------------------------------------- #

class BM25Index:
    """In-memory Okapi BM25 over a fixed document corpus.

    Documents are pre-tokenized at build time; scoring is O(q terms) with a
    term->postings map. Suited for a few hundred short event strings.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self) -> None:
        self._docs: List[List[str]] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        self._doc_len: List[int] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 0.0
        self._built = False

    def build(self, docs: Sequence[str]) -> None:
        self._docs = [tokenize(d) for d in docs]
        self._postings = {}
        self._doc_len = []
        df: Dict[str, int] = {}
        for di, d in enumerate(self._docs):
            tf: Dict[str, int] = {}
            for term in d:
                tf[term] = tf.get(term, 0) + 1
            self._doc_len.append(len(d))
            for term, c in tf.items():
                if term not in self._postings:
                    self._postings[term] = []
                self._postings[term].append((di, c))
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._built = True

    def reset(self) -> None:
        self._built = False
        self._docs = []
        self._postings = {}
        self._doc_len = []
        self._df = {}
        self._avgdl = 0.0

    @property
    def is_built(self) -> bool:
        return self._built

    def score(self, query: str) -> List[Tuple[int, float]]:
        """Return [(doc_idx, bm25_score)] for documents with any overlap."""
        if not self._built or not self._postings:
            return []
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        n = len(self._docs)
        scores: Dict[int, float] = {}
        for term in q_terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            df = self._df[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for di, tf in posting:
                dl = self._doc_len[di]
                denom = (
                    tf + self.K1 * (1.0 - self.B + self.B * dl / self._avgdl)
                    if self._avgdl
                    else 1.0
                )
                scores[di] = scores.get(di, 0.0) + idf * (tf * (self.K1 + 1.0)) / denom
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# RRF fusion
# --------------------------------------------------------------------------- #

def rrf_fusion(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    """Fuse ranked lists of ids via Reciprocal Rank Fusion.

    Each list is in best->worst order. Returns ids sorted by fused score.
    """
    fused = _rrf_scores(ranked_lists, k)
    return [eid for eid, _ in fused]


def rrf_fusion_scored(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> List[Tuple[str, float]]:
    """Like rrf_fusion but also returns each id's fused score (best first)."""
    return _rrf_scores(ranked_lists, k)


def _rrf_scores(ranked_lists: Sequence[Sequence[str]], k: int) -> List[Tuple[str, float]]:
    fused: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, eid in enumerate(ranking, start=1):
            fused[eid] = fused.get(eid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# Rerank gating (only spend the reranker when the coarse ranking is unreliable)
# --------------------------------------------------------------------------- #

def low_discrimination(
    vec_ids: Sequence[str],
    bm25_ids: Sequence[str],
    fused_scored: Sequence[Tuple[str, float]],
    vector_sims: Sequence[Optional[float]],
    *,
    ratio_threshold: float = 1.01,
    overlap_threshold: float = 0.3,
    sim_threshold: float = 0.3,
) -> Tuple[bool, Dict[str, float]]:
    """Heuristic: should we pay for a reranker on top of RRF?

    Returns (decision, metrics) so the caller can log what fired.
    A. RRF top1/top2 score RATIO is < 1.01 (nearly tied) — absolute gap is
       meaningless in RRF(k=60) scale (even unanimous consensus leaves only
       ~0.0005), so we compare the ratio.
    B. Vector and BM25 legs barely agree (low overlap) → fusion is suspect.
    C. The best vector hit has a weak similarity → semantic recall is weak.
    """
    metrics: Dict[str, float] = {}
    if len(fused_scored) >= 2:
        top1 = fused_scored[0][1]
        top2 = fused_scored[1][1]
        metrics["rrf_ratio"] = top1 / top2 if top2 > 0 else 1.0
        if top2 > 0 and metrics["rrf_ratio"] < ratio_threshold:
            return True, metrics
    s1 = set(vec_ids)
    s2 = set(bm25_ids)
    if s1 and s2:
        denom = min(len(s1), len(s2))
        if denom > 0:
            metrics["overlap"] = len(s1 & s2) / denom
            if metrics["overlap"] < overlap_threshold:
                return True, metrics
    else:
        metrics["overlap"] = 0.0
        return True, metrics
    if vector_sims:
        top_sim = vector_sims[0]
        if top_sim is not None:
            metrics["top_sim"] = top_sim
            if top_sim < sim_threshold:
                return True, metrics
    return False, metrics


def rerank(
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    documents: Sequence[str],
    top_k: Optional[int] = None,
    timeout: float = 4.0,
) -> List[Tuple[int, float]]:
    """Call a reranking API (OpenAI-compatible /rerank, e.g. SiliconFlow).

    Returns [(doc_idx, relevance_score)] sorted by score, best first.
    Raises on transport/HTTP errors so the caller can fail open.
    """
    if not documents or not query.strip():
        return []
    body: Dict[str, object] = {"model": model, "query": query, "documents": list(documents)}
    if top_k:
        body["top_n"] = top_k
    req = urllib.request.Request(
        base_url.rstrip("/") + "/rerank",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-memobase/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = data.get("results") or []
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return [(int(item["index"]), float(item["relevance_score"])) for item in results]


# --------------------------------------------------------------------------- #
# Event store (thin cache over the memobase events API)
# --------------------------------------------------------------------------- #

class EventStore:
    """Caches the user's full event list (id -> summary text) for BM25.

    Data source: GET /users/event/{user_id}?topk=<max>&need_summary=false
    Each event is flattened to its `event_tip` plus `profile_delta` contents.
    Identifiers are EVENT ids (same space as /users/event/search).
    """

    def __init__(
        self,
        client,
        user_id: str,
        ttl_secs: float = 300.0,
        max_events: int = 1000,
    ) -> None:
        self._client = client
        self._user_id = user_id
        self._ttl = ttl_secs
        self._max_events = max_events
        self._lock = threading.Lock()
        self._texts: Dict[str, str] = {}
        self._index = BM25Index()
        self._fetched_at: float = 0.0
        self._last_error_at: float = 0.0
        self._err_backoff = 30.0

    def _do_fetch(self) -> Tuple[Dict[str, str], BM25Index]:
        """Run the full HTTP fetch + index build WITHOUT holding the lock."""
        r = self._client.get(
            f"/users/event/{self._user_id}",
            params={"topk": str(self._max_events), "need_summary": "false"},
        )
        r.raise_for_status()
        data = r.json()
        errno = data.get("errno", 0)
        if errno:
            raise RuntimeError(f"Memobase events error: {data.get('errmsg', errno)}")
        events = (data.get("data") or {}).get("events") or []
        texts: Dict[str, str] = {}
        for ev in events:
            ed = ev.get("event_data") or {}
            parts: List[str] = []
            tip = ed.get("event_tip")
            if tip:
                parts.append(tip)
            for pd in ed.get("profile_delta") or []:
                c = pd.get("content")
                if c:
                    parts.append(c)
            if parts:
                texts[ev["id"]] = "\n".join(parts)
        index = BM25Index()
        index.build(list(texts.values()))
        return texts, index

    def refresh(self, force: bool = False) -> None:
        """Refresh the cache if stale/forced. Slow I/O happens outside the lock;
        a failed fetch backs off 30s instead of retrying every turn."""
        with self._lock:
            fresh = not force and (time.time() - self._fetched_at) <= self._ttl
            if fresh:
                return
            if time.time() - self._last_error_at < self._err_backoff:
                return
        try:
            texts, index = self._do_fetch()
        except Exception:
            with self._lock:
                self._last_error_at = time.time()
            raise
        with self._lock:
            self._texts = texts
            self._index = index
            self._fetched_at = time.time()

    @property
    def ready(self) -> bool:
        with self._lock:
            return bool(self._texts)

    def text_for(self, eid: str) -> Optional[str]:
        with self._lock:
            return self._texts.get(eid)

    def bm25_top(self, query: str, topk: int = 20) -> List[str]:
        with self._lock:
            if not self._index.is_built:
                return []
            hits = self._index.score(query)[:topk]
            ids = list(self._texts.keys())
            return [ids[di] for di, _ in hits]