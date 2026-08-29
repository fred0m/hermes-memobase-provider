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
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore

# --------------------------------------------------------------------------- #
# Timezone & Temporal helpers
# --------------------------------------------------------------------------- #

def get_timezone(tz_name: str) -> datetime.tzinfo:
    """Resolve a timezone name (e.g. 'Asia/Shanghai') to a tzinfo instance."""
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    if tz_name in ("Asia/Shanghai", "Asia/Chongqing", "PRC", "CST"):
        try:
            from datetime import timedelta
            return timezone(timedelta(hours=8))
        except Exception:
            pass
    return timezone.utc


def parse_created_at(s: Optional[str]) -> Optional[float]:
    """Parse ISO UTC string to epoch timestamp (float seconds). Fail-open."""
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


_CN_NUMS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_num(s: str) -> Optional[int]:
    if s.isdigit():
        return int(s)
    if s in _CN_NUMS:
        return _CN_NUMS[s]
    if len(s) == 2 and s.startswith("十") and s[1] in _CN_NUMS:
        return 10 + _CN_NUMS[s[1]]
    if len(s) == 2 and s.endswith("十") and s[0] in _CN_NUMS:
        return _CN_NUMS[s[0]] * 10
    if len(s) == 3 and s[1] == "十" and s[0] in _CN_NUMS and s[2] in _CN_NUMS:
        return _CN_NUMS[s[0]] * 10 + _CN_NUMS[s[2]]
    return None


_STATIC_TEMPORAL_PATTERNS: List[Tuple[re.Pattern, int, int]] = [
    # 离散日期差体系（今天=0，昨天=1）→ 单日窗收窄为精确闭区间（agy P1-3）
    (re.compile(r"今天|今晚|今早|今日"), 0, 0),
    (re.compile(r"昨天|昨晚"), 1, 1),
    (re.compile(r"前天"), 2, 2),
    (re.compile(r"最近|近期|这几天|这阵子"), 0, 6),
    (re.compile(r"这周|本周|这星期"), 0, 6),
    (re.compile(r"上周|上星期"), 7, 13),
    (re.compile(r"上个月|上月"), 30, 59),
    (re.compile(r"今年|今年以来"), 0, 365),
]

_DYNAMIC_TEMPORAL_PATTERNS: List[Tuple[re.Pattern, Callable[[int], Tuple[int, int]]]] = [
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*天前"), lambda n: (n, n)),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*(?:个)?(?:星期|周)前"), lambda n: (n * 7, (n + 1) * 7 - 1)),
    (re.compile(r"(\d+|[一二两三四五六七八九十]+)\s*(?:个)?月前"), lambda n: (n * 30, (n + 1) * 30 - 1)),
    # agy P2-6：口语"这N天/这N周/这N月"与"过去N天"合并
    (re.compile(r"(?:过去|这)\s*(\d+|[一二两三四五六七八九十]+)\s*天"), lambda n: (0, max(0, n - 1))),
    (re.compile(r"(?:过去|这)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:个)?(?:星期|周)"), lambda n: (0, max(0, n * 7 - 1))),
    (re.compile(r"(?:过去|这)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:个)?月"), lambda n: (0, max(0, n * 30 - 1))),
]


def parse_time_window(query: str) -> Optional[Tuple[int, int]]:
    """Extract temporal window (min_age_days, max_age_days) from query.

    Returns (min_age, max_age) left-closed right-closed integer days.
    Multiple matches pick the closest / most specific window.
    No match returns None.
    """
    if not query:
        return None
    candidates: List[Tuple[int, int]] = []
    for pat, min_d, max_d in _STATIC_TEMPORAL_PATTERNS:
        if pat.search(query):
            candidates.append((min_d, max_d))

    for pat, fn in _DYNAMIC_TEMPORAL_PATTERNS:
        for m in pat.finditer(query):
            raw_num = m.group(1)
            n = _parse_num(raw_num)
            if n is not None and n > 0:
                candidates.append(fn(n))

    if not candidates:
        return None
    # Pick the most specific window (smallest span, then smallest max_age)
    return min(candidates, key=lambda w: (w[1] - w[0], w[1]))


def temporal_factor(
    age_days: float,
    window: Tuple[int, int],
    *,
    gain: float = 1.6,
    half_life_days: float = 30.0,
    floor: float = 0.6,
) -> float:
    """Calculate temporal multiplier for an event.

    In-window events receive `gain` (e.g. 1.6x).
    Out-of-window events decay exponentially with overshoot distance down to `floor` (e.g. 0.6x).
    """
    if window[0] <= age_days <= window[1]:
        return gain
    overshoot = age_days - window[1] if age_days > window[1] else window[0] - age_days
    decay = math.exp(-overshoot / half_life_days)
    return max(floor, decay)


# --------------------------------------------------------------------------- #
# Entity Extraction
# --------------------------------------------------------------------------- #

_COMMON_UPPER_STOPWORDS = {
    "THE", "AND", "FOR", "NOT", "BUT", "ALL", "ANY", "ARE", "HAS", "HAD",
    "HER", "HIS", "HIM", "ITS", "NEW", "OUR", "OUT", "SEE", "TWO", "WHO",
    "CAN", "GET", "SET", "USE", "HOW", "WHY", "NOW", "YES", "NON", "OFF",
    "ONE", "TOP", "APP", "API", "URL", "URI", "HTTP", "POST", "JSON",
}

_CJK_KNOWN = [
    "图图", "沫沫", "小爱", "爱音", "克拉拉", "岁岁", "潜潜", "纪纪", "灯灯",
]

# 拉丁专名表：严格要求 \b 单词边界（agy P0-2：omp 子串匹配误报 prompt/company）
_LATIN_KNOWN_RES = [
    re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
    for name in [
        "agy", "omp", "dockercenter", "chromebook", "hermes", "memobase",
        "9router", "memobase-use", "qwen3-reranker-4b",
    ]
]

_ENTITY_RES = [
    (re.compile(r"\b[\w.-]+\.(?:py|md|yaml|yml|json|ts|tsx|go|rs|toml|sh|sql)\b", re.IGNORECASE), "FILE"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP"),
    (re.compile(r"\b[\w-]+\.(?:com|cn|dev|io|ai|org|net)\b", re.IGNORECASE), "DOMAIN"),
    (re.compile(r"\b(?=.*[a-zA-Z])[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)+\b"), "IDENT"),  # 至少含一个字母，防纯数字/金额（agy P1-4）
    (re.compile(r"\b[A-Za-z]+(?:[A-Z][a-z]+)+\b"), "CAMEL"),  # 允许缩写开头（BM25Index/RRFScored，agy P2-7）
    (re.compile(r"[\u201c\"]([^\u201d\"\n]{2,24})[\u201d\"]"), "QUOTED"),
    (re.compile(r"\u300a([^\u300b\n]{2,24})\u300b"), "BOOK"),
]

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")


def extract_entities(
    text: str, *, df_check: Optional[Callable[[str], int]] = None
) -> List[str]:
    """Extract structured and domain entities from text.

    df_check(term) -> int: optional callback to verify acronym df >= 2 in corpus.
    Returns unique entity list (lowercased).
    """
    if not text:
        return []
    entities: List[str] = []
    seen: Set[str] = set()

    def _add(ent: str) -> None:
        val = ent.strip().lower()
        if 2 <= len(val) <= 24 and val not in seen:
            seen.add(val)
            entities.append(val)

    text_lower = text.lower()
    for name in _CJK_KNOWN:
        if name.lower() in text_lower:
            _add(name)

    for pat in _LATIN_KNOWN_RES:
        for m in pat.finditer(text):
            _add(m.group(0))

    for pat, typ in _ENTITY_RES:
        if typ in ("QUOTED", "BOOK"):
            for m in pat.finditer(text):
                _add(m.group(1))
        else:
            for m in pat.finditer(text):
                _add(m.group(0))

    for m in _ACRONYM_RE.finditer(text):
        val = m.group(0)
        if val in _COMMON_UPPER_STOPWORDS:
            continue
        val_lower = val.lower()
        if df_check is not None and df_check(val_lower) < 2:
            continue
        _add(val)

    return entities


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

# Latin/number word: allow inner dots/hyphens/underscores but never trailing.
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
# Contiguous CJK runs (ideographs only — no CJK punctuation).
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
# 中文单字停用词（虚词/代词/助词）——文档侧 unigram 过滤，bigram 保留
_CJK_STOPWORDS = frozenset(
    "的了是在和有我你他她它这那就都而及与着或个们把被让对从向为以于只条点么吗"
)
# 通用英文高频词——排除 latin boost（agy P1-1：app/api/json 不该抢中文核心词权重）
_LATIN_STOPWORDS = frozenset(
    w.lower() for w in _COMMON_UPPER_STOPWORDS
) | {"app", "api", "web", "url", "uri", "http", "json", "post", "get", "the", "and", "for", "with"}


def tokenize(text: str, query_mode: bool = False) -> List[str]:
    """CJK unigram+bigram (per contiguous chunk) + latin words. Lowercased.

    query_mode=True: CJK chunks of len>=2 emit **bigrams only** — unigrams
    are high-df noise for short queries (生日/模型 match almost every doc);
    single-char chunks fall back to unigram. Document side keeps unigram+bigram
    so recall is preserved, but single-char stopwords (的/是/了) are dropped
    from the index to cut noise.
    """
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
            if chunk not in _CJK_STOPWORDS:
                tokens.append(chunk)
            continue
        if query_mode:
            # bigrams only — unigrams are high-df noise for queries
            for i in range(n - 1):
                tokens.append(chunk[i : i + 2])
        else:
            for ch in chunk:
                if ch not in _CJK_STOPWORDS:
                    tokens.append(ch)
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
        """Return [(doc_idx, bm25_score)] for documents with any overlap.

        Query side uses query_mode tokenization (CJK bigrams only) to cut
        high-df unigram noise; latin/number tokens get an idf boost since
        exact identifiers (9router, memobase-use, GOALS.md) are the strongest
        keyword signal.
        """
        if not self._built or not self._postings:
            return []
        q_terms = set(tokenize(query, query_mode=True))
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
            # latin/number exact tokens are the strongest keyword signal
            # (exclude generic english words — agy P1-1)
            if _LATIN_RE.fullmatch(term) and term not in _LATIN_STOPWORDS:
                idf *= 1.5
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
    """Caches the user's full event list (id -> summary text, timestamp, entity index).

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
        self._times: Dict[str, float] = {}
        self._entity_index: Dict[str, List[str]] = {}
        self._index = BM25Index()
        self._fetched_at: float = 0.0
        self._last_error_at: float = 0.0
        self._err_backoff = 30.0

    def _do_fetch(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, List[str]], BM25Index]:
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
        times: Dict[str, float] = {}
        for ev in events:
            eid = ev.get("id")
            if not eid:
                continue
            t = parse_created_at(ev.get("created_at"))
            if t is not None:
                times[eid] = t
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
                texts[eid] = "\n".join(parts)
        index = BM25Index()
        index.build(list(texts.values()))

        entity_index: Dict[str, List[str]] = {}
        for eid, txt in texts.items():
            for ent in extract_entities(txt, df_check=lambda term: index._df.get(term, 0)):
                eids = entity_index.setdefault(ent, [])
                if eid not in eids:
                    eids.append(eid)

        return texts, times, entity_index, index

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
            texts, times, entity_index, index = self._do_fetch()
        except Exception:
            with self._lock:
                self._last_error_at = time.time()
            raise
        with self._lock:
            self._texts = texts
            self._times = times
            self._entity_index = entity_index
            self._index = index
            self._fetched_at = time.time()

    @property
    def ready(self) -> bool:
        with self._lock:
            return bool(self._texts)

    def text_for(self, eid: str) -> Optional[str]:
        with self._lock:
            return self._texts.get(eid)

    def time_for(self, eid: str) -> Optional[float]:
        with self._lock:
            return self._times.get(eid)

    def bm25_top(self, query: str, topk: int = 20) -> List[str]:
        with self._lock:
            if not self._index.is_built:
                return []
            hits = self._index.score(query)[:topk]
            ids = list(self._texts.keys())
            return [ids[di] for di, _ in hits]

    def entity_leg(self, query: str, topk: int = 20) -> List[str]:
        """Entity leg: returns eid list sorted by inverse entity document frequency score.

        Score for matched eid = sum_{matched ent} 1 / df(ent).
        Returns [] if no entities match.
        """
        with self._lock:
            if not self._entity_index:
                return []
            ents = extract_entities(query, df_check=lambda term: self._index._df.get(term, 0))
            if not ents:
                return []
            scores: Dict[str, float] = {}
            for ent in ents:
                matched_eids = self._entity_index.get(ent)
                if not matched_eids:
                    continue
                df = len(matched_eids)
                if df <= 0:
                    continue
                weight = 1.0 / df
                for eid in matched_eids:
                    scores[eid] = scores.get(eid, 0.0) + weight
            if not scores:
                return []
            sorted_hits = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            return [eid for eid, _ in sorted_hits[:topk]]