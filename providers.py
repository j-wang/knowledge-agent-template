#!/usr/bin/env python3
"""
Pluggable embedding + reranking providers.

Every provider exposes a tiny, uniform interface and is *fail-soft*: any
network error, timeout, missing API key, bad response, or missing optional
dependency results in ONE concise stderr warning and a `None` return, so the
caller (search.py / build_embeddings.py) can degrade gracefully. Providers
never raise out to the caller.

Embedders implement:
    .available() -> bool
    .embed(texts: list[str]) -> np.ndarray | None      # (N, D) float32, L2-normalized

Rerankers implement:
    .available() -> bool
    .rerank(query: str, docs: list[str]) -> list[float] | None   # one score per doc

HTTP is done with stdlib urllib (no hard `requests` dependency) so the
zero-config install stays light. `sentence-transformers` and the OpenAI SDK
are NOT imported here — `openai` is reached purely over its HTTP API.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from retrieval_config import EmbedConfig, RerankConfig

# Track which warnings we've already emitted so a 762-query run doesn't print
# the same "endpoint unreachable" line hundreds of times.
_warned: set[str] = set()

# Wake commands already fired this process (keyed by the command string) so a
# batch run sends the magic packet once, not per request.
_woken: set[str] = set()


def _warn(key: str, msg: str) -> None:
    """Emit a one-line stderr warning at most once per unique key."""
    if key in _warned:
        return
    _warned.add(key)
    print(f"[retrieval] {msg}", file=sys.stderr)


def _endpoint_up(probe_url: str, timeout: float) -> bool:
    """Return True if probe_url answers at all. A GET that returns ANY HTTP
    status (even 4xx/5xx) means the server is up; only a transport-level
    failure (refused / no route / DNS / timeout) means it's still down."""
    try:
        urllib.request.urlopen(
            urllib.request.Request(probe_url, method="GET"), timeout=timeout
        )
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _wake_and_wait(wake_cmd: str, probe_url: str, wake_timeout: float,
                   req_timeout: float) -> bool:
    """Run the (operator-configured) wake command once, then poll probe_url
    until it answers or wake_timeout elapses. Returns True if it came up.

    The wake command is the operator's own shell string from the agent's .env
    (e.g. `wakeonlan AA:BB:..` or `ssh gpubox 'systemctl start tei'`); it is run
    via the shell on the machine running search — there is no untrusted input.
    Best-effort: a failing wake command is not fatal, we still poll."""
    if wake_cmd not in _woken:
        _woken.add(wake_cmd)
        _warn(f"wake:{wake_cmd}",
              f"endpoint unreachable — running wake command and waiting up to "
              f"{int(wake_timeout)}s for it to come up.")
        try:
            subprocess.run(wake_cmd, shell=True, timeout=min(30.0, wake_timeout),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass  # wake is best-effort; we still poll below
    deadline = time.monotonic() + wake_timeout
    while time.monotonic() < deadline:
        if _endpoint_up(probe_url, req_timeout):
            return True
        time.sleep(3.0)
    return False


def _normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product equals cosine similarity."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _do_post(url: str, payload: dict, timeout: float,
             headers: dict | None = None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: dict, timeout: float,
                    headers: dict | None = None, *,
                    wake_cmd: str | None = None, wake_timeout: float = 0.0):
    """POST JSON, return parsed JSON. Raises on transport/HTTP/parse errors;
    callers wrap this and degrade.

    If the call fails at the transport level (connection refused / no route /
    DNS / timeout — i.e. the host is down) AND a wake_cmd is configured, run it,
    wait for the endpoint to come up, then retry once. An HTTPError (the server
    answered with a status) is NOT a transport failure, so it never triggers a
    wake — it propagates immediately."""
    try:
        return _do_post(url, payload, timeout, headers)
    except urllib.error.HTTPError:
        raise  # server is up; this is a real HTTP-level error, don't wake
    except Exception:
        if wake_cmd and _wake_and_wait(wake_cmd, url, wake_timeout, timeout):
            return _do_post(url, payload, timeout, headers)
        raise


# TEI enforces a per-request input cap (`max_client_batch_size`, default 32) and
# rejects larger requests with HTTP 413. Both the embed and rerank endpoints
# share that cap, so we discover it once from `{base}/info` and sub-batch under
# it. Used by TEIEmbedder and TEIReranker.
TEI_DEFAULT_MAX_BATCH = 32


def _tei_max_client_batch(info_url: str | None, timeout: float) -> int:
    """Read max_client_batch_size from a TEI `/info`; default on any failure."""
    if not info_url:
        return TEI_DEFAULT_MAX_BATCH
    try:
        req = urllib.request.Request(info_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            info = json.loads(resp.read().decode())
        n = int(info.get("max_client_batch_size") or 0)
        return n if n > 0 else TEI_DEFAULT_MAX_BATCH
    except Exception:
        return TEI_DEFAULT_MAX_BATCH


# ===========================================================================
# Embedders
# ===========================================================================

class NullEmbedder:
    """Default: no dense retrieval."""

    def available(self) -> bool:
        return False

    def embed(self, texts: list[str]):
        return None


class OpenAIEmbedder:
    """OpenAI (or any OpenAI-compatible server) /v1/embeddings via plain HTTP.

    Endpoint defaults to https://api.openai.com. Point KB_EMBED_ENDPOINT at a
    self-hosted OpenAI-compatible server to use it instead.
    """

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        base = (cfg.endpoint or "https://api.openai.com").rstrip("/")
        # Allow either a bare host or a host already including /v1.
        if base.endswith("/v1"):
            self.url = base + "/embeddings"
        else:
            self.url = base + "/v1/embeddings"
        self.model = cfg.model or "text-embedding-3-small"

    def available(self) -> bool:
        # An API key is required for api.openai.com; a self-hosted compatible
        # server may not need one, so only require a key for the default host.
        if "api.openai.com" in self.url and not self.cfg.api_key:
            _warn("openai_nokey",
                  "embed provider=openai but no API key (KB_EMBED_API_KEY / "
                  "OPENAI_KEY / OPENAI_API_KEY) — dense retrieval disabled.")
            return False
        return True

    def embed(self, texts: list[str]):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        try:
            body = _http_post_json(
                self.url, {"model": self.model, "input": texts},
                self.cfg.timeout, headers,
                wake_cmd=self.cfg.wake_cmd, wake_timeout=self.cfg.wake_timeout,
            )
            data = sorted(body["data"], key=lambda x: x["index"])
            vecs = [d["embedding"] for d in data]
            return _normalize(np.array(vecs, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 — fail soft
            _warn("openai_fail",
                  f"embed provider=openai failed ({type(exc).__name__}: {exc}) "
                  f"at {self.url} — falling back to BM25.")
            return None


class TEIEmbedder:
    """HuggingFace text-embeddings-inference native API: POST {endpoint}/embed
    with {"inputs": [...]} -> [[...vec...], ...].

    TEI enforces a per-request input cap (`max_client_batch_size`, default 32);
    a request over it is rejected with HTTP 413, which would otherwise fail-soft
    the entire dense build to BM25. We discover that cap from `{endpoint}/info`
    (once, cached) and sub-batch transparently, so callers may pass an
    arbitrarily long `texts` list regardless of the server's limit."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        base = (cfg.endpoint or "").rstrip("/") if cfg.endpoint else None
        self.url = base + "/embed" if base else None
        self.info_url = base + "/info" if base else None
        self._max_batch: int | None = None

    def available(self) -> bool:
        if not self.url:
            _warn("tei_embed_noendpoint",
                  "embed provider=tei but KB_EMBED_ENDPOINT is not set — "
                  "dense retrieval disabled.")
            return False
        return True

    def _max_client_batch(self) -> int:
        """Server's max_client_batch_size from /info (cached; default on miss)."""
        if self._max_batch is None:
            self._max_batch = _tei_max_client_batch(self.info_url, self.cfg.timeout)
        return self._max_batch

    def embed(self, texts: list[str]):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        limit = self._max_client_batch()
        try:
            rows: list = []
            for i in range(0, len(texts), limit):
                body = _http_post_json(
                    self.url, {"inputs": texts[i:i + limit]}, self.cfg.timeout,
                    headers, wake_cmd=self.cfg.wake_cmd,
                    wake_timeout=self.cfg.wake_timeout,
                )
                # TEI returns a bare list of vectors (one per input).
                rows.extend(body)
            return _normalize(np.array(rows, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 — fail soft
            _warn("tei_embed_fail",
                  f"embed provider=tei failed ({type(exc).__name__}: {exc}) "
                  f"at {self.url} — falling back to BM25.")
            return None


class SentenceTransformersEmbedder:
    """Local sentence-transformers model. Optional dependency: if the package
    is not installed (or the model can't load), we warn once and report
    unavailable rather than crashing."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.model_id = cfg.model
        self._model = None
        self._tried = False

    def _ensure_model(self):
        if self._tried:
            return self._model
        self._tried = True
        if not self.model_id:
            _warn("st_nomodel",
                  "embed provider=sentence-transformers but KB_EMBED_MODEL is "
                  "not set — dense retrieval disabled.")
            return None
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
            self._model = SentenceTransformer(self.model_id)
        except Exception as exc:  # noqa: BLE001 — missing dep or load failure
            _warn("st_load_fail",
                  f"embed provider=sentence-transformers unavailable "
                  f"({type(exc).__name__}: {exc}) — install with "
                  f"`pip install sentence-transformers`; falling back to BM25.")
            self._model = None
        return self._model

    def available(self) -> bool:
        return self._ensure_model() is not None

    def embed(self, texts: list[str]):
        model = self._ensure_model()
        if model is None:
            return None
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            emb = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True,
            )
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — fail soft
            _warn("st_encode_fail",
                  f"embed provider=sentence-transformers encode failed "
                  f"({type(exc).__name__}: {exc}) — falling back to BM25.")
            return None


def get_embedder(cfg: EmbedConfig):
    """Factory: return an embedder for the configured provider."""
    provider = cfg.provider
    if provider == "openai":
        return OpenAIEmbedder(cfg)
    if provider == "tei":
        return TEIEmbedder(cfg)
    if provider == "sentence-transformers":
        return SentenceTransformersEmbedder(cfg)
    return NullEmbedder()


# ===========================================================================
# Rerankers
# ===========================================================================

class NullReranker:
    def available(self) -> bool:
        return False

    def rerank(self, query: str, docs: list[str]):
        return None


class TEIReranker:
    """TEI rerank API: POST {endpoint}/rerank with {"query","texts":[...]}
    -> [{"index": i, "score": s}, ...] (order not guaranteed)."""

    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        base = (cfg.endpoint or "").rstrip("/") if cfg.endpoint else None
        self.url = base + "/rerank" if base else None
        self.info_url = base + "/info" if base else None
        self._max_batch: int | None = None

    def available(self) -> bool:
        if not self.url:
            _warn("tei_rerank_noendpoint",
                  "rerank provider=tei but KB_RERANK_ENDPOINT is not set — "
                  "rerank disabled.")
            return False
        return True

    def _max_client_batch(self) -> int:
        """Server's max_client_batch_size from /info (cached; default on miss)."""
        if self._max_batch is None:
            self._max_batch = _tei_max_client_batch(self.info_url, self.cfg.timeout)
        return self._max_batch

    def rerank(self, query: str, docs: list[str]):
        if not docs:
            return []
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        limit = self._max_client_batch()
        try:
            # Cross-encoder scores are per-(query,doc) independent, so chunking
            # under the server cap and merging by index is exact. KB_RERANK_TOP_N
            # defaults to 50 > the TEI default of 32, so this path is load-bearing.
            scores = [0.0] * len(docs)
            for start in range(0, len(docs), limit):
                body = _http_post_json(
                    self.url, {"query": query, "texts": docs[start:start + limit]},
                    self.cfg.timeout, headers,
                    wake_cmd=self.cfg.wake_cmd, wake_timeout=self.cfg.wake_timeout,
                )
                # TEI returns chunk-local indices; offset back to the global list.
                for item in body:
                    idx = start + int(item["index"])
                    if 0 <= idx < len(docs):
                        scores[idx] = float(item["score"])
            return scores
        except Exception as exc:  # noqa: BLE001 — fail soft
            _warn("tei_rerank_fail",
                  f"rerank provider=tei failed ({type(exc).__name__}: {exc}) "
                  f"at {self.url} — keeping fused order.")
            return None


class CohereReranker:
    """Cohere /v1/rerank: POST {endpoint or https://api.cohere.com}/v1/rerank
    with {"model","query","documents":[...]}
    -> {"results": [{"index": i, "relevance_score": s}, ...]}."""

    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        base = (cfg.endpoint or "https://api.cohere.com").rstrip("/")
        if base.endswith("/v1"):
            self.url = base + "/rerank"
        else:
            self.url = base + "/v1/rerank"
        self.model = cfg.model or "rerank-english-v3.0"

    def available(self) -> bool:
        if not self.cfg.api_key:
            _warn("cohere_nokey",
                  "rerank provider=cohere but KB_RERANK_API_KEY is not set — "
                  "rerank disabled.")
            return False
        return True

    def rerank(self, query: str, docs: list[str]):
        if not docs:
            return []
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        try:
            body = _http_post_json(
                self.url,
                {"model": self.model, "query": query, "documents": docs},
                self.cfg.timeout, headers,
                wake_cmd=self.cfg.wake_cmd, wake_timeout=self.cfg.wake_timeout,
            )
            scores = [0.0] * len(docs)
            for item in body.get("results", []):
                idx = int(item["index"])
                if 0 <= idx < len(docs):
                    scores[idx] = float(item["relevance_score"])
            return scores
        except Exception as exc:  # noqa: BLE001 — fail soft
            _warn("cohere_fail",
                  f"rerank provider=cohere failed ({type(exc).__name__}: {exc}) "
                  f"at {self.url} — keeping fused order.")
            return None


class HTTPReranker:
    """Generic HTTP reranker. Contract:
        POST {endpoint}  body={"query": str, "documents": [str, ...]}
        response: either a bare list of floats (one score per document, in
        order) OR {"scores": [...]}.
    Use this for a custom server that doesn't speak TEI or Cohere."""

    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self.url = cfg.endpoint.rstrip("/") if cfg.endpoint else None

    def available(self) -> bool:
        if not self.url:
            _warn("http_rerank_noendpoint",
                  "rerank provider=http but KB_RERANK_ENDPOINT is not set — "
                  "rerank disabled.")
            return False
        return True

    def rerank(self, query: str, docs: list[str]):
        if not docs:
            return []
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        try:
            body = _http_post_json(
                self.url, {"query": query, "documents": docs},
                self.cfg.timeout, headers,
                wake_cmd=self.cfg.wake_cmd, wake_timeout=self.cfg.wake_timeout,
            )
            if isinstance(body, dict):
                body = body.get("scores", [])
            scores = [float(s) for s in body]
            if len(scores) != len(docs):
                raise ValueError(
                    f"expected {len(docs)} scores, got {len(scores)}")
            return scores
        except Exception as exc:  # noqa: BLE001 — fail soft
            _warn("http_rerank_fail",
                  f"rerank provider=http failed ({type(exc).__name__}: {exc}) "
                  f"at {self.url} — keeping fused order.")
            return None


def get_reranker(cfg: RerankConfig):
    """Factory: return a reranker for the configured provider."""
    provider = cfg.provider
    if provider == "tei":
        return TEIReranker(cfg)
    if provider == "cohere":
        return CohereReranker(cfg)
    if provider == "http":
        return HTTPReranker(cfg)
    return NullReranker()
