#!/usr/bin/env python3
"""
Retrieval configuration — pluggable, optional, env-driven.

This module centralizes all configuration for the retrieval pipeline so that
search.py and build_embeddings.py stay provider-agnostic. EVERYTHING here is
optional with safe defaults: with zero configuration the knowledge base runs
local BM25 only, no network, no extra dependencies.

Config is read from environment variables. A local `.env` file (next to this
module) is also loaded if present — env vars already set in the process take
precedence over `.env`.

----------------------------------------------------------------------------
Environment variables (all optional)
----------------------------------------------------------------------------
Embeddings (query->doc dense retrieval + doc-doc similarity):
  KB_EMBED_PROVIDER   none (default) | openai | tei | sentence-transformers
  KB_EMBED_MODEL      provider-specific model id (no default that points anywhere)
  KB_EMBED_ENDPOINT   base URL for an openai-compatible or TEI HTTP server
  KB_EMBED_API_KEY    api key; falls back to OPENAI_KEY / OPENAI_API_KEY
  KB_EMBED_TIMEOUT    per-request timeout in seconds (default 10)
  KB_EMBED_WAKE_CMD   optional shell command run if the embed endpoint is
                      unreachable (e.g. a self-hosted box that sleeps); after
                      running it we poll the endpoint until it answers, then
                      retry. Unset = no wake (just degrade to BM25).
  KB_EMBED_WAKE_TIMEOUT  seconds to wait for the endpoint after waking (default 120)

Reranking (precision pass over fused candidates):
  KB_RERANK_PROVIDER  none (default) | tei | cohere | http
  KB_RERANK_MODEL     provider-specific model id
  KB_RERANK_ENDPOINT  base URL for the rerank server
  KB_RERANK_API_KEY   api key (if the provider needs one)
  KB_RERANK_TIMEOUT   per-request timeout in seconds (default 10)
  KB_RERANK_TOP_N     how many fused candidates to rerank (default 50)
  KB_RERANK_WAKE_CMD     optional shell command run if the rerank endpoint is
                         unreachable (same semantics as KB_EMBED_WAKE_CMD)
  KB_RERANK_WAKE_TIMEOUT seconds to wait for the endpoint after waking (default 120)

Fusion / mode:
  KB_RETRIEVAL_MODE   auto (default) | bm25 | dense | hybrid
                      auto = hybrid (RRF of BM25 + dense) when doc embeddings
                      are present AND an embedder is available, else bm25-only.

See .env.example for documented, copy-paste-ready example blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ENV_PATH = SCRIPT_DIR / ".env"

# Valid choices (kept here so other modules can validate / message consistently)
EMBED_PROVIDERS = ("none", "openai", "tei", "sentence-transformers")
RERANK_PROVIDERS = ("none", "tei", "cohere", "http")
RETRIEVAL_MODES = ("auto", "bm25", "dense", "hybrid")

DEFAULT_TIMEOUT = 10.0
DEFAULT_RERANK_TOP_N = 50
DEFAULT_WAKE_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# .env loading (no third-party dep; precedence: real env > .env)
# ---------------------------------------------------------------------------

_dotenv_loaded = False


def load_dotenv(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file into os.environ (without overriding
    existing process env vars). Returns the dict that was parsed.

    Tolerant of comments (#...), blank lines, surrounding quotes, and an
    optional leading `export `. Never raises — a malformed/absent file just
    yields {}.
    """
    global _dotenv_loaded
    parsed: dict[str, str] = {}
    if not path.exists():
        _dotenv_loaded = True
        return parsed
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            parsed[key] = val
            # Don't clobber an already-exported env var.
            os.environ.setdefault(key, val)
    except Exception:
        # A broken .env should never break retrieval.
        pass
    _dotenv_loaded = True
    return parsed


def _get(name: str, default: str | None = None) -> str | None:
    if not _dotenv_loaded:
        load_dotenv()
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _resolve_embed_api_key() -> str | None:
    """KB_EMBED_API_KEY, falling back to OPENAI_KEY / OPENAI_API_KEY."""
    for name in ("KB_EMBED_API_KEY", "OPENAI_KEY", "OPENAI_API_KEY"):
        val = _get(name)
        if val:
            return val
    return None


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbedConfig:
    provider: str
    model: str | None
    endpoint: str | None
    api_key: str | None
    timeout: float
    wake_cmd: str | None = None
    wake_timeout: float = DEFAULT_WAKE_TIMEOUT

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class RerankConfig:
    provider: str
    model: str | None
    endpoint: str | None
    api_key: str | None
    timeout: float
    top_n: int
    wake_cmd: str | None = None
    wake_timeout: float = DEFAULT_WAKE_TIMEOUT

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    embed: EmbedConfig
    rerank: RerankConfig


def _norm_choice(value: str | None, choices: tuple[str, ...], default: str) -> str:
    if value is None:
        return default
    v = value.strip().lower()
    return v if v in choices else default


def load_config() -> RetrievalConfig:
    """Build the full retrieval config from env / .env. Always succeeds."""
    load_dotenv()

    embed = EmbedConfig(
        provider=_norm_choice(_get("KB_EMBED_PROVIDER"), EMBED_PROVIDERS, "none"),
        model=_get("KB_EMBED_MODEL"),
        endpoint=_get("KB_EMBED_ENDPOINT"),
        api_key=_resolve_embed_api_key(),
        timeout=_get_float("KB_EMBED_TIMEOUT", DEFAULT_TIMEOUT),
        wake_cmd=_get("KB_EMBED_WAKE_CMD"),
        wake_timeout=_get_float("KB_EMBED_WAKE_TIMEOUT", DEFAULT_WAKE_TIMEOUT),
    )
    rerank = RerankConfig(
        provider=_norm_choice(_get("KB_RERANK_PROVIDER"), RERANK_PROVIDERS, "none"),
        model=_get("KB_RERANK_MODEL"),
        endpoint=_get("KB_RERANK_ENDPOINT"),
        api_key=_get("KB_RERANK_API_KEY"),
        timeout=_get_float("KB_RERANK_TIMEOUT", DEFAULT_TIMEOUT),
        top_n=_get_int("KB_RERANK_TOP_N", DEFAULT_RERANK_TOP_N),
        wake_cmd=_get("KB_RERANK_WAKE_CMD"),
        wake_timeout=_get_float("KB_RERANK_WAKE_TIMEOUT", DEFAULT_WAKE_TIMEOUT),
    )
    mode = _norm_choice(_get("KB_RETRIEVAL_MODE"), RETRIEVAL_MODES, "auto")
    return RetrievalConfig(mode=mode, embed=embed, rerank=rerank)
