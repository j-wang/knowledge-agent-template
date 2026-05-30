# Knowledge Agent — Generalized Knowledge Base Builder

A toolkit for building structured, searchable knowledge bases from arbitrary source materials using LLM agents. Feed it PDFs, articles, books, or web content; it produces a layered system of **concept documents**, **thesis documents**, and **hybrid search** that turns a generic LLM into a domain-grounded analyst.

## Origin

This methodology was developed and battle-tested on a markets knowledge agent, meant to replicate James Wang's thought process in markets: ~10,000 pages of scanned financial market reference materials + ~100 web articles, producing 381 concept documents and 54 thesis documents with hybrid BM25 + semantic search. This was further tested on other specialized knowledge areas--including company-specific policy docs (for a "corporate knowledge agent") and rare research areas (women's sexual health, given James's background)--to great effect. The generalized version here captures a domain-agnostic methodology so it can be applied to any subject.

## How It Works

The system operates in three layers, each serving a different retrieval purpose:

| Layer | Location | Purpose | Scale |
|-------|----------|---------|-------|
| **Concepts** | `extracted/concepts/docs/` | "What is X?" — self-contained explanations of individual topics (~1,000 words each) | 15-25 per major source |
| **Theses** | `extracted/concepts/theses/` | Cross-cutting analytical essays threading multiple concepts together (1,500-3,000 words) | 5-10 per source + cross-source |
| **Source Extractions** | `extracted/sources/` | Raw extracted material for deep dives | Full source content |

Search is **BM25 by default** (local, zero-config, no network), with **optional** pluggable dense retrieval (query→doc embeddings, fused with BM25 via Reciprocal Rank Fusion) and an **optional** reranking pass — each enabled purely via environment config. See [Search Infrastructure](#search-infrastructure). A multi-pass retrieval methodology layered on top solves cross-domain vocabulary gaps.

## Quick Start

### 1. Fork/copy this repo for your domain

```bash
cp -r generalized-knowledge-agent/ my-domain-knowledge/
cd my-domain-knowledge/
```

### 2. Place source materials in `input_docs/`

Drop your PDFs, articles, or other source documents into the `input_docs/` directory. This is where all raw source materials should live before extraction.

### 3. Point an LLM agent at CLAUDE.md

The `CLAUDE.md` file contains the complete methodology — extraction strategies, concept synthesis pipeline, thesis generation, search index building, and PRIMER authoring. An LLM agent (Claude Code, or similar) reads this file and executes the pipeline:

```
Phase 1: Extract source material → extracted/sources/
Phase 2: Synthesize concept documents → extracted/concepts/docs/
Phase 3: Generate thesis documents → extracted/concepts/theses/
Phase 4: Build search index → python3 search.py --rebuild
Phase 5: Write the PRIMER → extracted/concepts/PRIMER.md
Phase 6: (Optional but recommended) Add dense retrieval — easiest path is an
        OpenAI key + text-embedding-3-small; then uv run build_embeddings.py.
        See "Search Infrastructure" for the one-line .env setup.
Phase 7: Retrieval test — verify end-to-end quality
```

### 4. Query the knowledge base

```bash
# Basic search
python3 search.py "your query" --top 10

# Verbose with related document expansion
python3 search.py "your query" --top 10 -v --related

# Filter by type
python3 search.py "your query" --type concept
python3 search.py "your query" --type thesis
```

## Project Structure

```
my-domain-knowledge/
├── CLAUDE.md                          # Agent instructions (the methodology)
├── SYSTEM.md                          # Architecture & design rationale
├── README.md                          # This file
├── search.py                          # Hybrid BM25 + semantic search
├── build_embeddings.py                # Dense embedding builder (pluggable provider)
├── retrieval_config.py                # Env-driven retrieval config (all optional)
├── providers.py                       # Pluggable embed/rerank providers
├── .env.example                       # Documented config for all providers
├── input_docs/                        # Place source materials here
│   └── (your PDFs, articles, etc.)
├── extracted/
│   ├── concepts/
│   │   ├── PRIMER.md                  # Agent orientation document (you write this)
│   │   ├── PRIMER_TEMPLATE.md         # Template for the PRIMER
│   │   ├── docs/                      # Concept synthesis documents
│   │   └── theses/                    # Cross-cutting thesis documents
│   └── sources/                       # One subdirectory per source
│       ├── {source-1-slug}/           # Extracted content from source 1
│       └── {source-2-slug}/           # Extracted content from source 2
└── templates/
    ├── concept.md                     # Template for concept documents
    └── thesis.md                      # Template for thesis documents
```

## Customizing for Your Domain

This repo is a **starting point**, not a rigid framework. When you fork it for a specific domain, you should customize:

### CLAUDE.md — The Agent Instructions

This is the most important file to adapt. The extraction methodology is domain-agnostic, but you should:

- **Update the Source Canon** with your actual sources, their biases, and coverage
- **Adjust extraction strategies** based on your source formats (the methodology section covers scanned PDFs, text PDFs, web articles, and more)
- **Tune concept granularity** — some domains benefit from more fine-grained concepts, others from broader ones
- **Add domain-specific synthesis guidance** — tell the agent what kinds of connections to look for when generating theses

### The PRIMER

The PRIMER (`extracted/concepts/PRIMER.md`) is the highest-leverage document in the system. It's what transforms a generic LLM into a domain expert. You must write it fresh for each domain. It should teach querying agents:

- What the knowledge base contains and what each source contributes
- How to search effectively (multi-pass retrieval is critical — see below)
- The domain's key analytical frameworks
- Common analytical patterns and how to apply them
- Known limitations and biases

Use `extracted/concepts/PRIMER_TEMPLATE.md` as a starting point.

### Templates

The concept and thesis templates in `templates/` work well as-is, but you may want to adjust section names or add domain-specific sections (e.g., "Regulatory Context" for legal domains, "Experimental Evidence" for scientific domains).

## Key Design Insights

These are the hard-won lessons from building the markets-knowledge base at scale:

### Multi-Pass Retrieval Solves Cross-Domain Search

This is the single most important insight. Complex questions span multiple domains, but a single search query only targets one domain's vocabulary. BM25 matches keywords, so a query using Domain A jargon will never surface Domain B material — even when they're deeply related.

**The fix:** Decompose questions into separate searches with domain-appropriate vocabulary, follow cross-references, then search again using vocabulary learned from the first pass. This must be documented in your PRIMER.

### Fine-Grained Concepts with Deliberate Overlap

381 concepts with intentional overlap outperforms 100 precise concepts. Different query phrasings hit different docs, and overlap ensures at least one relevant doc surfaces regardless of how the question is worded.

### Theses Are Where the Real Value Lives

Individual concepts explain mechanics. Theses encode the *connections* between concepts — the non-obvious analytical insights that a querying agent can't independently discover from reading individual concept docs.

### Keywords Make or Break Search

The Keywords section in each document (15-30 terms for concepts, 20-40 for theses) is what makes BM25 search work. Include abbreviations, synonyms, jargon, analytical vocabulary, and abstract terms. Be generous.

### The PRIMER Is the Highest-Leverage Document

A well-written PRIMER transforms a generic LLM into a domain analyst. Invest time explaining the domain's key frameworks, not just the file structure.

## Search Infrastructure

`search.py` runs a staged retrieval pipeline. **Only BM25 is on by default** — the
dense and rerank stages are opt-in via environment variables, and every optional
stage **degrades gracefully** (a missing key, an unreachable server, or absent
embeddings falls back to the next-best available ranking with a one-line stderr
note — it never crashes).

### The pipeline

1. **BM25** (always on) — lexical match over keyword-engineered text, with field
   weighting (title 3×, tags 2×, keywords 2×). Local, no network, no extra deps.
2. **Dense retrieval** (optional) — if an embedder is configured **and**
   `.doc_embeddings.npz` exists, the *query* is embedded and cosine-scored against
   the stored doc vectors. BM25 and dense lists are fused with **Reciprocal Rank
   Fusion** (RRF, k=60).
3. **Reranking** (optional) — if a reranker is configured, the top-N fused
   candidates (`KB_RERANK_TOP_N`, default 50) are reordered by a cross-encoder /
   rerank API.

> **Why no "semantic expansion" anymore?** Earlier versions averaged the rows of a
> doc-to-doc similarity matrix to surface "related" docs as primary results.
> Benchmarking showed this *hurt* ranking versus plain BM25 (it displaced strong
> lexical hits with loosely-related docs). It has been removed from the ranking
> path. True *query→doc* dense retrieval + optional rerank is the actual win. The
> doc-to-doc matrix is still built and used **only** for the `--related` display
> flag.

### Zero-config (the default)

```bash
python3 search.py --rebuild     # builds BM25 index + a local TF-IDF doc-doc matrix
python3 search.py "your query"  # BM25 ranking, fully offline
```

No API key, no GPU, no network. This is exactly how the template behaves out of
the box.

### Enabling dense retrieval + reranking

> **Easiest path to good results: an OpenAI API key + `text-embedding-3-small`.**
> If you don't have a GPU, this is the single lowest-effort upgrade over the
> default — no server to run, no model to download, just a key. It adds true
> *query→doc* dense retrieval, and in our benchmarking it gave a clear lift over
> BM25 alone (most of the gain is in *ranking quality* — how high the right
> document lands, not just whether it's somewhere in the results). It's also what
> this template originally shipped with. `text-embedding-3-large` scores slightly
> higher but costs roughly twice as much for a marginal gain, so `small` is the
> recommended default.
>
> ```bash
> # .env  — the whole setup
> KB_EMBED_PROVIDER=openai
> KB_EMBED_MODEL=text-embedding-3-small
> OPENAI_KEY=sk-...          # or OPENAI_API_KEY
> ```
>
> **If you have a GPU**, a local `sentence-transformers` model (e.g.
> `BAAI/bge-m3`) scored best in our benchmark *and* costs nothing per query — but
> it needs more setup (the model download + enough VRAM). Adding a **reranker**
> (TEI or Cohere) improves ranking further still, at the cost of a serving
> dependency and extra per-query latency, so treat it as an optional precision
> mode rather than a default.

Copy `.env.example` to `.env` and uncomment a provider block (the file documents
each one). The retriever is **not pinned to any model or server** — you choose:

| Stage | `KB_EMBED_PROVIDER` / `KB_RERANK_PROVIDER` | Notes |
|-------|--------------------------------------------|-------|
| Embeddings | `none` (default) | BM25 only |
| Embeddings | `openai` | OpenAI **or** any OpenAI-compatible `/v1/embeddings` server (`KB_EMBED_ENDPOINT`) |
| Embeddings | `tei` | HuggingFace text-embeddings-inference (`KB_EMBED_ENDPOINT`) |
| Embeddings | `sentence-transformers` | Local model (optional dep; install `sentence-transformers`) |
| Reranking | `none` (default) | no rerank |
| Reranking | `tei` | TEI rerank endpoint |
| Reranking | `cohere` | Cohere `/v1/rerank` |
| Reranking | `http` | Generic: POST `{query, documents}` → list of scores |

Then build the doc embeddings and refresh the index:

```bash
uv run build_embeddings.py            # embeds the corpus via your provider
python3 search.py --rebuild --bm25-only
python3 search.py "your query"        # now BM25 + dense (RRF), + rerank if configured
```

### Environment variable contract

| Variable | Default | Meaning |
|----------|---------|---------|
| `KB_EMBED_PROVIDER` | `none` | `none` \| `openai` \| `tei` \| `sentence-transformers` |
| `KB_EMBED_MODEL` | — | Provider-specific model id |
| `KB_EMBED_ENDPOINT` | — | Base URL for an OpenAI-compatible or TEI server |
| `KB_EMBED_API_KEY` | falls back to `OPENAI_KEY` / `OPENAI_API_KEY` | Embed API key |
| `KB_EMBED_TIMEOUT` | `10` | Per-request timeout (s) |
| `KB_EMBED_WAKE_CMD` | — | Shell command to wake an unreachable embed host (e.g. `wakeonlan ..`); then poll + retry |
| `KB_EMBED_WAKE_TIMEOUT` | `120` | Seconds to wait for the embed host after waking |
| `KB_RERANK_PROVIDER` | `none` | `none` \| `tei` \| `cohere` \| `http` |
| `KB_RERANK_MODEL` | — | Provider-specific model id |
| `KB_RERANK_ENDPOINT` | — | Base URL for the rerank server |
| `KB_RERANK_API_KEY` | — | Rerank API key |
| `KB_RERANK_TIMEOUT` | `10` | Per-request timeout (s) |
| `KB_RERANK_TOP_N` | `50` | How many fused candidates to rerank |
| `KB_RERANK_WAKE_CMD` | — | Shell command to wake an unreachable rerank host; then poll + retry |
| `KB_RERANK_WAKE_TIMEOUT` | `120` | Seconds to wait for the rerank host after waking |
| `KB_RETRIEVAL_MODE` | `auto` | `auto` \| `bm25` \| `dense` \| `hybrid` |

`auto` = hybrid (RRF of BM25 + dense) when doc embeddings are present and an
embedder is available, otherwise BM25-only. `--bm25-only` on the CLI forces BM25
regardless of config.

### Graceful-degradation chain

- Reranker configured but unreachable / errors → **skip rerank**, keep fused order.
- Embedder configured but unreachable, or `.doc_embeddings.npz` missing/stale →
  **BM25 only**.
- Endpoint unreachable **and** a `*_WAKE_CMD` is set → run it, poll the endpoint
  for up to `*_WAKE_TIMEOUT`, then retry once; if it still doesn't come up,
  fall back as above. (For a self-hosted box that auto-suspends.)
- `sentence-transformers` selected but not installed → warn once, **BM25 only**.
- Nothing configured → **BM25 only** (the default), fully offline.

In every case the search still returns results; the only signal is a single
stderr line noting the fallback.

## Setup

### Environment

**No configuration is required** for the default BM25 search. To enable optional
dense retrieval / reranking, copy `.env.example` to `.env` and uncomment a
provider block:

```bash
cp .env.example .env
# Edit .env — uncomment e.g. the OpenAI, TEI, or sentence-transformers block.
```

Every variable is documented in `.env.example` with copy-paste-ready examples for
OpenAI, a generic TEI endpoint, local sentence-transformers, a Cohere reranker,
and a TEI reranker. See the [environment variable contract](#environment-variable-contract)
above.

### Dependencies

Both Python scripts use [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so dependencies are declared in the scripts themselves. Run with [`uv`](https://github.com/astral-sh/uv):

```bash
# Search (also works with plain python3 if deps are installed)
uv run search.py "query"

# Build doc embeddings (uses whatever KB_EMBED_PROVIDER you configured;
# with the default provider=none it builds a local TF-IDF doc-doc matrix)
uv run build_embeddings.py
```

Core dependencies stay light: `numpy`, `rank-bm25`, `scikit-learn`. HTTP
to embedding/rerank servers uses the Python standard library (`urllib`), so no
extra runtime dependency is required for `openai`/`tei`/`cohere`/`http` providers.
The `sentence-transformers` provider is an **optional** dependency, imported
lazily only if you select it (`pip install sentence-transformers`).

### Search Indexes and Embeddings

The generated index and embedding files (`.search_index.pkl`, `.similarity.npy`, `.doc_embeddings.npz`) are excluded from git by default via `.gitignore`. However, once your knowledge base is mature and these files are expensive to recompute, we recommend removing them from `.gitignore` and committing them. This avoids unnecessary recalculation when others fork or specialize your knowledge agent.

## License

This project is licensed under the MIT License.

Created by James Wang · [Weighty Thoughts](https://weightythoughts.com)

If you use this template, a mention or link back is appreciated.
