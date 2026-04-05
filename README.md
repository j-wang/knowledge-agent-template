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

Search is hybrid BM25 (keyword) + semantic similarity (TF-IDF or OpenAI embeddings), with a multi-pass retrieval methodology that solves cross-domain vocabulary gaps.

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
Phase 6: (Optional) Build neural embeddings → uv run build_embeddings.py
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
├── build_embeddings.py                # Neural embedding builder (OpenAI API)
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

### BM25 + Semantic Hybrid

`search.py` implements hybrid search with:
- **BM25** with field weighting (title 3x, tags 2x, keywords 2x)
- **Semantic expansion** via precomputed document similarity matrix
- **Result fusion** (~70% BM25, ~30% semantic)

### Two Similarity Modes

| Mode | Command | Cross-Domain Quality | Requirement |
|------|---------|---------------------|-------------|
| **Neural embeddings** (recommended) | `uv run build_embeddings.py` | High (0.5-0.6 cosine) | OpenAI API key |
| **TF-IDF** (fallback) | `python3 search.py --rebuild` | Low (0.04-0.09 cosine) | None |

Neural embeddings are worth the API cost if your knowledge base spans multiple sources with different vocabularies.

## Setup

### Environment

Copy `.env.sample` to `.env` and add your OpenAI API key (only needed for neural embeddings):

```bash
cp .env.sample .env
# Edit .env and set OPENAI_KEY=sk-...
```

### Dependencies

Both Python scripts use [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so dependencies are declared in the scripts themselves. Run with [`uv`](https://github.com/astral-sh/uv):

```bash
# Search (also works with plain python3 if deps are installed)
uv run search.py "query"

# Build embeddings (requires OPENAI_KEY in .env)
uv run build_embeddings.py
```

Core dependencies: `numpy`, `rank-bm25`, `scikit-learn`, `scipy`, `pyyaml`, `openai` (for embeddings only).

### Search Indexes and Embeddings

The generated index and embedding files (`.similarity.npy`, `.embeddings.npz`, `.embedding_doc_order.json`, `.bm25_cache.pkl`) are excluded from git by default via `.gitignore`. However, once your knowledge base is mature and these files are expensive to recompute, we recommend removing them from `.gitignore` and committing them. This avoids unnecessary recalculation when others fork or specialize your knowledge agent.

## License

This project is licensed under the MIT License.

Created by James Wang · [Weighty Thoughts](https://weightythoughts.com)

If you use this template, a mention or link back is appreciated.
