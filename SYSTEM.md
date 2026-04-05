# Knowledge Base — System Design

## What This Is

A domain-agnostic toolkit for building structured, machine-readable knowledge bases from arbitrary source materials. The system extracts knowledge from PDFs, articles, and other documents, synthesizes it into a layered concept/thesis structure, and provides hybrid search for LLM agent retrieval.

This design was proven on ~10,000 pages of scanned financial reference materials and ~100 web articles, producing 406 indexed documents with hybrid BM25 + semantic search. The architecture generalizes to any knowledge domain.

## Architecture

### Layer 1: Concept Documents

**Location:** `extracted/concepts/docs/`

Each concept is a self-contained markdown document covering a single topic. They average ~1,000 words and follow a consistent format: YAML frontmatter with tags/sources/related concepts, then Summary, Key Mechanics, Historical Examples, Practical Implications, Cross-References, and Keywords sections.

**Design rationale:** The concept layer answers "what is X?" and "how does X work?" The granularity is deliberately fine — many concepts with intentional overlap — because finer granularity improves RAG retrieval. Different query phrasings hit different docs, and overlap means at least one relevant doc surfaces regardless of phrasing.

**Format choices:**
- YAML frontmatter enables structured filtering before semantic search
- Cross-references use `[[slug]]` format for programmatic linking
- Keywords section (15-30 terms per doc) maximizes search recall — includes abbreviations, synonyms, jargon, analytical vocabulary
- Tags (8-15 per doc) are broader categorical labels for faceted search
- `source:` tag in frontmatter tracks provenance back to the original source

### Layer 2: Thesis Documents

**Location:** `extracted/concepts/theses/`

Each thesis is a longer analytical essay (1,500-3,000 words) that argues a position by threading a narrative across multiple concepts. These encode the non-obvious cross-concept connections — the analytical insights that individual concept docs can't provide.

**Design rationale:** Concept docs are encyclopedic — they explain mechanics neutrally. The most valuable knowledge lives in *connections* between concepts. The thesis layer was specifically designed to address a failure mode observed in testing: agents could find the right concept docs but struggled to make non-obvious analytical connections across them.

**Relationship to concepts:** Each thesis references 5-10 concept docs. During thesis writing, agents also enrich keywords on referenced concept docs with analytical/abstract terms, improving concept-layer search recall as a side effect.

### Layer 3: Source Extractions

**Location:** `extracted/sources/{source-slug}/`

Full markdown extractions from original source material. These contain raw text, structured chart descriptions, data tables, and detailed analysis. Too large and unstructured for direct RAG retrieval, but invaluable as a second-pass resource for depth.

**Extraction methodology varies by source format:**
- Scanned PDFs: Claude vision (chunk-per-agent) with Tesseract OCR fallback
- Text PDFs: pdftotext with chart annotation
- Web articles: text extraction with chart annotation
- See CLAUDE.md for full methodology documentation

### Supporting Files

- **`PRIMER.md`** — Agent orientation document. Teaches querying agents how the knowledge base is organized, how to search effectively, and how to think in the domain's analytical frameworks. Should be loaded first in any retrieval pipeline.
- **`search.py`** — Hybrid BM25 + semantic search. Indexes all concept and thesis docs.
- **`build_embeddings.py`** — Neural embedding builder for higher-quality cross-domain similarity (requires OpenAI API).

## How Retrieval Should Work

The intended retrieval flow for a querying agent:

1. **Load the primer** — orients the agent's analytical framework
2. **Decompose complex questions** into distinct analytical components
3. **Search each component separately** with domain-appropriate vocabulary
4. **Read concept docs** (5-15 relevant), follow cross-references
5. **Search thesis docs** for analytical depth across concepts
6. **Go to source docs if needed** — for specific data, charts, deeper detail

Steps 1-4 are sufficient for most questions. Step 5 significantly improves analytical depth. Step 6 is only for deep dives.

## Why This Architecture

### Why three layers instead of one?

Source documents alone are too large for efficient retrieval. Concept summaries alone lose analytical depth and miss cross-concept connections. The three-layer design lets retrieval start fast and narrow (concepts), expand analytically (theses), and go deep when needed (sources).

### Why fine-grained concepts with deliberate overlap?

RAG retrieval is a recall problem. If the user asks about "X" and our only doc is called "Y" (where X and Y are related but different terms), the semantic distance might cause a miss. Multiple fine-grained docs with overlapping coverage create multiple retrieval targets for any query. The overlap means we'll hit at least one relevant doc regardless of phrasing.

### Why thesis documents exist separately?

Cross-concept insights don't live in any single concept doc. They live in the *relationships* between concepts. Thesis docs explicitly encode these relationships so querying agents don't have to independently discover them.

### Why keyword enrichment during thesis writing?

Concept docs initially have domain-specific keywords. But an analyst might search using abstract or analytical vocabulary that doesn't match domain jargon. During thesis writing, agents discover these cross-concept connections and add abstract vocabulary to concept keywords. This enrichment is analytically driven — the terms are added because a thesis proved the connection.

## Search Infrastructure

### Hybrid BM25 + Semantic Search (`search.py`)

**Architecture:**
1. **BM25 layer** — Okapi BM25 with title (3x), tags (2x), keyword (2x) weighting
2. **Semantic similarity layer** — Precomputed NxN doc-to-doc similarity matrix. For each BM25 hit, surfaces related docs from other domains sharing no query vocabulary
3. **Result fusion** — ~70% BM25 slots, ~30% reserved for semantic expansion. Both normalized to [0,1] before combining

**Two modes for the similarity matrix:**
- **Neural embeddings (recommended):** `build_embeddings.py` with OpenAI API. Best quality for cross-domain bridging (0.5-0.6 cosine between related cross-domain docs)
- **TF-IDF fallback:** Built by `search.py --rebuild`. Fully local, no API. Adequate within-domain but weak cross-domain (0.04-0.09 cosine between cross-domain docs)

Both write to `.similarity.npy` — `search.py` loads whichever exists.

### Why Multi-Pass Retrieval Beats Better Search Infrastructure

The single most important retrieval insight: no search infrastructure — not embeddings, not TF-IDF, not RRF — can overcome the vocabulary gap between source domains in a single query. Neural embeddings correctly rank same-domain docs higher than cross-domain docs because they *are* more relevant to any single-domain query.

The fix is in how agents use search, not in search itself. Complex questions need decomposition into separate searches, each with domain-appropriate vocabulary. This is documented in the PRIMER so querying agents learn to do it.

## Maintenance

When the knowledge base changes:

1. Rebuild BM25 + TF-IDF: `python3 search.py --rebuild`
2. Rebuild neural embeddings: `uv run build_embeddings.py` (optional, requires OpenAI API)
3. Both scripts use PEP 723 inline metadata for `uv run` compatibility
4. Update CLAUDE.md status tracking
5. Update PRIMER.md if new sources change what querying agents need to know
