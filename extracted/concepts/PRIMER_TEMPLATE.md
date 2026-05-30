# {Domain} Knowledge Base — Agent Primer

You are querying a structured knowledge base built from curated sources on {domain description}. This primer tells you what's available, how to search it effectively, and how to think about the material.

## What This Knowledge Base Contains

The knowledge base draws from {N} primary sources:

**Source 1: {Source Name} ({size/scope})**
{Brief description: what it covers, what analytical framework it uses, what it's strongest on.}

**Source 2: {Source Name} ({size/scope})**
{Brief description. Note how it complements or contrasts with Source 1.}

{Add more sources as needed.}

The material is organized in three layers:

**Layer 1: Concept Documents ({N} files in `concepts/docs/`)**
Short synthesis documents (~1,000 words), each covering a single concept. These are your primary search targets. Each has YAML frontmatter with tags, source references, and cross-references to related concepts. The Keywords section lists search terms designed to maximize retrieval.

**Layer 2: Thesis Documents ({N} files in `concepts/theses/`)**
Longer analytical essays (1,500–3,000 words) that argue positions by threading narratives across multiple concepts. These make the non-obvious cross-concept connections. Highest value for complex analytical questions.

**Layer 3: Source Extractions**
Full extractions from original materials in `sources/`. Use when you need depth beyond what concept/thesis docs provide.

## How to Search This Knowledge Base

You have **two retrieval operations**. They are the contract — everything below assumes only these two:

| Operation | What it does | Parameters |
|---|---|---|
| **search** | ranked hybrid (keyword + semantic) search; returns id, title, snippet, cross-references | `query` (required) · `top` (default 10) · `type` = `concepts` \| `theses` (optional) · `related` = also list semantically-similar docs per hit (optional) |
| **getdoc** | return the full text of one document | `id` (required) |

Your runtime exposes these in **one of two ways — use whichever your environment gives you** (you do not need both, and you should not assume one without checking):

**A. Shell CLI** — for agents that can run commands (e.g. Claude Code):
```bash
python3 search.py "your query terms" --top 10 -v          # search
python3 search.py "your query" --top 10 -v --related      # search + semantic neighbors per hit
python3 search.py "your query" --type theses              # filter to one layer
python3 search.py "your query" --json                     # machine-readable output
cat extracted/concepts/docs/<id>.md                       # getdoc: read a full document
#   (theses live in extracted/concepts/theses/<id>.md)
```

**B. Native tool calls** — for API-driven models given function-calling tools:
```
search({"query": "...", "top": 10, "type": "theses", "related": true})   # → ranked results
getdoc({"doc_id": "..."})                                                 # → full document text
```
These must be wired as **first-class tools** by your harness. Do **not** drive them through a hand-rolled "emit `ACTION: …`" text protocol — a model parsing its own text output over-searches and fails to converge. If neither binding (A nor B) is present, your harness is not set up: stop and see the deployment contract in `README.md` before querying.

The **related / semantic-neighbor** option is valuable — it surfaces cross-domain connections you wouldn't find by keyword alone.

### Multi-Pass Retrieval (Adaptive — gate it)

Decomposing a question into several searches is the **highest-leverage lever for hard, multi-faceted questions** — but it is **not free, so apply it adaptively.** Decomposition only helps when a single search *under-retrieves*. If you decompose a question that one broad query already covers, the extra searches pull in loosely-related documents — and the danger is what you then do with them: padding an answer with marginally-relevant material leads to **fabricated specifics** (you synthesize details the docs don't actually support). On a focused/factual knowledge base this *hurts* answer quality; on a broad analytical one it helps. So gate decomposition on **retrieval completeness, not on how "hard" the question feels.**

**The lazy / gap-filling pattern:**
1. **Start with one broad search** for the question as asked.
2. **Assess coverage.** For every facet the question names, did the results surface a relevant doc *with tight relevance*? (Concretely: each named facet has at least one hit that is clearly on-topic for it, not just adjacent.) If yes → **stop and answer. Do not decompose further.**
3. **Decompose only the gaps.** For each facet left uncovered, run a separate search with vocabulary appropriate to that domain.
4. **Follow cross-references** in the docs you find.
5. **Search again** with vocabulary learned from the first passes — and **stop when new searches stop surfacing new relevant docs** (retrieval has saturated).
6. **Synthesize** across everything retrieved.

The signal is single-shot recall *sufficiency*: decompose when one query leaves facets uncovered; don't when it already has them.

**Example:** "{Example complex question spanning two domains}"

- **One broad search first.** If it already returns on-topic docs for *both* domain A and domain B, answer from them — you're done.
- **If a facet is missing** (e.g. the broad query only surfaced domain A material): search "{domain B terms}" to fill that gap, then search bridging themes discovered in both sets → finds the thesis linking them.

A single combined query often can't reach a second domain's vocabulary at all — that's the gap multi-pass closes. But reach for it only when the first pass demonstrably left a facet uncovered, not reflexively.

### Search Tips

- **Start broad, then narrow.** First search teaches you which concepts exist and what vocabulary the knowledge base uses.
- **Use thesis docs for analytical depth.** If you find a relevant concept, check cross-references for theses that synthesize it with other concepts.
- **Source extractions for raw detail.** Concept/thesis layer is synthesis. Go to source files for specific data points, charts, or detailed case studies.
- **All sources matter.** For complex questions, material from different sources complements each other — {explain how sources complement}.

## Key Analytical Frameworks

{This section is domain-specific and should be written when populating the knowledge base. It should explain the 3-5 most important analytical frameworks or mental models that the source material uses. This is what transforms a generic LLM into a domain-grounded analyst.}

### {Framework 1}
{Explain the framework and how the source material applies it.}

### {Framework 2}
{Explain the framework and how the source material applies it.}

### {Framework 3}
{Explain the framework and how the source material applies it.}

## How to Apply These Frameworks

{Domain-specific guidance on answering different types of questions. Examples:}

**For "explain X" questions:** Search for X, read relevant concept docs, pull key mechanics and examples.

**For "compare X to Y" questions:** Search for both separately. Identify structural similarities and differences. Be explicit about where analogies hold and where they break.

**For "what would happen if" questions:** Trace the causal chain using the domain's frameworks. What is the first-order effect? What secondary dynamics does it trigger? What's the historical precedent?

**For "lessons from history" questions:** Search for the historical episode, then separately search for the modern parallel. Map the mechanics (not surface details) to the current situation.

## Important Caveats

- {Source 1} reflects {perspective/timeframe}. {What's dated vs. still relevant.}
- {Source 2} covers {timeframe} and is {more/less current} on {specific topics}.
- Source material may have {known quality issues — OCR errors, missing charts, etc.}
- Strongest coverage: {topics}. Thinner on: {topics}.

## File Structure Quick Reference

```
extracted/
├── concepts/
│   ├── PRIMER.md                  ← You are here
│   ├── docs/                      ← {N} concept synthesis documents
│   └── theses/                    ← {N} cross-cutting analytical documents
└── sources/
    ├── {source-1-slug}/           ← {Source 1} extractions
    └── {source-2-slug}/           ← {Source 2} extractions
```
