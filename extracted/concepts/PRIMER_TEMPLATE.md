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

**Use `search.py` for retrieval:**

```bash
# Basic search
python3 search.py "your query terms" --top 10 -v

# With semantically related docs shown per hit
python3 search.py "your query" --top 10 -v --related

# Filter by type
python3 search.py "your query" --type theses

# JSON output for programmatic use
python3 search.py "your query" --json
```

The `--related` flag is valuable — it shows documents semantically similar to each hit, surfacing cross-domain connections you wouldn't find by keyword alone.

### Multi-Pass Retrieval (Critical)

**Do not rely on a single search query.** Complex analytical questions have multiple facets, and each facet may require its own search. This is especially important when a question spans different source domains.

The pattern:
1. **Decompose the question** into its distinct analytical components
2. **Search each component separately** with domain-appropriate vocabulary
3. **Follow cross-references** in the docs you find
4. **Search again** based on what you learned — the first pass teaches you the vocabulary of the knowledge base

**Example:** "{Example complex question spanning two domains}"

This has two components requiring separate searches:
- **Component A:** Search "{domain A terms}" → finds concepts about A
- **Component B:** Search "{domain B terms}" → finds concepts about B
- **Bridging:** After reading both sets, search for connecting themes → finds thesis linking them

A single combined query would only find one domain's material. Multi-pass retrieval closes this gap.

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
