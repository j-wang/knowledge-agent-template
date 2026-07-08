# Toward a Learning Knowledge Agent — Design Notes

**Status:** Design thesis + roadmap. Nothing here is built yet. Captured 2026-06-15 to pick back up.

These notes record a direction for making the knowledge base *learn* semi-autonomously — where the user still curates what goes in and corrects verbally, but the system accumulates, ages, and prioritizes knowledge on its own.

---

## The Core Thesis

Every attempt to put "learning" into a **model** has hurt:

- SLMs — worse than a frontier LLM reading good artifacts.
- Rerankers — learned ranking on a small, evolving corpus overfits and is uninspectable; it silently buried relevant docs.

Every time learning happened in the **corpus** it helped:

- Keyword enrichment during thesis writing (transparent, reversible recall improvement).
- Fine-grained overlapping concepts.
- The adaptive multi-pass retrieval rule itself.

**Design commitment:** the knowledge base is the learnable substrate, and learning is a **write policy over text and metadata — not learned parameters.** A frontier LLM reading good artifacts beats a small model with tuned weights. And unlike weights, every "learned" thing is a **diff you can read, approve verbally, or revert.** That is exactly the semi-autonomous shape we want, with the user as the merge gate.

Corollary: none of this requires new retrieval infrastructure. The intelligence in prioritization and temporal handling lives in metadata the agent can read, plus a loop that keeps the corpus honest.

---

## Problem 1 — Temporal change: type your claims, don't version your docs

Temporality feels hard because concept docs mix epistemically different content. Split it by claim type:

- **Mechanisms** ("how X works") — roughly timeless. Most of the concept layer.
- **State claims** ("X is currently Y") — dated facts that go stale.
- **Judgments / forecasts** — dated and *resolvable*. They can later be scored right or wrong, which is the **best learning signal available.**

Give docs (or sections) frontmatter:

```yaml
as_of: 2024-03-01
status: current | superseded | contested
superseded_by: [[new-slug]]
```

**Never delete or filter — demote and annotate.** A superseded claim still retrieves, but carries its status so the querying agent reasons "this was true as of 2024, replaced by [[new-thing]]." Forgetting-by-demotion is robust; forgetting-by-deletion destroys lineage and makes corrections unrecoverable.

**Reconciliation pass:** when a new source lands, an agent diffs the new extraction's claims against existing docs and emits a **contradiction list**. That list — not the raw source — is what the user reviews.

---

## Problem 2 — Prioritization: deterministic boosts from provenance, not learned ranking

The reranker lesson generalizes. Make priority an **explicit, auditable field**, not a learned weight:

- **Corroboration count** — how many independent sources support this.
- **Endorsement level** — `user-confirmed` vs `agent-inferred` vs `contested`.
- **Recency** — for state claims.

Then either apply small fixed boosts/penalties on top of BM25+RRF, or — likely better — **don't touch the ranker at all.** Surface the metadata in the result snippet and let the *querying agent* do the weighting. LLMs are good at "prefer the user-confirmed 2026 doc over the contested 2023 one" when the metadata is visible; they can't fix a ranker that silently buried it.

Remember: **keywords already are the learnable ranking surface.** Adding a keyword is a transparent, reversible ranking update — which is why that mechanism has worked.

---

## Problem 3 — The learning loop: propose → review → write, plus consolidation

A PR workflow over the KB, with the user as merge gate.

- **Ingestion** stays as-is, but ends with a **proposal queue**: new concepts, keyword additions, contradiction flags, supersession links — presented as diffs to approve or correct verbally.
- **Corrections are first-class artifacts.** When the user says "that's wrong / that changed," the agent updates the doc *and* records why in a correction note, marking the old claim `superseded` with the reason. Corrections are the highest-value training data; never let them evaporate into chat history.
- **Consolidation** is the missing piece that makes it feel like learning. Log every query and — crucially — which docs got **cited in the final answer**, not just retrieved. Periodically run a "librarian" pass:
  - queries that retrieved nothing good → **gap list** (what to acquire next)
  - docs retrieved often but never cited → **keyword pollution**, prune
  - recurring cross-domain query patterns → **candidate new theses**
  - recent corrections → **propagate** to related docs via the `[[link]]` graph

This is the reflection/consolidation pattern from the generative-agents and MemGPT lines of work, but with a human approving the writes.

**The single best priority signal: retrieved-but-never-cited vs. cited.** It's free to collect and requires no model training.

---

## Build Order (first things to do in this repo)

1. **Claim typing in the template.** Add `as_of` / `status` / `superseded_by` / `endorsement` to the concept template, and teach the PRIMER how to read them. Cheap, immediately useful.
2. **Query log.** `search.py` appends query + returned docs; answering agents append which docs they cited. This is the substrate for consolidation.
3. **`/reconcile` consolidation skill.** Reads the log + recent additions, emits a proposal diff for review (contradictions, prune candidates, gap list, new-thesis candidates, correction propagation).

None of this adds retrieval infrastructure. The instinct that the answer wasn't another ranking model was right.

---

## Open questions to revisit

- Granularity of claim typing: per-doc frontmatter vs. per-section annotations. Per-section is more precise but heavier to author — maybe start per-doc and split only docs that mix mechanism + state.
- How forecasts get *scored* when they resolve, and whether that score feeds back as an endorsement signal.
- Whether the query/citation log lives as flat append-only JSONL (simple, greppable) or something queryable — start flat.
- What the proposal-queue artifact actually looks like on disk so it survives across sessions and is reviewable as a diff.
