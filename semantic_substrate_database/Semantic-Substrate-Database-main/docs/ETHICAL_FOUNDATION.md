# Ethical Foundation and Transparency

Semantic Substrate Database (SSDB) explores value-aligned data storage. The project treats ethics
as a first-class design concern and documents the assumptions that shape its behaviour.

## Why Declare Values?
- **Traceability** – Users can see, inspect, and change the value anchor rather than inheriting
  opaque defaults.
- **Comparability** – Forks can substitute their own anchors and make the differences explicit.
- **Accountability** – Researchers can reason about system behaviour against the published anchor.

## The Reference Anchor
The baseline system measures four semantic coordinates (love, power, wisdom, justice) relative to a
reference point nicknamed “Jehovah” at `(1.0, 1.0, 1.0, 1.0)`. This reference is:

- A convenient mathematical ideal (unit vector in each dimension).
- A reflection of the author’s worldview.
- A prompt to consider how alternative anchors would influence behaviour.

Nothing in the repository prevents replacing the anchor with a different tuple. Swap the values,
adjust the sacred-number tables, and document the new interpretation.

## Practical Effects
- Semantic distance and “divine resonance” scores are computed relative to the anchor.
- Sacred-number heuristics default to widely cited biblical numerology but can be patched.
- ICE framework decisions factor the coordinates when recommending execution strategies.
- Modern semantic enrichment interprets everyday language (analytical, ethical, clarity-focused) and can be tuned via `SSDB_SEMANTIC_WEIGHT` / `SSDB_ICE_WEIGHT`.
- Optional sentence-embedding support (`SSDB_USE_EMBEDDINGS`) provides deeper semantic similarity when `sentence-transformers` is installed.

## Responsible Use
1. **Disclose modifications** – If you publish a fork, state how the anchor changed.
2. **Report limitations** – The numbers are heuristics, not moral proofs.
3. **Invite critique** – Transparency is only useful if stakeholders can respond.

## Further Reading
- `docs/TECHNICAL_WHITEPAPER.md` – Architectural deep dive.
- `docs/DOCKER.md` – Container deployment notes.
- `api/API_README.md` – REST API usage guide.

Maintained by BruinGrowly • MIT License.
