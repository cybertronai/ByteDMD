# Renaming Data Movement Distance Metric

**Source:** https://gemini.google.com/share/af75298aa427
**Created with:** Gemini 3 Deep Think, April 8, 2026
**Topic:** Brainstorming alternative names for ByteDMD

---

## Prompt
Reuse the idea of data movement distance but relax "byte" and use something else instead of ByteDMD. Brainstorm names that emphasize a concrete count, tightening up the original complexity-based metric.

## Brainstormed Names

### 1. Emphasizing "Exact Tally" (contrasting with Big-O)
- **ExactDMD** — directly contrasts with asymptotic bounds
- **TraceDMD** — highlights dynamic execution tracing
- **CountDMD** — emphasizes discrete counting
- **TallyDMD** — implies running sum, instruction by instruction
- **DiscreteDMD** — highlights jump from continuous bounds to discrete integer steps

### 2. Generalizing the Data Unit (replacing "Byte")
- **ElementDMD / ElemDMD** — matches the README's "element-level LRU stack"
- **WordDMD** — classic architecture term, not locked to 8 bits
- **ItemDMD** — Pythonic and abstract
- **UnitDMD** — flexible, determined by bytes_per_element multiplier

### 3. Evolving the Acronym (replacing "Distance")
- **DMC (Data Movement Cost)** — the actual sum total you pay
- **DMC (Data Movement Count)** — emphasizes counting
- **DMS (Data Movement Score)** — benchmarking/leaderboard feel

### 4. Physical Metaphor (VLSI & 2D Routing)
- **WireDMD / WireCost** — evokes physical, measurable connection length
- **GridDMD** — evokes 2D Manhattan distance grid
- **RouteCost** — concrete toll for routing length

## Summary Recommendations
- For concrete number emphasis: **ExactDMD** or **TraceDMD**
- For fixing the "byte" issue while keeping the rhythm: **ElementDMD**
- For benchmarking metric feel: **Data Movement Cost (DMC)**
