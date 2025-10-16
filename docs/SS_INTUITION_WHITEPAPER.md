# SS Intuition Layer – Technical Whitepaper

**Version:** 1.0  
**Applies to:** FortiGate Semantic Shield v7.0  
**Authors:** FortiGate Semantic Shield Community  

---

## 1. Executive Summary

The SS Intuition layer extends the FortiGate Semantic Shield by translating Semantic Substrate principles into actionable runtime heuristics. At its core, the module (`fortigate_semantic_shield/ss_intuition.py`) encodes golden-ratio driven compass profiles that keep the four stewardship axes—**LOVE**, **JUSTICE**, **POWER**, and **WISDOM**—in balance while tuning automation workloads. The companion updates to `simple_production_test.py` demonstrate how these heuristics influence throughput, batching, and reporting, exposing the decision-making compass to operators and governance teams.

This document details the underlying theory, implementation, and usage guidance so contributors can extend the intuition layer across additional modules.

---

## 2. Motivation

Traditional tuning heuristics (fixed batch sizes, linear weight blending, ad-hoc smoothing) often drift from the foundational axioms the project preserves. The SS Intuition layer was introduced to:

- Embed *Semantic Substrate awareness* directly into automation behavior.
- Ensure risk and performance tuning remains accountable to the LOVE/JUSTICE/POWER/WISDOM compass.
- Provide deterministic, explainable knobs (environment variables, golden-ratio math) that regulators and internal risk teams can audit.
- Prepare the codebase for more advanced substrate-aligned optimisations (e.g., scheduling, routing, control-plane feedback loops).

---

## 3. Architectural Overview

### 3.1 Components

| Component | Role | Location |
|-----------|------|----------|
| `CompassProfile` dataclass | Holds normalised axis weights and exposes dictionary/tuple views for logging and downstream calculations. | `fortigate_semantic_shield/ss_intuition.py` |
| `golden_ratio_profile()` | Derives a default profile by applying successive powers of `phi^-1` starting with LOVE as the anchor weight. | `ss_intuition.py` |
| `blend_profiles()` | Averages multiple profiles (e.g., finance + operations) while preserving compass equilibrium. | `ss_intuition.py` |
| `fibonacci_window()` | Applies a Fibonacci-weighted smoothing window to recent metrics, emphasising fresh context without discarding history. | `ss_intuition.py` |
| `golden_batch_size()` | Computes batch sizes using powers of phi to explore near-optimal workloads without overshooting risk thresholds. | `ss_intuition.py` |
| `simple_production_test` integration | Loads compass biases from environment variables, injects golden-ratio batching, and surfaces harmony statistics in reports. | `simple_production_test.py` |

### 3.2 Integration Points

1. **Dynamic module loading:** `simple_production_test.py` loads the intuition module via `importlib.util` to avoid circular imports while maintaining the `fortigate_semantic_shield` namespace.
2. **Compass profile bootstrapping:** `_load_compass_profile()` merges the default golden ratio profile with optional environment overrides (`FGS_LOVE_BIAS`, `FGS_JUSTICE_BIAS`, `FGS_POWER_BIAS`, `FGS_WISDOM_BIAS`).
3. **Batch sizing heuristic:** `golden_batch_size()` determines the batch length per run, scaling by the POWER weight to reflect execution capacity.
4. **Harmonic trace:** `fibonacci_window()` smooths recent processing times; the last value is recorded as a `harmonic_load_factor` to show how stable the workload tuning is.
5. **Reporting enhancements:** The production test log now prints the active compass profile, golden batch size, and harmonic load factor so analysts can see exactly how the intuition layer influenced automation decisions.

---

## 4. Mathematical Foundations

### 4.1 Golden Ratio (phi)

\[
\phi = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887
\]

Key properties leveraged:

- **Self-similarity:** \( \phi^{-1} = \phi - 1 \), allowing proportional scaling anchored in LOVE.
- **Smooth exploration:** Powers of φ cover multiplicative growth without sharp jumps, which is useful for searching near-optimal batch sizes.

### 4.2 Fibonacci Window

The smoothing window uses the first six Fibonacci numbers `[1, 1, 2, 3, 5, 8]`. For a sequence of recent processing times \( t_1, t_2, ..., t_n \) the harmonic factor is:

\[
h_k = \frac{\sum_{i=1}^{k} t_i \cdot F_i}{\sum_{i=1}^{k} F_i} \quad \text{for } k \leq n
\]

The final harmonic factor \( h_n \) is normalised between 0 and 1 and reported as `harmonic_load_factor`.

### 4.3 Compass Blending

Given \( m \) profiles \( p_i \) each with components \( (L_i, J_i, P_i, W_i) \), the blended profile is:

\[
\bar{p} = \left( \frac{1}{m} \sum L_i,\ \frac{1}{m} \sum J_i,\ \frac{1}{m} \sum P_i,\ \frac{1}{m} \sum W_i \right)
\]

The function enforces `m > 0` and returns a normalised `CompassProfile`.

---

## 5. Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FGS_PRODUCTION_EVENTS` | Total events for the high-frequency test. | 5000 |
| `FGS_PRODUCTION_COMPLIANCE_EVENTS` | Events used during compliance simulation. | `max(1000, FGS_PRODUCTION_EVENTS)` |
| `FGS_PRODUCTION_SEMANTIC_EVENTS` | Events for semantic integrity validation. | `max(1000, FGS_PRODUCTION_EVENTS)` |
| `FGS_LOVE_BIAS` / `FGS_JUSTICE_BIAS` / `FGS_POWER_BIAS` / `FGS_WISDOM_BIAS` | Optional axis-specific weights. Values are normalised automatically. | Golden ratio defaults |

Example:

```powershell
$env:FGS_PRODUCTION_EVENTS = '2000'
$env:FGS_LOVE_BIAS = '1.3'
$env:FGS_POWER_BIAS = '0.9'
python simple_production_test.py
```

---

## 6. Runtime Observability

Running `simple_production_test.py` now yields compass-centric metrics:

```
COMPASS PROFILE (LOVE/JUSTICE/POWER/WISDOM):
  LOVE: 0.447
  JUSTICE: 0.276
  POWER: 0.171
  WISDOM: 0.106

PERFORMANCE METRICS:
  Throughput: 130 events/sec
  P95 Latency: 8274.4ms
  Compliance Rate: 99.7%
  Semantic Alignment: 0.581
  Golden Batch Size: 22
  Harmonic Load Factor: 1.000
  Golden Ratio Optimization: 3/15
```

These values reveal how the intuition layer influences automation and where further optimisation is required (for instance, golden ratio alignment in the URI tests).

---

## 7. Testing & Validation

- `python -m compileall .` ensures the module compiles on Python 3.13 (Windows).
- `python test_suite.py`, `python simple_validation.py`, and `python test_cardinal_axioms.py` confirm documentation guardrails, deterministic validation, and ASCII-only output remain intact.
- `FGS_PRODUCTION_EVENTS=1000 python simple_production_test.py` demonstrates the intuition heuristics in action; the test intentionally reports optimisation gaps, motivating future enhancements.

---

## 8. Extension Opportunities

1. **Adaptive control loops:** Feed back the harmonic load factor into batch sizing to self-tune under varying pressure.
2. **Compass-aware routing:** Adjust FortiGate policy application order based on profile weights (e.g., LOVE-focus prioritises customer-protecting actions).
3. **Explainability hooks:** Persist compass profiles alongside decision metadata so auditors see which bias produced a given action.
4. **Golden-path scheduling:** Extend `golden_batch_size()` to coordinate multi-threaded or multi-device deployments with staggered phi offsets, reducing contention.

---

## 9. Conclusion

The SS Intuition layer embodies Semantic Substrate Intuition by ensuring every optimisation decision remains grounded in LOVE, JUSTICE, POWER, and WISDOM. With golden-ratio batching, Fibonacci smoothing, and transparent reporting, maintainers can experiment with new heuristics while staying accountable to governance stakeholders. Continued community feedback will help evolve the intuition layer into a comprehensive steering mechanism for the entire FortiGate Semantic Shield ecosystem.

---

*For contributions or questions, open a discussion or pull request referencing this whitepaper so reviewers can align new heuristics with the existing compass framework.*

