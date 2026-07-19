# Fairness Pareto front — averaged over seeds

3 seeds (7, 42, 2024), full 217k data, identical protocol per seed. Deltas are **paired**: each variant is compared against the no-mitigation baseline *of its own seed*, so the seed's intrinsic difficulty cancels out.

| variant | Macro-F1 | ΔF1 (paired) | DI | ΔDI (paired) | verdict |
|---|---:|---:|---:|---:|---|
| none | 0.7591 ± 0.0049 | +0.0000 ± 0.0000 | 3.828 ± 0.094 | +0.000 ± 0.000 | reference |
| mask-names | 0.7575 ± 0.0051 | -0.0016 ± 0.0015 | 3.816 ± 0.051 | -0.012 ± 0.073 | **no measurable effect** |
| scrub | 0.7566 ± 0.0039 | -0.0025 ± 0.0010 | 3.462 ± 0.024 | -0.366 ± 0.071 | real effect |
| scrub+mask | 0.7563 ± 0.0032 | -0.0028 ± 0.0027 | 3.464 ± 0.059 | -0.365 ± 0.100 | real effect |
| counterfactual | 0.7522 ± 0.0060 | -0.0069 ± 0.0019 | 3.281 ± 0.061 | -0.547 ± 0.105 | real effect |

## Reading

- **mask-names** — ΔDI -0.012 (sd 0.073, 0.2× its own spread), costing 0.1313 Macro-F1 per DI point. Per-seed ΔDI: [-0.084, -0.015, 0.062].
- **scrub** — ΔDI -0.366 (sd 0.071, 5.2× its own spread), costing 0.0068 Macro-F1 per DI point. Per-seed ΔDI: [-0.401, -0.413, -0.285].
- **scrub+mask** — ΔDI -0.365 (sd 0.100, 3.6× its own spread), costing 0.0076 Macro-F1 per DI point. Per-seed ΔDI: [-0.465, -0.364, -0.265].
- **counterfactual** — ΔDI -0.547 (sd 0.105, 5.2× its own spread), costing 0.0126 Macro-F1 per DI point. Per-seed ΔDI: [-0.652, -0.546, -0.443].
