# Reconstructed VQLS Corrections for AC Power Flow

**Research reproducibility repository** for the study *Reconstructed VQLS Corrections for AC Power Flow: A Negative Certification Benchmark on IEEE 9- and 14-Bus Systems*.

This repository documents a fixed-seed comparison of single-step VQLS candidate corrections with classical Newton baselines, finite-shot ideal-outcome sampling, candidate-centred interval root-inclusion tests, and limited IEEE 9-bus security screening.

> **Scope and limitations.** The VQLS experiments use NumPy sampling from ideal observable-outcome probabilities. They are **not executed Qiskit/Aer circuits or quantum hardware experiments**. The second directed-decimal interval checker is separately implemented but **not externally audited**. IEEE 14-bus results are **root-inclusion studies only**, because native thermal branch ratings are unspecified and some generator voltage setpoints conflict with native upper limits. No quantum speedup is demonstrated.

## Planned repository map

- `archived_study/` — fixed-seed experiment scripts, source case data, and original results.
- `independent_verification/` — separately written directed-decimal interval checker, cross-check scripts, and inclusion outputs.
- `supplement/` — trial-level diagnostics, all-seed plots, ideal-sampling checks, and mathematical provenance.
- `scripts/` — lightweight regeneration/diagnostic scripts.
- `metadata/` — software environment, claim-to-artifact mapping, SHA-256 manifest.
- `requirements.txt` — Python dependencies for the available experiments.

## Run the lightweight checks

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python scripts/validate_ideal_oracle.py
OPENBLAS_NUM_THREADS=1 python scripts/build_additional_diagnostics.py
```

The full archived training and interval audit are much more computationally expensive and should be run only after reading `archived_study/README.md` and `independent_verification/README_VERIFICATION.md`.

**Reproducibility:** the fixed seed list is 711, 713, 717, with four operating scenarios for each of IEEE 9-bus and 14-bus (24 training trials). The experiment archive reports observed inclusion counts, **not estimates of population success probabilities**. In particular, failure of a sufficient interval-inclusion test does not prove the grid is physically infeasible.

After the complete release is pushed, see `metadata/claim_manifest.json` for the output artifact supporting each reported result. Exact historical package versions and an immutable upstream MATPOWER input commit were not fully preserved; consult `metadata/environment_provenance.json`.

Repository: https://github.com/divinelybless/VQLS-Corrections-for-AC-Power-Flow

## Research integrity

The manuscript’s independent numerical verification is *independent in implementation*, not in all inputs or failure modes: it shares benchmark scenarios, trained candidates, and related preconditioner logic with the original analysis. External audit remains welcome. The work does not claim a quantum computational advantage, operational-scale uncertainty certification, or transient stability.
