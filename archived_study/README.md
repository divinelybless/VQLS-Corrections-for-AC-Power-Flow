# IEEE power-grid robustness and novelty-gate reproducibility package

**Status:** exploratory numerical research; no executed Qiskit/Aer or quantum device. 2 November 2026.

## Main deliverables

- `robustness_novelty_addendum.tex` and `.pdf`: five-page research addendum, statement and proof, fixed-budget robustness, classical fallback audit, and publication stopping criteria.
- `results.json`: complete 24-trial outcomes (3 seeds x 4 scenarios x 2 network cases). Both parameter-interval levels use the exact same test protocol for each correction method.
- `interval_audit.json`: root-centred classical checks at four physical uncertainty levels using an independent candidate-centred Krawczyk routine without an internal nonlinear root solve. Results use a different and explicitly broader radius grid from the main fixed-budget comparison.
- `summary_compact.json`, `precision_check.json`, `parity_check.json`: numerical summaries and cross-checks.
- `acceptance_by_scenario.pdf`: acceptance plot.
- `references.bib`: editable bibliography.

## Reproduction

Use Python 3.11+ with NumPy, SciPy, mpmath, matplotlib. Qiskit and Qiskit Aer are optional for separate reference circuits.

```bash
python -m pip install numpy scipy mpmath matplotlib
OPENBLAS_NUM_THREADS=1 python run_robustness.py
OPENBLAS_NUM_THREADS=1 python run_interval_audit.py
OPENBLAS_NUM_THREADS=1 python precision_check.py
OPENBLAS_NUM_THREADS=1 python parity_check.py
OPENBLAS_NUM_THREADS=1 python make_report.py
pdflatex -interaction=nonstopmode robustness_novelty_addendum.tex
pdflatex -interaction=nonstopmode robustness_novelty_addendum.tex
```

IMPORTANT: the saved outputs already contain the results and are sufficient to inspect the findings. Running all experiments can take a few minutes. No `bibtex` executable is required for PDF generation: the manuscript includes a built-in bibliography, while the BibTeX file is included separately for later manuscript integration.

A reference circuit (not executed in the study) is in `deps/qiskit_gate_verification_optional.py`; to run in a Qiskit-enabled environment:

```bash
python -m pip install qiskit qiskit-aer
PYTHONPATH=deps OPENBLAS_NUM_THREADS=1 python deps/qiskit_gate_verification_optional.py
```

A gate-level NumPy reference is also in `deps/gate_level_test.py`.

## Fixed experimental design

IEEE 9: intact, +20% active/reactive demand, N-1 removal of line 4-5, and +10% load with removal of line 9-4.

IEEE 14: intact, +20% demand, removal of line 1-2, and +10% load with removal of line 5-6.

All tests use 120 sampled SPSA iterations and fixed seed list [711,713,717]. The starting state is a small perturbation of the *base-case classical power-flow solution* and is shared by all operating conditions. The new scenario root is never used for quantum training. This means initial warm-start computation is a classical cost and must not be hidden when interpreting quantum efficiency.

Main certificate comparisons: primitive relative branch uncertainty 1e-7 or 1e-6; injection uncertainty equal to 0.01 times the primitive relative uncertainty (per unit). The same predeclared radius grid is used for all method candidates in each network.

Secondary classical audit: uncertainty 1e-7, 1e-6, 1e-4, 1e-3, separately using smaller candidate-centred radius boxes to verify that fixed-large-box failures are not physical infeasibility statements.

## Important scientific limitations

1. Interval arithmetic uses `mpmath.iv` with high precision but has not been independently formally audited for outward rounding, input decimal exactness, IEEE physical model limits, or thermodynamic realism.
2. The quantum training and holdout records are sampled from *ideal analytically computed probability distributions*. Circuit-based Qiskit and physical quantum runs were NOT performed. An attempted Qiskit installation failed because this environment exposed no installable distributions.
3. Three training seeds for each physical condition are not sufficient to estimate broad deployment acceptance rates reliably.
4. A candidate-centred Krawczyk strict inclusion proves conditional local root enclosure, not full security: voltage and branch limits are separate tests. IEEE 14 uses native voltage upper limit 1.06 while several setpoints exceed it; no thermal branch ratings were assumed for that case.
5. Failure of an inclusion check says the selected box did not certify the root. It is not a proof of system instability or infeasibility.
6. The deliberately residual-matched classical LU comparator is a diagnostic, not a naturally cost-matched classical algorithm.
7. Classical computations dominate for these small cases; no quantum advantage is established.

## Publication decision

The evidence supports a limited **decision-aware quantum-compatible certification/feasibility paper** only if a genuinely differentiating theorem or integrated algorithm is established in a focused prior-art audit. It does not support claims of new VQLS power-flow computation or quantum speedup. Do not keep extending trials without a new falsifiable hypothesis.
