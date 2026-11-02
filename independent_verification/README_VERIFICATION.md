# Separate-implementation root verification and originality assessment

This package contains a self-contained cross-implementation verification, not an external peer audit.

* Source: `independent_decimal_krawczyk.py` constructs AC interval power flow, analytic Jacobian interval rows, and a Krawczyk test from original text decimal parameters. No `mpmath.iv` is used.
* Directed decimal arithmetic at 65 digits and exact-rational endpoint Taylor remainders enclose trigonometric evaluations. Original float candidate and preconditioner entries are treated as exactly represented binary IEEE-754 values. Assumption: fixed PV/PQ status and the parameter box defined in the paper.
* `quantum_full_audit_corrected.py`: reruns 24 training seed/scenario cases from the fixed original algorithm and checks each at the specified box radii. This script uses the included archive of original training code, which must be extracted into the expected `work/` location.
* `classical_comparison_audit.py`: tests classical Newton candidate inclusion independently; all 8 scenarios have a positive radius at 1e-5 or 3e-5 at δ=1e-7.
* `quantum_full_corrected.json`: comparison data for 24 quantum trained candidates. Only 3 9-bus base-case candidates pass. All 14-bus tests are root-only and none passes.
* The independent 9-bus complex branch-flow screen checks both ends of all 9 rated branches, in MVA on the 100MVA base.
* `numerical_and_novelty_audit.pdf`: review, reported margins, limitations, source comparison and publication decision.
* `manuscript_with_crosscheck.pdf`: revised manuscript with the new cross-verification section.

Important limits: no third-party audit of the new independent code, no independent certification of floating-point shot allocation optimisation, no hardware/Qiskit experiment, no generator Q-limit switching, no IEEE14 native thermal security statement. Do not call uncertainty-failed root inclusions grid instability.
