from pathlib import Path
p=Path('/mnt/data/referee_revision/manuscript_revised.tex')
out=Path('/mnt/data/authors_second_revision/manuscript_authors_revision.tex')
s=p.read_text()
def R(a,b):
 global s
 assert a in s, a[:90]
 s=s.replace(a,b)
R('What Does a Reconstructed Variational Quantum Correction Add to AC Power-Flow Certification?\\\\ A Negative Feasibility Benchmark on IEEE 9- and 14-Bus Systems', 'Reconstructed VQLS Corrections for AC Power Flow:\\\\ A Negative Certification Benchmark on IEEE 9- and 14-Bus Systems')
R('Revised research manuscript --- 2 November 2026', 'Revised research manuscript --- 2 November 2026')
a=s.index('\\begin{abstract}')
b=s.index('\\end{abstract}',a)+len('\\end{abstract}')
s=s[:a]+r'''\begin{abstract}
We assess single-step Newton corrections reconstructed from ideal-outcome, finite-shot variational quantum linear-solver (VQLS) simulations on IEEE 9- and 14-bus AC power-flow models. In eight fixed operating scenarios with three fixed training seeds each, we compare the same warm start against exact undamped one-step Newton, a residual-matched classical diagnostic, and converged Newton. An explicit importance-sampled Pauli estimator and signed-amplitude interference model establish what the simulator's 2.40--2.56 million ideal outcomes represent; the circuits were not executed and no hardware claims are made. Once the full correction is read out, an additional quantum residual estimate is redundant with classical evaluation. Likewise, the tested nonlinear root-inclusion contractor constructs a classical inverse-Jacobian preconditioner; the measured workflow does not avoid factorization. A second, separately written directed-rounding checker agrees with the 24 archived quantum inclusion decisions (3/12 on IEEE 9-bus and 0/12 on IEEE 14-bus) and establishes local enclosures for converged classical candidates in eight scenarios; this is not an external audit. Comparison with numerical reference roots separates three accepted 9-bus base candidates, three 14-bus base candidates lying within the largest tested boxes but failing inclusion, and 18 stressed candidates whose reference roots lie outside all tested candidate-centred boxes. Exact one-step Newton sharply reduces base-case errors; remaining stressed-case errors reflect both single-step nonlinearity and VQLS approximation. Only the three 9-bus base candidates undergo native-voltage and rated-branch screening under an extremely small parameter-uncertainty model; 14-bus results are root-inclusion tests only. The evidence supports an implementation-specific negative certification and feasibility benchmark, not a novel solver or a general quantum disadvantage theorem.
\end{abstract}'''+s[b:]
# Replace incomplete quantum measurement section with operational model
start=s.index('\\subsection{Experimental operator and readout convention}')
end=s.index('\\subsection{Residual redundancy and the appropriate physical check}', start)
s=s[:start]+r'''\subsection{Experimental operator and ideal-outcome sampling protocol}
The final 24-run archive uses the \emph{direct padded nonsymmetric Jacobian}, not the earlier Hermitian embedding. Write $A=J_{p_0}(x_0)\in\mathbb R^{d\times d}$, $p=2^q\ge d$, $\widetilde A=\mathrm{diag}(A,0_{p-d})$, $M=\widetilde A/\|A\|_2$, $K=M^\dagger M$, and $u=(b/\|b\|_2,0_{p-d})$. The normalized real-amplitude tree ansatz $v(\theta)=U_\theta|0^q\rangle$ has $p-1$ trainable rotation angles. With $M=\sum_{\ell}c_\ell P_\ell$ and $K=\sum_j d_jP_j$, the trace-rule coefficients $c_\ell=2^{-q}\operatorname{tr}(P_\ell M)$ may be complex (the nonsymmetric $A$ has odd-$Y$ terms), while $d_j$ are real up to roundoff because $K$ is Hermitian. No imaginary term is discarded. The numerator uses respectively 200 and 864 nonzero $M$ terms (88 and 416 with nonzero imaginary part); the denominator uses respectively 136 and 528 nonzero $K$ terms.

Define $\Gamma_M=\sum_\ell|c_\ell|$, $\Gamma_K=\sum_j|d_j|$ and select independent term indices $L\sim|c_\ell|/\Gamma_M$ or $J\sim|d_j|/\Gamma_K$. Conditional on $L=\ell$, an ideal phase-adjusted Hadamard overlap test prepares
\begin{equation}\label{eq:overlapcircuit}
\frac{|0\rangle|u\rangle+|1\rangle e^{\mathrm i\arg c_\ell}P_\ell|v\rangle}{\sqrt2},\qquad
\mathbb E[H_\ell\mid L=\ell]=\operatorname{Re}\!\left[e^{\mathrm i\arg c_\ell}\langle u|P_\ell|v\rangle\right]
\end{equation}
by measuring ancilla $X$, with $H_\ell\in\{-1,+1\}$. A numerator shot is $Z_a=\Gamma_M H_L$. For the denominator, a Pauli measurement on $|v\rangle$ gives $Q_J\in\{-1,+1\}$ and a shot is $Z_D=\Gamma_K\operatorname{sgn}(d_J)Q_J$. These choices correspond to the \emph{separately and classically assembled} $K=M^\dagger M$, not term-pair sampling. The exact ideal distribution has
\begin{align}\label{eq:unbiased}
\mathbb E Z_a&=a:=\operatorname{Re}\langle u|Mv\rangle,&\operatorname{Var}(Z_a)&=\Gamma_M^2-a^2,\\
\mathbb E Z_D&=D:=\langle v|Kv\rangle,&\operatorname{Var}(Z_D)&=\Gamma_K^2-D^2.
\end{align}
Hence each empirical mean based on $N$ independent outcomes has variance $\operatorname{Var}(Z)/N$ and $|Z_a|\le\Gamma_M$, $|Z_D|\le\Gamma_K$. Equation~\eqref{eq:unbiased} is an estimator specification, not a claim of improved shot complexity. The implementation samples multinomial term counts and binomial signs directly from the analytical Born probabilities, without executing the controlled preparation gates. The input construction of $u$, $M$, and $K$ is classical and should be included in any end-to-end cost ledger.

\paragraph{Signed coordinate measurement and phase convention.} Each physical coordinate $i<d$ is sampled independently with a separate ancilla test preparing $(|0\rangle|i\rangle+|1\rangle|v\rangle)/\sqrt2$. Its $X$ outcome $H_i\in\{\pm1\}$ obeys $\mathbb E H_i=\operatorname{Re}\langle i|v\rangle=v_i$ for the real tree ansatz. Thus $\widehat v_i=2B_i/N_i-1$ for $B_i\sim\operatorname{Binomial}(N_i,(1+v_i)/2)$, $\operatorname{Var}(\widehat v_i)=(1-v_i^2)/N_i$. This measures \emph{signed interference}, not $|v_i|^2$. It requires coherent controlled $U_\theta$ and a specified relative phase between the $|i\rangle$ and $|v\rangle$ preparations. The same circuit-level phase convention is used for the signed numerator overlap $a$; global phase is therefore fixed only relative to these coherent preparation controls, not by computational-basis probabilities. In the archived simulation the ideal circuit amplitudes determine \emph{sampling probabilities}, not the output estimates; no signed amplitude is copied directly into the reconstructed correction. The padded coordinates $i\ge d$ are not read out, physical estimates are \emph{not renormalized}, and different $i$ use disjoint binomial draws. The unmeasured padded component remains part of the normalized training state and can reduce physical correction norm.

\paragraph{Training and holdout scale safeguards.} The sampled training objective is the global projective cost
\begin{equation}\label{eq:traincost}
 \widehat L(\theta)=1-\frac{\widehat a(\theta)^2}{\max\{\widehat D(\theta),0.08\}},\qquad
 L(\theta)=1-\frac{a(\theta)^2}{D(\theta)}.
\end{equation}
The clipping at $0.08$ applies \emph{only during training}. The archived holdout code computes
\begin{equation}\label{eq:zreconstruct}
 \widehat s=\frac{\|b\|_2}{\|A\|_2}\frac{\widehat a}{\widehat D},\qquad \widehat z=\widehat s\widehat v_{1:d},\qquad \widehat x=x_0+\widehat z,
\end{equation}
if $\widehat D>0$, and returns an undefined scale (NaN) otherwise; it does \emph{not} silently use the training clip. All 24 archived holdouts were flagged as allowing a positive-denominator confidence interval. For an executable deployment the acceptance policy must \emph{reject/abstain} if $\widehat D\le0$, if a simultaneously valid lower confidence limit for $D$ is nonpositive when interval scale control is required, or if any reconstructed value is nonfinite. No near-zero-denominator stability guarantee is claimed for the original training code. The supplemental algorithm specifies the original record and this prospective safe-abstention policy separately.

\begin{figure}[htbp]\centering
\fbox{\begin{minipage}{.91\linewidth}\small\textbf{Algorithm 1 (ideal-outcome, fixed-seed record).} Given $A,b,\theta$, decompose $M$ and the classically formed $K=M^\dagger M$. At each SPSA cost call, independently draw 4000 outcomes of $Z_a$ and 4000 of $Z_D$, then use Eq.~\eqref{eq:traincost}. After 120 updates, estimate $a$ and $D$ using 100000 fresh outcomes each. Draw 10000 signed interference outcomes \emph{per physical coordinate} to construct $\widehat v$. The archived study also draws 10000 extra Pauli-residual outcomes per coordinate (redundant after reconstruction). Reconstruct $\widehat z$ by Eq.~\eqref{eq:zreconstruct} when defined; compute $A\widehat z-b$ classically and apply the separate interval checker. All random outcomes in the archived execution are simulated from ideal probabilities, not circuit outputs.\end{minipage}}
\caption{Operational specification and shot ledger for the archived sampling abstraction. The indicated conditional circuits are mathematical realizations of each outcome, not circuits executed in this study.}\label{fig:oraclealgorithm}
\end{figure}

'''+s[end:]
# strengthen theorem
R('Fix a candidate box $\\boxx=\\{x:|x-\\widehat x|\\le \\bar t\\}$ with a positive vector $\\bar t$, and a fixed nonsingular preconditioner $B_0$.', 'Fix a candidate box $\\boxx=\\{x:|x-\\widehat x|\\le \\bar t\\}$ with a positive vector $\\bar t$, and a fixed nonsingular preconditioner $B_0$. Assume $F_p$ is continuously differentiable on an open neighbourhood of $\\boxx$, uniformly for the considered $p\\in\\pp$.')
R('then for every $p\\in\\pp$ there exists exactly one root $x_p^*$ in $\\boxx$', 'then $\\forall p\\in\\pp\\;\\exists!\\,x_p^*\\in\\boxx$ such that $F_p(x_p^*)=0$')
R('direct uncertainty $D_j$ and gradient majorant', 'validated allowance $D_j$ for centre-evaluation and parameter variation at the centre and gradient majorant')
# preconditioner narrower precise
start=s.index('\\section{Certification preconditioner and cost dominance}')
end=s.index('\\section{Experimental protocol and fixed-seed provenance}', start)
s=s[:start]+r'''\section{Computed preconditioner, validated arithmetic and classical cost}\label{sec:precond}
The evaluated contractor forms the binary floating-point matrix
\begin{equation}\label{eq:preconditioner}
B_0=\operatorname{fl}\!\left(J_{p_0}(\widehat x)^{-1}\right)
\end{equation}
by dense numerical inversion at the \emph{reconstructed candidate centre}, not at the warm start $x_0$. The theorem needs a fixed nonsingular $B_0$ for which $|I-B_0J_p(x)|$ is enclosed on the box; it neither requires $B_0$ to equal the exact inverse nor requires an interval matrix enclosing the exact inverse. One may treat each stored binary number in $B_0$ as \emph{exactly represented} and use directed arithmetic when importing and multiplying those entries. The approximation error relative to the mathematical inverse is already represented by the certified $I-B_0J$ defect. No exact inverse containment assertion is made. A strict $\|I-B_0J\|_w<1$ for a nonsingular nominal $J$ also implies $B_0$ is nonsingular. The independent checker imports $B_0$ using exact binary-to-decimal conversion and bounds the subsequent matrix products and strict comparisons; it does not certify the exactness of the numerical factorization.

Computing $B_0$ nevertheless performs a classical Jacobian factorization sufficient to support a further classical Newton update $\Delta x=-B_0F_{p_0}(\widehat x)$ (up to the inverse approximation). This is \emph{not} the same step as the exact one-step comparator from $x_0$, and their end-to-end iteration counts and certification-call counts may differ. The implemented workflow does not avoid classical Jacobian factorization: it adds quantum training and reconstruction before a certification stage that already computes a factorization suitable for Newton refinement. \emph{The reported experiments provide no evidence of an end-to-end benefit.} No general lower bound forbidding savings for other candidate quality, certification methods, iterations or quantum architectures is asserted.

Once the entire correction is reconstructed, a new quantum measurement of $A\widehat z-b$ is informationally redundant; it can be computed classically in $O(\operatorname{nnz}A)$ arithmetic, or enclosed with verified floating-point arithmetic. The archived residual-oracle shots are retained in resource accounting only. We do not claim a new general stopping or shot-allocation theorem for this reconstructed-output design.

'''+s[end:]
# update orig costs section phrasing
R('Hence the classical algebra necessary for a Newton refinement has \\emph{already been performed}. There is no plausible net solver-cost gain from the extra quantum stage in \\emph{this implementation}, independently of a hypothetical per-shot clock speed. A state-only, no-readout solver with an inexpensive certified preconditioner would be a different algorithm.', 'The same factorization could support a classical Newton refinement, but the cost of this candidate-centred calculation is not identical to the warm-start comparator and a universal runtime impossibility does not follow. The added ideal-outcome training, signed readout and redundant residual measurements are uncompetitive in the observed small-system comparison. A state-only, no-readout solver with another certification mechanism would be a different algorithm.')
R('The core finding is stronger than this illustrative timing ratio: the present classical factorization and full readout eliminate the purported avoided classical solve.', 'The precise structural finding is that full readout and candidate-centred classical inversion remove the \emph{claimed avoided factorization} in this implementation; the data do not imply a hardware-independent theorem about every end-to-end solver.')
# add scoping in method after shot description
key='These are \\emph{simulated ideal Pauli outcomes}, not Qiskit-Aer circuits, hardware shots, calibrated noise tests or a physical device gate-level cost.'
R(key,key+r' The sampling implementation computes the conditional Born probabilities from an exact NumPy statevector solely to simulate ideal outcomes; this probability oracle is not evidence that a real circuit estimates the same quantities at the counted gate cost. The independently produced supplemental sampler check verifies unbiasedness against those ideal probabilities in a fresh Monte Carlo test, not gate-level implementation.')
# matched residual clipping qualification
R('The damped LU solution already pays for a complete factorization and is \\emph{not} an inexpensive approximate solver.', 'The damped LU solution already pays for a complete factorization; it is a \emph{diagnostic control}, neither an accuracy-matched nor cost-matched competing solver. In the archived 24 VQLS trials the target residual exceeded $\\|b\\|_\\infty$ in 0 trials, so clipping did not prevent exact infinity-norm matching in this fixed data set.')
# replace figure
R('\\includegraphics[width=\\linewidth]{distance_to_root.pdf}', '\\includegraphics[width=\\linewidth]{all_seed_root_distances.pdf}')
R('Distance to the separately converged nominal root, comparing the identical warm start, median three-seed reconstructed VQLS and residual-matched LU candidates, and the deterministic \\emph{exact one-step Newton} centre. Each curve uses the same eight physical scenarios. The figure was generated from the archived seeds, not an additional optimizer experiment.', 'All 24 fixed-seed reconstructed VQLS candidates plotted individually against the separately computed numerical nominal reference root. Short black bars show within-scenario medians; dashed horizontal levels indicate the largest tested box radii for the respective networks. The vertical axis is a mixed-coordinate numerical infinity norm (angles in radians, PQ magnitudes in p.u.), not a dimensionless physical metric or an interval enclosure. No additional training was performed.')
# insert geometric table before uncertainty paragraph
marker='\\paragraph{Uncertainty generality.}'
geo=r'''\subsection{Reference-root geometry and fixed-radius non-certification}
Every candidate was compared with the \emph{numerically computed nominal reference root} $x^*$ using the largest tested box radius $r_{\max}=0.006$ for IEEE 9-bus and $0.010$ for IEEE 14-bus. This is a diagnostic classification: a numerical reference root outside a box does not prove that \emph{no other} root exists inside it. The 24 trials decompose as follows.
\begin{table}[htbp]\centering\small
\caption{Mutually exclusive diagnoses across the 24 fixed-seed VQLS candidates. ``Outside'' refers solely to the separately computed nominal reference root; ``no inclusion'' is failure of a sufficient test, never proof of absence of a solution.}
\begin{tabular}{@{}lrrr@{}}\toprule
Network & Accepted inclusion & Reference root outside largest box & Reference root inside, no inclusion \\\midrule
IEEE 9-bus (12 trials) & 3 & 9 & 0 \\
IEEE 14-bus (12 trials) & 0 & 9 & 3 \\
Total & 3 & 18 & 3 \\\bottomrule
\end{tabular}\label{tab:geometry}
\end{table}
The three 14-bus base cases demonstrate the distinction most clearly: the nominal reference root lies within the largest candidate-centred box, but the chosen interval majorant does not establish strict inclusion. In all 18 stressed trials the numerical reference root is beyond the largest box, consistent with poor candidate placement and single-step warm-start limitations. Each raw seed and reference-root distance is supplied in \texttt{trial\_level\_geometry.csv}. The gap does not establish that the contractor alone caused all the 14-bus failures.

'''
R(marker,geo+marker)
# independence table in section
ind=r'''\begin{table}[htbp]\centering\scriptsize
\caption{Independence boundary of the second checker. An independently written arithmetic kernel does not make the entire numerical evidence independent.}
\begin{tabularx}{\linewidth}{@{}p{34mm}X@{}}\toprule
Component & Relationship between original and second checker \\\midrule
Scenario inputs and decimals & Same benchmark numerical inputs and outage specifications; second checker parses source decimal tokens directly rather than first rounding them to binary floats. \\
Candidate corrections & Shared/replayed fixed-seed VQLS outputs and warm-start protocol; not independently generated training data. \\
Admittance assembly & Separately coded from the same primitive branch model and identical physical assumptions. \\
Power flow / Jacobian & Separately coded interval equations and interval Jacobian; both share coordinate ordering and PV/PQ conventions. \\
Transcendental intervals & Original \texttt{mpmath.iv}; second exact-rational Taylor remainder plus directed-\texttt{Decimal} arithmetic. \\
Preconditioner & Same classically computed floating inverse at each candidate; second checker imports its binary entries as exact numbers. \\
Test radii / acceptance & Same predetermined radius list; independent strict endpoint comparisons. \\
Voltage / branch screens & Independently coded IEEE 9-bus directed-interval screening, sharing the same native limits and rated branch data; no complete native 14-bus screen. \\\bottomrule
\end{tabularx}\label{tab:independence}
\end{table}
Agreement on the three \emph{positive} VQLS inclusions provides distinct evidence from agreement on the 21 \emph{failed sufficient tests}. A negative flag is not an independently verified root-exclusion theorem, and an external audit has not been performed.
'''
R('To test the reliability of the archived \\texttt{mpmath.iv} verdicts without relying on their float-converted interval endpoints,',ind+'\nTo test the reliability of the archived \\texttt{mpmath.iv} verdicts without relying on their float-converted interval endpoints,')
# modify data avail & discussion some tone
R('The local reproducibility package accompanies this manuscript and contains fixed seeds, the archived operating-case inputs, ideal-outcome sampling scripts, complete numerical logs, and both interval implementations.', 'The local reproducibility package contains fixed seeds, exact benchmark input snapshots used by the code, all four network scenarios and their outage indices, ideal-outcome sampling scripts, trial-level geometry and residual diagnostics, and both interval implementations. A machine-readable claim-to-artifact manifest lists the source of each main numerical claim. It is provided as a working reproducibility bundle, not an immutable public archive.')
R('More extensive quantum training, additional test networks, or heuristic stopping claims are not required to interpret the current evidence.', 'No conclusion is drawn for iterated quantum--Newton methods, noisy devices or realistically uncertain operating regions.')
R('The independently authored code, exact input-token parsing, fixed seeds and endpoint audit results are supplied as supplementary material.', 'The separately written code, exact input-token parsing, fixed seeds and endpoint audit results are supplied as supplementary material, with the shared dependencies disclosed in Table~\\ref{tab:independence}.')
# reference related add verified related source in opening
R('Early quantum power-flow studies \\citep{feng2021,feng2023}', 'Early quantum power-flow studies \\citep{feng2021,feng2023}')
# Supplementary appendix - no novel theorem claim, explicit measurement implementation
append=r'''\appendix
\section{Oracle implementation and reproducibility checks}\label{app:oracle}
For the numerator, conditional on $L=\ell$, the archive computes the ideal mean $\mu_\ell=\operatorname{Re}[(c_\ell/|c_\ell|)\langle u|P_\ell|v\rangle]$ and samples the number of positive ancilla outcomes from $\operatorname{Binomial}(N_\ell,(1+\mu_\ell)/2)$, where $(N_\ell)_\ell\sim\operatorname{Multinomial}(N,|c_\ell|/\Gamma_M)$. The returned mean equals $\Gamma_M(2N_+/N-1)$. The denominator uses the analogous sign-weighted $K$-Pauli distribution; signs of $d_j$ are absorbed in the conditional means. The same random stream creates disjoint coordinate-wise binomial draws for signed amplitudes. Replaying these distributions against exact matrix products in \texttt{validate\_ideal\_oracle.py} verifies the expected means and single-batch variances at fixed test states. This tests the \emph{stochastic specification}, not the cost of a compiled controlled-$U$ circuit, realistic reset/readout, or noise.

The large $K=M^\dagger M$ is formed and Pauli-decomposed classically, which also incurs matrix preparation and coefficient-storage work. The required controlled $U_\theta$ and $U_u$ preparations, and the selection and phase rotations for the Pauli terms, are not charged by an ideal-outcome count. Even the physical amplitudes can only be interpreted as signed overlap samples if such coherent controls are available. Computational-basis measurements alone would give $v_i^2$ and cannot reproduce the archive's binomial law. The main resource conclusions do not assume these omitted costs vanish.

\section{Claim traceability and limitations}\label{app:trace}
The project archive contains \texttt{trial\_level\_geometry.csv}, \texttt{diagnostics.csv}, \texttt{geometry\_summary.json}, \texttt{oracle\_sampler\_check.json}, \texttt{claim\_manifest.json} and exact copies of the base input / experiment scripts. Our new geometric classification reuses archived training states and separately computed nominal reference roots; no new VQLS runs are introduced. Reproducing the same numbers requires keeping the physical coordinate order (non-reference voltage angles in radians followed by PQ magnitudes in p.u.), load multipliers, branch-outage indices, fixed seeds and package versions. Pseudorandom outcomes generated by NumPy need not be byte-for-byte identical under different library versions, and no hardware randomness is modelled.

Before permanent publication, the repository should be assigned an immutable revision and a DOI, and external reviewers should be able to validate the directed-decimal transcendental bounds. Our main interval conclusions concern the explicit boxes and small parameter intervals only. A second checker corroborating \emph{failed} inclusion does not establish physical infeasibility.
'''
R('\\section*{Data and code availability}',append+'\n\\section*{Data and code availability}')
# avoid schematic figure floating inside long section if too much; compile
out.write_text(s)
print('written',out,len(s))
