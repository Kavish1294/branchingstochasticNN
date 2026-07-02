# Branching Stochastic Neural Networks — Manuscript & Validation Suite

This bundle contains a **draft** manuscript and an independent, object-oriented
suite that empirically checks the mathematical claims made in it.

---

## ⚠️ Important notice about the manuscript

`fissionNeuralNetwork_DRAFT.pdf` is an **incomplete work in progress**.

- It has **not** been peer-reviewed, independently verified, or approved for any purpose.
- **No one should use, rely on, cite, quote, reproduce, or distribute** the manuscript —
  or any figure, equation, method, benchmark, or result in it — in its present form.
- Its definitions, proofs, and results are provisional, may contain errors, and are
  subject to substantial revision or withdrawal without notice.
- The manuscript is a **mathematical abstraction only**. It specifies no material
  compositions, cross sections, device geometries, enrichment levels, control protocols,
  or operational design parameters, and is not intended for the design or operation of any
  physical system.

The same disclaimer appears on a cover page and as a watermark inside the PDF itself.

---

## Contents

| File | Description |
|------|-------------|
| `fissionNeuralNetwork_DRAFT.pdf` | The draft manuscript, with a disclaimer cover page + per-page watermark and restricted permissions (see below). |
| `bsnn_validation.py` | Object-oriented validation / benchmark suite (26 checks across 9 theorem areas). |
| `fig_T2_moments.png` | Mean & variance propagation: simulation vs. theory (Thm 6.1). |
| `fig_T3_criticality.png` | Population norm vs. generation in the sub-/critical/supercritical regimes (Thm 7.1). |
| `fig_T8_comparison.png` | BSNN vs. unconstrained RNNs: one-step RMSE and homogeneity violation (Sec 14). |
| `README.md` | This file. |

---

## About the PDF restrictions (and their limits)

The PDF is encrypted (AES-256) with an **empty user password**, so it **opens normally
without prompting**, but an owner password locks its permissions:

- **Content copying (text/graphics extraction): disabled.**
- **Printing (low- and high-resolution): disabled.**
- **Editing / annotation / form-filling / assembly: disabled** (read-only).

Two honest caveats:

1. **These flags are advisory.** Compliant viewers (Adobe Acrobat, most browsers'
   built-in viewers, macOS Preview) honor them, but they can be removed with free tools.
   They deter casual copying/printing; they are **not** strong security.
2. **A PDF file cannot prevent its own download.** "No download" is a property of the
   website/app that *hosts* a file, not of the file itself. Once someone has the file, they
   have it. Disabling printing is the nearest file-level analog and is included above.

---

## Validation suite

### What it does

`bsnn_validation.py` does not take the manuscript's math on faith. It re-checks each
central claim by simulation, exact linear algebra, **independent reverse-mode automatic
differentiation** (the `autograd` package), and finite differences.

| Test | Manuscript claim | Method |
|------|------------------|--------|
| **T1** | Branching PGF factorization `G_{x+y}=G_x·G_y` (Thm 4.3) | Analytic (Poisson head) + Monte-Carlo with a correlated **non-Poisson** offspring law |
| **T2** | Mean `x_{n+1}=M x_n` and covariance `Σ_{n+1}=MΣ_nMᵀ+ΣᵢxₙᵢVᵢ` (Thm 6.1) | 400k-trajectory simulation vs. closed-form recursions |
| **T3** | `‖xₙ‖^{1/n} → ρ(M)`, three regimes + critical limit vector (Thm 7.1) | Normalized power iteration; Perron-vector check |
| **T4** | Extinction = minimal fixed point of `q=g(q)`; `ρ⋚1` dichotomy (Thm 7.3 / App A) | Fixed-point iteration vs. simulated extinction frequency |
| **T5** | Backprop through the mean model **==** adjoint sensitivity (Thm 10.1) | Adjoint vs. `autograd` vs. finite differences |
| **T6** | Dominant-eigenvalue sensitivity `dρ/dθ = uᵀ(dM)v` (Thm 10.2) | Closed form vs. finite differences |
| **T7** | Discrete-adjoint benchmarks (Sec 10.1 & 11.1) | Exact 3-cell rationals; AES `AD == adjoint == FD` to roundoff |
| **T8** | BSNN vs. RNNs + homogeneity/additivity limits (Sec 12 & 14) | Fit 3 models; measure structural violations |
| **T9** | Finite-sample Hoeffding bound for `M̂` (Thm 9.1) | Empirical coverage over 3,000 trials |

**Result:** all **26/26** checks pass. The 3-cell benchmark reproduces the paper's exact
rationals (`J = 8/85`, `dJ/dθ = −1072/7225`), and reverse-mode AD matches the hand-derived
adjoint to ~1e-17.

### Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`
- `autograd`  (for the independent gradient check)

```bash
pip install numpy scipy matplotlib autograd
```

### Run

```bash
python3 bsnn_validation.py
```

It prints a PASS/FAIL line per check with quantitative diagnostics, a summary, and writes
the three figures to the working directory. Exit code is `0` iff all checks pass.

### Architecture

The suite is organized in four layers:

1. **Models** — `OffspringLaw` (abstract) with `PoissonOffspring`, `BinomialOffspring`,
   `MultinomialFissionOffspring`; a single `BranchingProcess` that composes any law.
2. **Analysis** — `MeanFieldOperator` (spectral radius, Perron vectors, moment
   propagation, power iteration), `ExtinctionSolver`, `MeanRecurrenceAdjoint`,
   `EigenvalueSensitivity`, and the `DiscreteAdjointBenchmark` subclasses.
3. **Fitting** — `SequenceModel` (abstract) with `BSNNMeanModel`, `LinearRNNModel`,
   `NonlinearRNNModel`; plus `StructureDiagnostics`.
4. **Validation** — a `CheckResult` dataclass, a `Reporter`, an abstract `Validator`,
   eight concrete validators (one per theorem area), and a `ValidationSuite` orchestrator.

Adding a new check is a new `Validator` subclass registered in `ValidationSuite`.

---

## How to read the results

Passing these tests validates the manuscript's **internal mathematical consistency** — that
the theorems are correctly stated and the benchmarks reproduce. It does **not** validate the
*physical* premise that real fission satisfies the independent-branching approximation; that
is an empirical modeling question outside the manuscript's synthetic scope. The manuscript's
own §12 (expressivity limits) and Appendix C (limitations) should be read alongside these
results.

---

## Attribution

The manuscript authorship is "to be added" per the draft. This validation suite is an
independent harness and is **not** affiliated with or endorsed by the manuscript authors.
