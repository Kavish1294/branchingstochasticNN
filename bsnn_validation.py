#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bsnn_validation.py  --  object-oriented validation / benchmark suite
=====================================================================

Empirical validation of the manuscript

    "Branching Stochastic Neural Networks: A Count-Valued Recurrent Network
     View of Fission Multiplication"

The suite is organised into four layers:

    1. MODELS      -- OffspringLaw hierarchy + BranchingProcess
    2. ANALYSIS    -- MeanFieldOperator, ExtinctionSolver, adjoint/benchmark tools
    3. FITTING     -- SequenceModel hierarchy + StructureDiagnostics  (for Sec 14)
    4. VALIDATION  -- Reporter, Validator hierarchy, ValidationSuite

Each Validator subclass targets one theorem/claim and reports PASS/FAIL with
quantitative diagnostics.  Run:

    pip install autograd
    python3 bsnn_validation.py

Dependencies: numpy, scipy, matplotlib, autograd.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import nnls

import autograd.numpy as anp
from autograd import grad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================== #
#  Numeric helpers (module-level: used by several layers and by autograd)
# =========================================================================== #
def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    x = np.asarray(x, float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def ag_softplus(x):
    """Softplus written in autograd primitives (for differentiable graphs)."""
    return anp.log1p(anp.exp(-anp.abs(x))) + anp.maximum(x, 0.0)


# =========================================================================== #
#  LAYER 1 -- MODELS
# =========================================================================== #
class OffspringLaw(ABC):
    """One-parent offspring distribution of a multitype branching process.

    A type-i parent produces a random offspring vector Y_i in N_0^m.  The
    mean-offspring matrix is M[j, i] = E[(Y_i)_j].
    """

    @property
    @abstractmethod
    def n_types(self) -> int: ...

    @abstractmethod
    def mean_matrix(self) -> np.ndarray:
        """M with M[j, i] = E[(Y_i)_j]."""

    @abstractmethod
    def offspring_from(self, parent_type: int, n: int, rng) -> np.ndarray:
        """Sample n offspring vectors (shape (n, m)) from n type-`parent_type` parents."""

    def covariances(self) -> Optional[List[np.ndarray]]:
        """List [Cov(Y_i)]; None if not provided in closed form."""
        return None

    def one_parent_pgf(self, z: np.ndarray) -> np.ndarray:
        """Vector (g_1(z), ..., g_m(z)); override where a closed form exists."""
        raise NotImplementedError

    def conditional_next(self, X: np.ndarray, rng) -> Optional[np.ndarray]:
        """Fast sampler for X_{n+1} | X_n = X (batch, shape (B, m)).

        Return None to signal that no closed-form conditional is available,
        in which case BranchingProcess falls back to per-parent sampling.
        """
        return None


class PoissonOffspring(OffspringLaw):
    """Independent-Poisson head: (Y_i)_j ~ Poisson(M[j, i]) independently.

    Because sums of independent Poissons are Poisson, the exact conditional is
    X_{n+1, j} | X_n ~ Poisson((M X_n)_j)  (the manuscript's Eq. 73).
    Moments: E[Y_i] = M[:, i], Cov(Y_i) = diag(M[:, i]).
    """

    def __init__(self, M: np.ndarray):
        self._M = np.asarray(M, float)
        self._m = self._M.shape[0]

    @property
    def n_types(self) -> int:
        return self._m

    def mean_matrix(self) -> np.ndarray:
        return self._M.copy()

    def covariances(self) -> List[np.ndarray]:
        return [np.diag(self._M[:, i]) for i in range(self._m)]

    def one_parent_pgf(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        return np.exp((z - 1.0) @ self._M)            # entry i = g_i(z)

    def offspring_from(self, parent_type: int, n: int, rng) -> np.ndarray:
        lam = np.broadcast_to(self._M[:, parent_type], (n, self._m))
        return rng.poisson(lam)

    def conditional_next(self, X: np.ndarray, rng) -> np.ndarray:
        return rng.poisson(X @ self._M.T).astype(float)


class BinomialOffspring(OffspringLaw):
    """Bounded head: (Y_i)_j ~ Binomial(B, p[j, i]), so (Y_i)_j in [0, B] a.s.

    Used to exercise the Hoeffding bound (Thm 9.1), which needs bounded support.
    Mean matrix M[j, i] = B * p[j, i].
    """

    def __init__(self, trials: int, probs: np.ndarray):
        self._B = int(trials)
        self._P = np.asarray(probs, float)            # P[j, i] = p_ji
        self._m = self._P.shape[0]

    @property
    def n_types(self) -> int:
        return self._m

    def mean_matrix(self) -> np.ndarray:
        return self._B * self._P

    def offspring_from(self, parent_type: int, n: int, rng) -> np.ndarray:
        p = self._P[:, parent_type][None, :]
        return rng.binomial(self._B, p, size=(n, self._m))


class MultinomialFissionOffspring(OffspringLaw):
    """Genuinely non-Poisson, correlated head (a fission-like multiplicity model).

    A type-i parent draws a total multiplicity K_i in {0, ..., Kmax} from
    `mult_probs[i]`, then assigns each of the K_i children a type from
    categorical `type_probs[i]`.  Offspring components are correlated (shared
    total) and non-Poisson -- ideal for stress-testing the branching
    factorization so the identity is not a Poisson tautology.
    """

    def __init__(self, mult_probs: Sequence[Sequence[float]],
                 type_probs: Sequence[Sequence[float]]):
        self._mult = [np.asarray(p, float) for p in mult_probs]
        self._tp = [np.asarray(p, float) for p in type_probs]
        self._m = len(self._tp)

    @property
    def n_types(self) -> int:
        return self._m

    def mean_matrix(self) -> np.ndarray:
        # E[(Y_i)_j] = E[K_i] * p_type[i][j]
        M = np.zeros((self._m, self._m))
        for i in range(self._m):
            mean_K = np.dot(np.arange(len(self._mult[i])), self._mult[i])
            M[:, i] = mean_K * self._tp[i]
        return M

    def offspring_from(self, parent_type: int, n: int, rng) -> np.ndarray:
        pmult = self._mult[parent_type]
        ptype = self._tp[parent_type]
        K = rng.choice(len(pmult), size=n, p=pmult)
        out = np.zeros((n, self._m), dtype=int)
        for kval in range(1, len(pmult)):             # K=0 contributes nothing
            idx = np.where(K == kval)[0]
            if idx.size:
                out[idx] = rng.multinomial(kval, ptype, size=idx.size)
        return out


class BranchingProcess:
    """A multitype branching process driven by an OffspringLaw.

    Provides one-generation stepping, full trajectory simulation, the
    transition PGF G_x(z) = prod_i g_i(z)^{x_i}, replicated single-generation
    sampling (for factorization tests), and extinction-frequency estimation
    with freezing of exploded trajectories.
    """

    def __init__(self, law: OffspringLaw, seed: int = 0):
        self.law = law
        self.m = law.n_types
        self.rng = np.random.default_rng(seed)

    # -- moments / PGF ------------------------------------------------------ #
    @property
    def mean_matrix(self) -> np.ndarray:
        return self.law.mean_matrix()

    def transition_pgf(self, x: np.ndarray, z: np.ndarray) -> float:
        g = self.law.one_parent_pgf(z)
        return float(np.prod(g ** np.asarray(x, float)))

    # -- dynamics ----------------------------------------------------------- #
    def _step_batch(self, X: np.ndarray, rng) -> np.ndarray:
        fast = self.law.conditional_next(X, rng)
        if fast is not None:
            return fast
        out = np.zeros_like(X, dtype=float)
        Xi = X.astype(int)
        for b in range(X.shape[0]):
            for i in range(self.m):
                k = int(Xi[b, i])
                if k:
                    out[b] += self.law.offspring_from(i, k, rng).sum(axis=0)
        return out

    def step(self, X: np.ndarray) -> np.ndarray:
        return self._step_batch(np.asarray(X, float), self.rng)

    def simulate(self, x0: np.ndarray, n_gen: int, n_traj: int) -> np.ndarray:
        """Return trajectory tensor of shape (n_gen + 1, n_traj, m)."""
        X = np.tile(np.asarray(x0, float), (n_traj, 1))
        out = np.empty((n_gen + 1, n_traj, self.m))
        out[0] = X
        for n in range(n_gen):
            X = self.step(X)
            out[n + 1] = X
        return out

    def replicate_generation(self, x: np.ndarray, n_rep: int) -> np.ndarray:
        """n_rep independent realisations of ONE generation from a fixed x."""
        x = np.asarray(x, int)
        tot = np.zeros((n_rep, self.m), dtype=int)
        for i in range(self.m):
            if x[i]:
                kids = self.law.offspring_from(i, n_rep * int(x[i]), self.rng)
                tot += kids.reshape(n_rep, int(x[i]), self.m).sum(axis=1)
        return tot

    def extinction_frequency(self, x0: np.ndarray, horizon: int, n_traj: int,
                             cap: int = 10 ** 6, seed: Optional[int] = None) -> float:
        """Empirical P(extinction).  Trajectories whose total exceeds `cap` are
        frozen (they never go extinct), which avoids Poisson overflow in the
        supercritical regime; extinction means hitting exactly zero."""
        rng = np.random.default_rng(seed)
        X = np.tile(np.asarray(x0, float), (n_traj, 1))
        for _ in range(horizon):
            tot = X.sum(axis=1)
            active = (tot > 0) & (tot < cap)
            if not active.any():
                break
            X[active] = self._step_batch(X[active], rng)
        return float(np.mean(X.sum(axis=1) == 0))


# =========================================================================== #
#  LAYER 2 -- ANALYSIS
# =========================================================================== #
class MeanFieldOperator:
    """Positive-operator theory for the mean matrix M (>= 0).

    Wraps spectral radius, Perron vectors, mean/covariance propagation, and
    normalised power iteration.
    """

    def __init__(self, M: np.ndarray):
        self.M = np.asarray(M, float)
        self.m = self.M.shape[0]

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.M))))

    def perron_vectors(self):
        """Return (k, v, u): dominant eigenvalue with right/left eigenvectors
        made positive and normalised so that u^T v = 1."""
        w, V = np.linalg.eig(self.M)
        k = int(np.argmax(w.real))
        v = np.real(V[:, k]); v = np.abs(v / np.sign(v[np.argmax(np.abs(v))]))
        wl, Vl = np.linalg.eig(self.M.T)
        kl = int(np.argmax(wl.real))
        u = np.real(Vl[:, kl]); u = np.abs(u / np.sign(u[np.argmax(np.abs(u))]))
        u = u / (u @ v)
        return float(w[k].real), v, u

    def propagate_mean(self, x0: np.ndarray, n: int) -> List[np.ndarray]:
        xs = [np.asarray(x0, float)]
        for _ in range(n):
            xs.append(self.M @ xs[-1])
        return xs

    def propagate_covariance(self, x0: np.ndarray, offspring_covs: List[np.ndarray],
                             n: int):
        """Return (means, covs) using
        Sigma_{n+1} = M Sigma_n M^T + sum_i x_{n,i} V_i, with Sigma_0 = 0."""
        means = self.propagate_mean(x0, n)
        covs = [np.zeros((self.m, self.m))]
        for k in range(n):
            xk = means[k]
            noise = sum(xk[i] * offspring_covs[i] for i in range(self.m))
            covs.append(self.M @ covs[-1] @ self.M.T + noise)
        return means, covs

    def power_iteration(self, x0: np.ndarray, n_max: int) -> dict:
        """Normalised power iteration.  The per-step growth factor -> rho fast;
        exp(mean log factor) = ||x_n||^{1/n} -> rho slowly (Thm 7.1)."""
        x = np.asarray(x0, float) / np.linalg.norm(x0, 1)
        log_cum, lognorms, ratios = 0.0, [0.0], []
        for _ in range(n_max):
            y = self.M @ x
            r = np.linalg.norm(y, 1)
            ratios.append(r)
            log_cum += np.log(r)
            lognorms.append(log_cum)
            x = y / r
        return {
            "ratio": ratios[-1],
            "nth_root": float(np.exp(log_cum / n_max)),
            "lognorms": np.array(lognorms),
        }


class ExtinctionSolver:
    """Minimal fixed point of q = g(q) by monotone iteration from q = 0."""

    def __init__(self, pgf: Callable[[np.ndarray], np.ndarray], n_types: int,
                 iters: int = 20_000, tol: float = 1e-14):
        self.pgf = pgf
        self.m = n_types
        self.iters = iters
        self.tol = tol

    def solve(self) -> np.ndarray:
        q = np.zeros(self.m)
        for _ in range(self.iters):
            qn = self.pgf(q)
            if np.max(np.abs(qn - q)) < self.tol:
                q = qn
                break
            q = qn
        return np.clip(q, 0.0, 1.0)


class MeanRecurrenceAdjoint:
    """Mean-BSNN recurrence  x_{n+1} = softplus(A) x_n,  x_0 fixed,
    loss L = 0.5 ||x_N||^2.  Exposes the adjoint gradient (Thm 10.1), an
    independent reverse-mode-AD gradient, and a finite-difference gradient.
    """

    def __init__(self, A: np.ndarray, x0: np.ndarray, n_steps: int):
        self.A = np.asarray(A, float)
        self.x0 = np.asarray(x0, float)
        self.N = int(n_steps)

    def _forward(self, A: np.ndarray):
        M = softplus(A)
        xs = [self.x0.copy()]
        for _ in range(self.N):
            xs.append(M @ xs[-1])
        return M, xs

    def loss(self, A: Optional[np.ndarray] = None) -> float:
        _, xs = self._forward(self.A if A is None else A)
        return 0.5 * float(xs[self.N] @ xs[self.N])

    def adjoint_gradient(self) -> np.ndarray:
        M, xs = self._forward(self.A)
        lambdas: List[Optional[np.ndarray]] = [None] * (self.N + 1)
        lambdas[self.N] = xs[self.N].copy()           # grad of 0.5||x_N||^2
        for n in range(self.N - 1, -1, -1):
            lambdas[n] = M.T @ lambdas[n + 1]
        sig = sigmoid(self.A)                          # dM/dA entrywise
        G = np.zeros_like(self.A)
        for n in range(self.N):
            G += sig * np.outer(lambdas[n + 1], xs[n])
        return G

    def autograd_gradient(self) -> np.ndarray:
        x0, N = self.x0, self.N

        def loss_ag(A):
            M = ag_softplus(A)
            x = x0
            for _ in range(N):
                x = M @ x
            return 0.5 * anp.sum(x * x)

        return grad(loss_ag)(self.A)

    def finite_difference_gradient(self, h: float = 1e-6) -> np.ndarray:
        G = np.zeros_like(self.A)
        for a in range(self.A.shape[0]):
            for b in range(self.A.shape[1]):
                Ap = self.A.copy(); Ap[a, b] += h
                Am = self.A.copy(); Am[a, b] -= h
                G[a, b] = (self.loss(Ap) - self.loss(Am)) / (2 * h)
        return G


class EigenvalueSensitivity:
    """Sensitivity of rho(softplus(A)) to A (Thm 10.2):
    d rho / d A_ab = sigma(A_ab) * u_a * v_b, with u^T v = 1."""

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, float)

    def formula_gradient(self) -> np.ndarray:
        op = MeanFieldOperator(softplus(self.A))
        _, v, u = op.perron_vectors()
        return sigmoid(self.A) * np.outer(u, v)

    def finite_difference_gradient(self, h: float = 1e-6) -> np.ndarray:
        G = np.zeros_like(self.A)
        for a in range(self.A.shape[0]):
            for b in range(self.A.shape[1]):
                Ap = self.A.copy(); Ap[a, b] += h
                Am = self.A.copy(); Am[a, b] -= h
                rp = MeanFieldOperator(softplus(Ap)).spectral_radius
                rm = MeanFieldOperator(softplus(Am)).spectral_radius
                G[a, b] = (rp - rm) / (2 * h)
        return G


class LinearResponseSystem:
    """Solve A phi = q; response J = w^T phi; adjoint A^T lambda = w."""

    def __init__(self, A: np.ndarray, q: np.ndarray, w: np.ndarray):
        self.A = np.asarray(A, float)
        self.q = np.asarray(q, float)
        self.w = np.asarray(w, float)
        self.phi = np.linalg.solve(self.A, self.q)
        self.adjoint = np.linalg.solve(self.A.T, self.w)

    @property
    def response(self) -> float:
        return float(self.w @ self.phi)


class DiscreteAdjointBenchmark(ABC):
    """Base for the two discrete-adjoint benchmarks (Sec 10.1 / 11.1)."""

    @abstractmethod
    def run(self, reporter: "Reporter") -> None: ...


class ThreeCellBenchmark(DiscreteAdjointBenchmark):
    """Fully specified 3-cell benchmark (Sec 11.1): exact rationals + FD."""

    def __init__(self, theta: float = 0.5):
        self.theta = theta
        self.b = np.array([1.0, 0.0, 0.0])
        self.c = np.array([0.0, 0.0, 1.0])

    def _A(self, t: float) -> np.ndarray:
        return np.array([[2 + t, -1, 0],
                         [-1, 2 + t, -1],
                         [0, -1, 2 + t]], float)

    def _J(self, t: float) -> float:
        return float(self.c @ np.linalg.solve(self._A(t), self.b))

    def run(self, reporter: "Reporter") -> None:
        sys = LinearResponseSystem(self._A(self.theta), self.b, self.c)
        phi, psi, J = sys.phi, sys.adjoint, sys.response
        dJ = -psi @ phi                                # since dA/dtheta = I
        dJ_fd = (self._J(self.theta + 1e-6) - self._J(self.theta - 1e-6)) / 2e-6

        phi_paper = np.array([42, 20, 8]) / 85.0
        psi_paper = np.array([8, 20, 42]) / 85.0
        J_paper, dJ_paper = 8 / 85.0, -1072 / 7225.0

        reporter.info(f"phi        = {phi}   (paper {phi_paper})")
        reporter.info(f"psi        = {psi}   (paper {psi_paper})")
        reporter.info(f"J          = {J:.10f}  (paper {J_paper:.10f})")
        reporter.info(f"dJ/dtheta  = {dJ:.10f}  (paper {dJ_paper:.10f})")
        reporter.info(f"dJ/dtheta (finite diff) = {dJ_fd:.10f}")

        ok = (np.allclose(phi, phi_paper, atol=1e-12)
              and np.allclose(psi, psi_paper, atol=1e-12)
              and abs(J - J_paper) < 1e-12
              and abs(dJ - dJ_paper) < 1e-12
              and abs(dJ - dJ_fd) < 1e-7)
        reporter.check("3-cell: phi, psi, J, dJ/dtheta match exact rationals; adjoint==FD",
                       ok, f"|dJ_adj - dJ_fd| = {abs(dJ - dJ_fd):.2e}")


class DiffusionAdjointBenchmark(DiscreteAdjointBenchmark):
    """Diffusion-like discrete adjoint (Sec 10.1):
    A(alpha, beta) = diag(a0 + softplus(alpha)) + D^T diag(d0 + softplus(beta)) D.
    Verifies reverse-mode AD == closed-form adjoint == finite differences."""

    def __init__(self, N: int = 10, seed: int = 11, a0: float = 1.0, d0: float = 1.0):
        self.N, self.a0, self.d0 = N, a0, d0
        rng = np.random.default_rng(seed)
        self.alpha = rng.normal(size=N) * 0.3
        self.beta = rng.normal(size=N - 1) * 0.3
        self.q = rng.uniform(0.5, 1.5, size=N)
        self.w = rng.uniform(0.5, 1.5, size=N)
        self.D = np.zeros((N - 1, N))
        for e in range(N - 1):
            self.D[e, e], self.D[e, e + 1] = -1.0, 1.0

    def _build(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        a = self.a0 + softplus(alpha)
        d = self.d0 + softplus(beta)
        return np.diag(a) + self.D.T @ np.diag(d) @ self.D

    def _J(self, alpha: np.ndarray, beta: np.ndarray) -> float:
        return float(self.w @ np.linalg.solve(self._build(alpha, beta), self.q))

    def run(self, reporter: "Reporter") -> None:
        sys = LinearResponseSystem(self._build(self.alpha, self.beta), self.q, self.w)
        phi, lam, J = sys.phi, sys.adjoint, sys.response

        g_alpha = -lam * phi * sigmoid(self.alpha)                 # Eq. 49
        g_beta = -(self.D @ lam) * (self.D @ phi) * sigmoid(self.beta)  # Eq. 50

        N, D, q, w = self.N, self.D, self.q, self.w
        a0, d0 = self.a0, self.d0

        def J_ag(params):
            al, be = params[:N], params[N:]
            a = ag_softplus(al) + a0
            d = ag_softplus(be) + d0
            A = anp.diag(a) + D.T @ anp.diag(d) @ D
            return w @ anp.linalg.solve(A, q)

        g_ad = grad(J_ag)(np.concatenate([self.alpha, self.beta]))
        g_alpha_ad, g_beta_ad = g_ad[:N], g_ad[N:]

        # finite differences at a few coordinates
        h, fd_err = 1e-6, 0.0
        for i in range(0, N, 3):
            ap = self.alpha.copy(); ap[i] += h
            am = self.alpha.copy(); am[i] -= h
            fd_err = max(fd_err, abs((self._J(ap, self.beta) - self._J(am, self.beta)) / (2 * h) - g_alpha[i]))
        for e in range(0, N - 1, 3):
            bp = self.beta.copy(); bp[e] += h
            bm = self.beta.copy(); bm[e] -= h
            fd_err = max(fd_err, abs((self._J(self.alpha, bp) - self._J(self.alpha, bm)) / (2 * h) - g_beta[e]))

        e_alpha = float(np.max(np.abs(g_alpha - g_alpha_ad)))
        e_beta = float(np.max(np.abs(g_beta - g_beta_ad)))
        reporter.info(f"J = w^T phi = {J:.10f}")
        reporter.info(f"max|adjoint-AD|  absorption = {e_alpha:.2e}, diffusion = {e_beta:.2e}")
        reporter.info(f"max|adjoint-FD|  (sampled)  = {fd_err:.2e}")
        reporter.check("Reverse-mode AD == closed-form discrete adjoint (to roundoff)",
                       max(e_alpha, e_beta) < 1e-10, f"max abs diff = {max(e_alpha, e_beta):.2e}")
        reporter.check("Closed-form adjoint == finite differences (sampled)",
                       fd_err < 1e-6, f"max abs diff = {fd_err:.2e}")
        reporter.info("NOTE: the paper's specific value J=6.5122621426 depends on its exact")
        reporter.info("      (unstated) RNG protocol; the *equivalence* it validates")
        reporter.info("      reproduces exactly regardless of the particular random draw.")


# =========================================================================== #
#  LAYER 3 -- FITTING (competing sequence models for Sec 14)
# =========================================================================== #
class SequenceModel(ABC):
    """A one-step predictor X_{t+1} ~ F(X_t)."""

    name: str = "model"
    is_branching_valid: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SequenceModel": ...

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Deterministic mean map F(x) for a single population vector."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.apply(x) for x in X])

    def linear_operator(self) -> Optional[np.ndarray]:
        return None


class BSNNMeanModel(SequenceModel):
    """Branching-valid mean model: row-wise nonnegative least squares -> M >= 0."""

    name = "BSNN (NNLS)"
    is_branching_valid = True

    def __init__(self, n_types: int):
        self.m = n_types
        self.M = np.zeros((n_types, n_types))

    def fit(self, X, Y):
        for j in range(self.m):
            coef, _ = nnls(X, Y[:, j])
            self.M[j, :] = coef
        return self

    def apply(self, x):
        return self.M @ x

    def predict(self, X):
        return X @ self.M.T

    def linear_operator(self):
        return self.M


class LinearRNNModel(SequenceModel):
    """Unconstrained linear RNN with bias: least squares.  Bias breaks the
    homogeneity/additivity that a branching mean law must satisfy."""

    name = "linear RNN + bias"

    def __init__(self, n_types: int):
        self.m = n_types
        self.W = np.zeros((n_types, n_types))
        self.b = np.zeros(n_types)

    def fit(self, X, Y):
        Xaug = np.hstack([X, np.ones((X.shape[0], 1))])
        Wb, *_ = np.linalg.lstsq(Xaug, Y, rcond=None)
        self.W = Wb[:self.m, :].T
        self.b = Wb[self.m, :]
        return self

    def apply(self, x):
        return self.W @ x + self.b

    def predict(self, X):
        return X @ self.W.T + self.b

    def linear_operator(self):
        return self.W


class NonlinearRNNModel(SequenceModel):
    """Unconstrained nonlinear cell: fixed random tanh features + LS readout."""

    name = "nonlinear RNN (RF)"

    def __init__(self, n_types: int, hidden: int = 60, seed: int = 9, scale: float = 0.15):
        self.m = n_types
        rng = np.random.default_rng(seed)
        self.R = rng.normal(size=(hidden, n_types)) * scale
        self.c = rng.normal(size=hidden) * 0.1
        self.B = np.zeros((hidden, n_types))

    def _features(self, X):
        return np.tanh(X @ self.R.T + self.c)

    def fit(self, X, Y):
        self.B, *_ = np.linalg.lstsq(self._features(X), Y, rcond=None)
        return self

    def apply(self, x):
        return np.tanh(self.R @ x + self.c) @ self.B

    def predict(self, X):
        return self._features(X) @ self.B


class StructureDiagnostics:
    """Measure violations of the branching mean law:
    homogeneity F(a x) = a F(x) and additivity F(x + y) = F(x) + F(y)."""

    def __init__(self, seed: int = 77, n_probe: int = 300, scale: float = 30.0):
        self.rng = np.random.default_rng(seed)
        self.n_probe = n_probe
        self.scale = scale

    def evaluate(self, apply_fn: Callable[[np.ndarray], np.ndarray], m: int):
        he = ae = 0.0
        for _ in range(self.n_probe):
            x = self.rng.uniform(0, self.scale, size=m)
            y = self.rng.uniform(0, self.scale, size=m)
            a = self.rng.uniform(0.2, 3.0)
            fx, fy = apply_fn(x), apply_fn(y)
            he = max(he, np.linalg.norm(apply_fn(a * x) - a * fx) / (np.linalg.norm(a * fx) + 1e-9))
            ae = max(ae, np.linalg.norm(apply_fn(x + y) - fx - fy) / (np.linalg.norm(fx + fy) + 1e-9))
        return he, ae


# =========================================================================== #
#  LAYER 4 -- VALIDATION FRAMEWORK
# =========================================================================== #
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class Reporter:
    """Console reporter: sections, indented info lines, PASS/FAIL checks, summary."""

    def __init__(self):
        self.results: List[CheckResult] = []

    def section(self, title: str) -> None:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    def info(self, message: str) -> None:
        print("      " + message)

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, bool(passed), detail))
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))

    def summary(self) -> bool:
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for r in self.results:
            print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name}")
        n_pass = sum(r.passed for r in self.results)
        print("-" * 78)
        print(f"  {n_pass}/{len(self.results)} checks passed")
        return n_pass == len(self.results)


class Validator(ABC):
    """Base for a group of related checks (one theorem area)."""

    title: str = "validator"

    def __init__(self, reporter: Reporter, figure_dir: str = "."):
        self.reporter = reporter
        self.figure_dir = figure_dir

    def check(self, name, passed, detail=""):
        self.reporter.check(name, passed, detail)

    def info(self, message):
        self.reporter.info(message)

    def figure_path(self, name: str) -> str:
        os.makedirs(self.figure_dir, exist_ok=True)
        return os.path.join(self.figure_dir, name)

    @abstractmethod
    def run(self) -> None: ...


# --------------------------------------------------------------------------- #
#  T1 -- Branching factorization (Thm 4.3)
# --------------------------------------------------------------------------- #
class FactorizationValidator(Validator):
    title = "T1  Branching factorization  G_{x+y}(z) = G_x(z) G_y(z)   (Thm 4.3)"

    def run(self):
        # (a) analytic, Poisson head (exact)
        M = np.array([[0.6, 0.2, 0.1],
                      [0.3, 0.5, 0.2],
                      [0.1, 0.3, 0.5]])
        bp = BranchingProcess(PoissonOffspring(M))
        rng = np.random.default_rng(1)
        max_analytic = 0.0
        for _ in range(200):
            x = rng.integers(0, 5, size=3)
            y = rng.integers(0, 5, size=3)
            z = rng.uniform(0, 1, size=3)
            lhs = bp.transition_pgf(x + y, z)
            rhs = bp.transition_pgf(x, z) * bp.transition_pgf(y, z)
            max_analytic = max(max_analytic, abs(lhs - rhs))
        self.check("Poisson-head factorization (analytic PGF)",
                   max_analytic < 1e-12, f"max|LHS-RHS| = {max_analytic:.2e}")

        # (b) Monte-Carlo, correlated non-Poisson head
        mult_probs = [[0.02, 0.10, 0.30, 0.38, 0.15, 0.05],
                      [0.05, 0.15, 0.35, 0.30, 0.12, 0.03],
                      [0.03, 0.12, 0.33, 0.34, 0.14, 0.04]]
        type_probs = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
        fb = BranchingProcess(MultinomialFissionOffspring(mult_probs, type_probs), seed=7)

        x, y, N = np.array([2, 1, 1]), np.array([1, 2, 0]), 400_000
        Ox = fb.replicate_generation(x, N).astype(float)
        Oy = fb.replicate_generation(y, N).astype(float)
        Oxy = fb.replicate_generation(x + y, N).astype(float)

        z_tests = [np.array([0.3, 0.5, 0.7]), np.array([0.9, 0.8, 0.95]),
                   np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.9, 0.4])]
        worst = 0.0
        self.info("z-point                G_{x+y}      G_x*G_y      |diff|")
        for z in z_tests:
            Gx = float(np.mean(np.prod(z ** Ox, axis=1)))
            Gy = float(np.mean(np.prod(z ** Oy, axis=1)))
            Gxy = float(np.mean(np.prod(z ** Oxy, axis=1)))
            err = abs(Gxy - Gx * Gy)
            worst = max(worst, err)
            self.info(f"{np.array2string(z, precision=2):<22} {Gxy:.6f}    {Gx*Gy:.6f}    {err:.2e}")
        mc_tol = 6.0 / np.sqrt(N)
        self.check("Non-Poisson factorization (Monte-Carlo, correlated offspring)",
                   worst < mc_tol, f"max|diff| = {worst:.2e}  (MC tol {mc_tol:.2e})")


# --------------------------------------------------------------------------- #
#  T2 -- Mean & covariance propagation (Thm 6.1)
# --------------------------------------------------------------------------- #
class MomentValidator(Validator):
    title = "T2  Mean & covariance propagation   (Thm 6.1)"

    def run(self):
        B = np.array([[0.55, 0.10, 0.05],
                      [0.25, 0.60, 0.10],
                      [0.05, 0.20, 0.50]])
        op0 = MeanFieldOperator(B)
        M = 1.05 * B / op0.spectral_radius
        law = PoissonOffspring(M)
        bp = BranchingProcess(law, seed=123)

        x0 = np.array([20.0, 15.0, 10.0])
        n_gen, n_traj = 8, 400_000
        traj = bp.simulate(x0, n_gen, n_traj)

        op = MeanFieldOperator(M)
        means, covs = op.propagate_covariance(x0, law.covariances(), n_gen)

        mean_err, cov_err = [], []
        self.info("gen   ||E-emp||/||E||   ||Cov-emp||/||Cov||")
        for n in range(n_gen + 1):
            emp_mean = traj[n].mean(axis=0)
            emp_cov = np.cov(traj[n].T)
            me = np.linalg.norm(emp_mean - means[n]) / (np.linalg.norm(means[n]) + 1e-12)
            ce = 0.0 if n == 0 else np.linalg.norm(emp_cov - covs[n]) / (np.linalg.norm(covs[n]) + 1e-12)
            mean_err.append(me); cov_err.append(ce)
            self.info(f"{n:>3}      {me:.3e}          {ce:.3e}")

        self.check("Mean recursion  x_{n+1}=M x_n",
                   max(mean_err) < 5e-3, f"max rel. error = {max(mean_err):.2e}")
        self.check("Covariance recursion  Sigma_{n+1}=M Sigma_n M^T + sum x_i V_i",
                   max(cov_err) < 3e-2, f"max rel. error = {max(cov_err):.2e}")

        self._plot(traj, means, covs, n_gen)

    def _plot(self, traj, means, covs, n_gen):
        gens = np.arange(n_gen + 1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for j in range(3):
            ax[0].plot(gens, [traj[n][:, j].mean() for n in gens], "o", ms=4, label=f"type {j+1} (sim)")
            ax[0].plot(gens, [means[n][j] for n in gens], "-", lw=1.5, label=f"type {j+1} (theory)")
        ax[0].set_title("Mean population per type")
        ax[0].set_xlabel("generation"); ax[0].set_ylabel("E[X]"); ax[0].legend(fontsize=7)
        for j in range(3):
            ax[1].plot(gens[1:], [np.cov(traj[n].T)[j, j] for n in gens[1:]], "o", ms=4)
            ax[1].plot(gens[1:], [covs[n][j, j] for n in gens[1:]], "-", lw=1.5)
        ax[1].set_title("Variance per type (dots=sim, lines=theory)")
        ax[1].set_xlabel("generation"); ax[1].set_ylabel("Var[X]")
        fig.tight_layout(); fig.savefig(self.figure_path("fig_T2_moments.png"), dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
#  T3 -- Spectral-radius criticality (Thm 7.1)
# --------------------------------------------------------------------------- #
class CriticalityValidator(Validator):
    title = "T3  Spectral-radius criticality  ||M^n x_0||^{1/n} -> rho(M)   (Thm 7.1)"

    def run(self):
        B = np.array([[0.55, 0.10, 0.05],
                      [0.25, 0.60, 0.10],
                      [0.05, 0.20, 0.50]])
        B = B / MeanFieldOperator(B).spectral_radius       # rho(B) = 1
        x0 = np.array([1.0, 1.0, 1.0])
        regimes = {"subcritical": 0.8, "critical": 1.0, "supercritical": 1.25}
        n_max = 300

        fig, ax = plt.subplots(figsize=(7, 4.3))
        ok_all, rows = True, []
        for name, scale in regimes.items():
            op = MeanFieldOperator(scale * B)
            rho = op.spectral_radius
            pit = op.power_iteration(x0, n_max)
            ok = abs(pit["ratio"] - rho) < 1e-6
            ok_all &= ok
            rows.append((name, rho, pit["ratio"], pit["nth_root"]))

            if name == "critical":
                k, v, u = op.perron_vectors()
                limit_theory = (v * (u @ x0)) / (u @ v)
                x_big = np.linalg.matrix_power(op.M, 400) @ x0
                lim_err = np.linalg.norm(x_big - limit_theory) / np.linalg.norm(limit_theory)
                self.check("Critical-regime limit vector (v u^T x0)/(u^T v)",
                           lim_err < 1e-6, f"rel. error = {lim_err:.2e}")

            cum = np.exp(pit["lognorms"]) * np.linalg.norm(x0, 1)
            ax.semilogy(range(n_max + 1), cum, label=f"{name} (rho={rho:.3f})")

        ax.set_title(r"$\|M^n x_0\|_1$  vs generation (log scale)")
        ax.set_xlabel("generation n"); ax.set_ylabel(r"$\|x_n\|_1$"); ax.legend()
        fig.tight_layout(); fig.savefig(self.figure_path("fig_T3_criticality.png"), dpi=110); plt.close(fig)

        self.info("regime          rho(M)     ratio(->rho fast)   ||x_n||^(1/n) (slow)")
        for name, rho, ratio, nth in rows:
            self.info(f"{name:<14}  {rho:.6f}   {ratio:.6f}            {nth:.6f}")
        self.check("Growth rate = rho(M) (per-step ratio) in all three regimes",
                   ok_all, "sub/critical/super all match rho to <1e-6")


# --------------------------------------------------------------------------- #
#  T4 -- Extinction (Thm 7.3 / App A)
# --------------------------------------------------------------------------- #
class ExtinctionValidator(Validator):
    title = "T4  Extinction = minimal fixed point of q=g(q); rho<=>1 dichotomy   (Thm 7.3 / App A)"

    def run(self):
        # ---- one-type Galton-Watson (Poisson offspring) ----
        def one_type_q(mu):
            solver = ExtinctionSolver(lambda q: np.exp(mu * (q - 1.0)), 1)
            return float(solver.solve()[0])

        for mu in [0.8, 1.0, 1.5, 2.4]:
            q_fp = one_type_q(mu)
            bp = BranchingProcess(PoissonOffspring(np.array([[mu]])))
            q_emp = bp.extinction_frequency([1.0], horizon=250, n_traj=60_000,
                                            seed=int(1000 * mu))
            tol = 0.02 if mu > 1 else 0.06
            self.check(f"1-type extinction, mu={mu}", abs(q_fp - q_emp) < tol,
                       f"q_fixedpoint={q_fp:.4f}  q_empirical={q_emp:.4f}")

        dich_ok = (one_type_q(0.8) > 0.999 and one_type_q(1.0) > 0.999 and one_type_q(2.4) < 1.0)
        self.check("1-type dichotomy  mu<=1 => q=1,  mu>1 => q<1", dich_ok)

        # ---- multitype (Poisson head) ----
        Bsub = np.array([[0.4, 0.1, 0.05],
                         [0.2, 0.4, 0.1],
                         [0.05, 0.15, 0.35]])
        Bsup = 1.6 * Bsub / MeanFieldOperator(Bsub).spectral_radius
        self._multitype(Bsub, "subcritical", expect_extinct=True)
        self._multitype(Bsup, "supercritical", expect_extinct=False)

    def _multitype(self, M, label, expect_extinct):
        law = PoissonOffspring(M)
        solver = ExtinctionSolver(law.one_parent_pgf, M.shape[0])
        q_fp = solver.solve()
        rho = MeanFieldOperator(M).spectral_radius

        bp = BranchingProcess(law)
        x0 = np.zeros(M.shape[0]); x0[0] = 1.0
        q_emp0 = bp.extinction_frequency(x0, horizon=250, n_traj=40_000, seed=42)

        tol = 0.03 if rho > 1 else 0.06
        self.info(f"{label}:  rho={rho:.4f}  q_fp={np.array2string(q_fp, precision=4)}"
                  f"  q_emp(type0)={q_emp0:.4f}")
        self.check(f"Multitype extinction ({label})", abs(q_fp[0] - q_emp0) < tol,
                   f"q_fp[0]={q_fp[0]:.4f} vs emp {q_emp0:.4f}")
        if expect_extinct:
            self.check(f"  dichotomy: rho<=1 => q=1 ({label})", bool(np.all(q_fp > 0.999)))
        else:
            self.check(f"  dichotomy: rho>1 => q<1 ({label})", bool(np.all(q_fp < 1.0 - 1e-6)))


# --------------------------------------------------------------------------- #
#  T5 + T6 -- Adjoint / sensitivity (Thm 10.1, 10.2)
# --------------------------------------------------------------------------- #
class AdjointSensitivityValidator(Validator):
    title = "T5/T6  Backprop==adjoint (Thm 10.1) & eigenvalue sensitivity (Thm 10.2)"

    def run(self):
        # ---- T5: backprop through the mean recurrence == adjoint ----
        rng = np.random.default_rng(3)
        m, N = 4, 6
        A = rng.normal(size=(m, m)) * 0.5
        x0 = rng.uniform(0.5, 1.5, size=m)
        rec = MeanRecurrenceAdjoint(A, x0, N)

        G_adj = rec.adjoint_gradient()
        G_ad = rec.autograd_gradient()
        G_fd = rec.finite_difference_gradient()

        scale = np.max(np.abs(G_adj)) + 1e-12
        e_adj_ad = float(np.max(np.abs(G_adj - G_ad)))
        e_adj_fd = float(np.max(np.abs(G_adj - G_fd)) / scale)   # relative (loss ~1e4)
        self.check("adjoint  ==  reverse-mode AD (autograd)",
                   e_adj_ad < 1e-9, f"max abs diff = {e_adj_ad:.2e}")
        self.check("adjoint  ==  finite differences",
                   e_adj_fd < 1e-5, f"max REL diff = {e_adj_fd:.2e} (loss scale ~{scale:.1e})")

        # ---- T6: dominant-eigenvalue sensitivity ----
        rng2 = np.random.default_rng(5)
        As = rng2.normal(size=(m, m)) * 0.4
        eig = EigenvalueSensitivity(As)
        err = float(np.max(np.abs(eig.formula_gradient() - eig.finite_difference_gradient())))
        self.check("d rho / d theta   formula vs finite differences",
                   err < 1e-6, f"max abs diff = {err:.2e}")


# --------------------------------------------------------------------------- #
#  T7 -- Discrete-adjoint benchmarks (Sec 10.1 / 11.1)
# --------------------------------------------------------------------------- #
class BenchmarkValidator(Validator):
    title = "T7  Discrete-adjoint benchmarks (Sec 11.1 exact 3-cell; Sec 10.1 diffusion)"

    def run(self):
        self.info("-- 3-cell benchmark (Sec 11.1) --")
        ThreeCellBenchmark(theta=0.5).run(self.reporter)
        self.info("-- diffusion-like benchmark (Sec 10.1) --")
        DiffusionAdjointBenchmark(N=10, seed=11).run(self.reporter)


# --------------------------------------------------------------------------- #
#  T8 -- BSNN vs RNNs + structure diagnostics (Sec 14)
# --------------------------------------------------------------------------- #
class ModelComparisonValidator(Validator):
    title = "T8  BSNN vs linear RNN vs nonlinear RNN + structure diagnostics   (Sec 14)"

    def run(self):
        B = np.array([[0.55, 0.10, 0.05],
                      [0.25, 0.60, 0.10],
                      [0.05, 0.20, 0.50]])
        M_star = 1.08 * B / MeanFieldOperator(B).spectral_radius
        rho_star = MeanFieldOperator(M_star).spectral_radius
        m, n_gen = 3, 6

        def make(n_traj, seed):
            bp = BranchingProcess(PoissonOffspring(M_star), seed=seed)
            return bp.simulate(np.array([30.0, 20.0, 10.0]), n_gen, n_traj)

        def pairs(traj):
            return traj[:-1].reshape(-1, m), traj[1:].reshape(-1, m)

        Xtr, Ytr = pairs(make(30, seed=1))
        Xte, Yte = pairs(make(300, seed=2))

        models: List[SequenceModel] = [
            BSNNMeanModel(m).fit(Xtr, Ytr),
            LinearRNNModel(m).fit(Xtr, Ytr),
            NonlinearRNNModel(m).fit(Xtr, Ytr),
        ]

        diag = StructureDiagnostics()
        rmse = lambda P: float(np.sqrt(np.mean((P - Yte) ** 2)))

        stats = {}
        for mdl in models:
            he, ae = diag.evaluate(mdl.apply, m)
            W = mdl.linear_operator()
            drho = abs(MeanFieldOperator(W).spectral_radius - rho_star) if W is not None else None
            relM = np.linalg.norm(W - M_star) / np.linalg.norm(M_star) if W is not None else None
            negm = float(np.sum(np.clip(-W, 0, None))) if W is not None else None
            stats[mdl.name] = dict(rmse=rmse(mdl.predict(Xte)), drho=drho, relM=relM,
                                   he=he, ae=ae, negm=negm, valid=mdl.is_branching_valid)

        # extinction (BSNN only)
        M_bsnn = models[0].linear_operator()
        q_bsnn = ExtinctionSolver(PoissonOffspring(M_bsnn).one_parent_pgf, m).solve()
        q_true = ExtinctionSolver(PoissonOffspring(M_star).one_parent_pgf, m).solve()
        ext_err_b = float(np.linalg.norm(q_bsnn - q_true))

        # report table
        self.info(f"rho(M_star) = {rho_star:.4f}")
        self.info(f"{'model':<26}{'RMSE':>9}{'|d rho|':>9}{'relM':>9}"
                  f"{'homog':>9}{'addit':>9}{'negmass':>9}  valid?")
        for mdl in models:
            s = stats[mdl.name]
            fmt = lambda v: f"{v:>9.3f}" if isinstance(v, float) else f"{'-':>9}"
            self.info(f"{mdl.name:<26}{s['rmse']:>9.3f}{fmt(s['drho'])}{fmt(s['relM'])}"
                      f"{s['he']:>9.3f}{s['ae']:>9.3f}{fmt(s['negm'])}"
                      f"   {'yes' if s['valid'] else 'no'}")
        self.info(f"BSNN extinction-vector error vs truth: {ext_err_b:.4f}"
                  f"   (q_true={np.array2string(q_true, precision=3)})")

        sb, sl, sn = stats["BSNN (NNLS)"], stats["linear RNN + bias"], stats["nonlinear RNN (RF)"]
        self.check("BSNN preserves homogeneity & additivity exactly",
                   sb["he"] < 1e-9 and sb["ae"] < 1e-9, f"homog={sb['he']:.1e}, addit={sb['ae']:.1e}")
        self.check("linear RNN + bias violates homogeneity/additivity (as claimed)",
                   sl["he"] > 1e-3 or sl["ae"] > 1e-3, f"homog={sl['he']:.3f}, addit={sl['ae']:.3f}")
        self.check("nonlinear RNN violates branching structure strongly",
                   sn["he"] > sl["he"] and sn["ae"] > sl["ae"], f"homog={sn['he']:.3f}, addit={sn['ae']:.3f}")
        self.check("BSNN and linear-RNN one-step RMSE are comparable",
                   abs(sb["rmse"] - sl["rmse"]) / sb["rmse"] < 0.15,
                   f"RMSE BSNN={sb['rmse']:.3f}, linRNN={sl['rmse']:.3f}")

        self._plot(models, stats, Xte, Yte)

    def _plot(self, models, stats, Xte, Yte):
        names = [m.name.split()[0] for m in models]
        rmse = lambda P: float(np.sqrt(np.mean((P - Yte) ** 2)))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(names, [rmse(m.predict(Xte)) for m in models], color=["#2a7", "#37a", "#a53"])
        ax[0].set_title("One-step test RMSE"); ax[0].set_ylabel("RMSE")
        ax[1].bar(names, [stats[m.name]["he"] for m in models], color=["#2a7", "#37a", "#a53"])
        ax[1].set_title("Homogeneity violation  ||F(ax)-aF(x)||"); ax[1].set_ylabel("relative error")
        fig.tight_layout(); fig.savefig(self.figure_path("fig_T8_comparison.png"), dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
#  T9 -- Finite-sample Hoeffding bound (Thm 9.1)
# --------------------------------------------------------------------------- #
class FiniteSampleValidator(Validator):
    title = "T9  Finite-sample Hoeffding bound for M-hat: empirical coverage   (Thm 9.1)"

    def run(self):
        m, B, delta, Nmin = 3, 4, 0.05, 800
        P = np.random.default_rng(0).uniform(0.05, 0.4, size=(m, m))   # P[j,i]=p_ji
        law = BinomialOffspring(B, P)
        M_true = law.mean_matrix()
        bound = B * np.sqrt(np.log(2 * m * m / delta) / (2 * Nmin))

        n_trials, exceed, max_errs = 3000, 0, []
        for t in range(n_trials):
            rng = np.random.default_rng(1000 + t)
            Mhat = np.zeros((m, m))
            for i in range(m):
                Mhat[:, i] = law.offspring_from(i, Nmin, rng).mean(axis=0)
            err = float(np.max(np.abs(Mhat - M_true)))
            max_errs.append(err)
            exceed += (err > bound)
        frac = exceed / n_trials

        self.info(f"Hoeffding bound (delta={delta}) = {bound:.4f}")
        self.info(f"empirical max-error: mean={np.mean(max_errs):.4f}, "
                  f"95th pct={np.percentile(max_errs, 95):.4f}, max={np.max(max_errs):.4f}")
        self.info(f"fraction of {n_trials} trials exceeding the bound = {frac:.4f}"
                  f"  (guarantee: <= {delta})")
        self.check("Hoeffding coverage: P(max error > bound) <= delta",
                   frac <= delta, f"exceed frac = {frac:.4f} <= {delta}")


# =========================================================================== #
#  ORCHESTRATOR
# =========================================================================== #
class ValidationSuite:
    """Registers and runs all validators, then prints a summary."""

    def __init__(self, figure_dir: str = "."):
        self.reporter = Reporter()
        self.validators: List[Validator] = [
            FactorizationValidator(self.reporter, figure_dir),
            MomentValidator(self.reporter, figure_dir),
            CriticalityValidator(self.reporter, figure_dir),
            ExtinctionValidator(self.reporter, figure_dir),
            AdjointSensitivityValidator(self.reporter, figure_dir),
            BenchmarkValidator(self.reporter, figure_dir),
            ModelComparisonValidator(self.reporter, figure_dir),
            FiniteSampleValidator(self.reporter, figure_dir),
        ]
        self.figure_dir = figure_dir

    def run(self) -> bool:
        print("#" * 78)
        print("#  BSNN manuscript validation suite (object-oriented)")
        print("#" * 78)
        for v in self.validators:
            self.reporter.section(v.title)
            v.run()
        ok = self.reporter.summary()
        print(f"  Figures written to: {os.path.abspath(self.figure_dir)}")
        return ok


def main() -> bool:
    return ValidationSuite(figure_dir=".").run()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
