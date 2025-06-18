import cvxpy as cp
import numpy as np
import torch
from torch import Tensor
from typing import Tuple



class SparseCovarianceCVXPY:
    """Solve eq. (17) from the paper:
    'Structured Nyquist Correlation Reconstruction for DOA Estimation With Sparse Arrays'
     with CVXPY (nuclear-norm relaxation)."""

    def __init__(self, system_model):
        self.sys   = system_model
        self.phi   = self._build_phi().cpu().numpy()        # Φ : numpy for CVXPY
        self.phi_H = self.phi.conj().T
        self.U     = self.phi.shape[1]                      # |U|
        self.S     = self.phi.shape[0]                      # |S|


    def __call__(self,
                 Rx: Tensor,
                 mu: float = 2.5e-3,
                 solver: str = "SCS") -> Tensor:
        """
        Parameters
        ----------
        Rx : Tensor (|S|,|S|)
             Sample covariance matrix.
        mu : float
            nuclear-norm weight μ in (17)
        solver : str
            CVXPY solver name (“SCS”, “MOSEK”, …)

        Returns
        -------
        R_tilde : torch.Tensor (|U|, |U|)
            Toeplitz-PSD covariance fitted by nuclear-norm minimisation
        """
        R_xx = Rx.cpu().numpy()      # (|S|,|S|)

        # CVXPY variable (complex Hermitian)
        R = cp.Variable((self.U, self.U), hermitian=True)

        # Toeplitz constraint: equality of diagonals
        toeplitz_constraints = []
        for k in range(-self.U + 1, self.U):
            diag = cp.diag(R, k)
            toeplitz_constraints.append(diag == cp.mean(diag))

        # PSD constraint
        constraints = toeplitz_constraints + [R >> 0]

        # Objective
        fit = cp.norm(self.phi @ R @ self.phi_H - R_xx, p='fro')
        obj = cp.Minimize(fit + mu * cp.normNuc(R))

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=solver, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed: {prob.status}")

        return torch.tensor(R.value, dtype=torch.complex64, device=Rx.device)


    def _build_phi(self) -> Tensor:
        s = torch.tensor(self.sys.array,          dtype=torch.long)
        v = torch.tensor(self.sys.virtual_array,  dtype=torch.long)
        phi = torch.zeros(s.numel(), v.numel())
        for i, p in enumerate(s):
            idx = (v == p).nonzero(as_tuple=True)[0]
            phi[i, idx] = 1.
        return phi


class SparseCovarianceADMM:
    r"""
    Batched ADMM solver for the nuclear-norm covariance completion

        min ‖ΦRΦᴴ − Rₓₓ‖_F² + μ‖R‖_*
        s.t.  R  Hermitian–Toeplitz – PSD.

    Parameters
    ----------
    system_model  :  object
        Must expose
          • array           (physical sensor indices)
          • virtual_array   (contiguous co-array indices)
    """

    # ───────────────────────── initialisation ────────────────────────
    def __init__(self, system_model):
        self.sys   = system_model
        self.phi   = self._build_phi()                     # (|S|,|U|)
        self.phi_H = self.phi.t()
        self.U     = self.phi.shape[1]                     # |U|

        # mask diagonal  (P = Φᴴ Φ) → 1-D of length |U|²
        m = (self.phi_H @ self.phi).diag()  # (|U|,)
        self.P = (m[:, None] * m[None, :]).flatten()  # (|U|²,)

    # ───────────────────────── public API ────────────────────────────
    @torch.no_grad()
    def __call__(self,
                 Rx: Tensor,           # (B,|S|,|S|)
                 mu: float = 2.5e-3,
                 rho: float = 2,
                 max_iter: int = 400,
                 tol_primal: float = 1e-11,
                 tol_dual: float = 1e-11,
                 verbose: bool = True) -> Tensor:
        """
        Returns
        -------
        R_tilde : (B,|U|,|U|) complex tensor
        """
        B, S, _ = Rx.shape
        U = self.U
        dev, dtype = Rx.device, Rx.dtype

        phi   = self.phi.to(dev, dtype)
        phi_H = self.phi_H.to(dev, dtype)

        # ------------------------------------------------------------
        # measured part  Φᴴ Rₓₓ Φ  →  (B,|U|,|U|)
        # ------------------------------------------------------------
        meas = phi_H @ Rx @ phi
        vec_meas = meas.reshape(B, -1)   # (B, |U|²)

        # ------------------------------------------------------------
        # diagonal coefficients  (mask + 2ρI)⁻¹  (1-D then broadcast)
        # ------------------------------------------------------------
        inv_coeff = 1.0 / (self.P.to(dev, dtype) + 2.0 * rho)
        inv_coeff = inv_coeff.expand(B, -1)                # (B, |U|²)

        # ------------------------------------------------------------
        # initial variables  (B,|U|,|U|)
        # ------------------------------------------------------------
        R = self._hermitian_proj(meas)
        S = R.clone()
        T = self._toeplitz_proj(R)
        Udual = torch.zeros_like(R)
        Vdual = torch.zeros_like(R)

        S_prev, T_prev = S.clone(), T.clone()

        # ------------------------------------------------------------
        # ADMM iterations
        # ------------------------------------------------------------

        for k in range(max_iter):
            # R-update  (diagonal solve, batched)
            rhs = vec_meas + rho * (S - Udual + T - Vdual).reshape(B, -1)
            vec_R = inv_coeff * rhs
            R = vec_R.view(B, U, U)

            # S-update  (SVT)
            Z = R + Udual
            S = self._svt(Z, mu / rho)

            # T-update  (Herm-Toeplitz-PSD)
            W = R + Vdual
            T = self._psd_proj(self._toeplitz_proj(self._hermitian_proj(W)))

            # dual ascent
            Udual += R - S
            Vdual += R - T

            # convergence criteria (batch max)
            r_norm = torch.max(
                (R - S).flatten(1).norm(dim=1),
                (R - T).flatten(1).norm(dim=1)
            ).max()              # global primal residual

            s_norm = rho * torch.max(
                (S - S_prev).flatten(1).norm(dim=1),
                (T - T_prev).flatten(1).norm(dim=1)
            ).max()              # global dual residual

            if verbose and (k % 25 == 0):
                print(f"iter {k:4d} | primal {r_norm:.3e} | dual {s_norm:.3e}")

            if r_norm < tol_primal and s_norm < tol_dual:
                break

            S_prev.copy_(S)
            T_prev.copy_(T)

        return T

    # ─────────────────── helper: build Φ (|S|×|U|) ──────────────────
    def _build_phi(self) -> Tensor:
        s = torch.tensor(self.sys.array,         dtype=torch.int64)
        v = torch.tensor(self.sys.virtual_array, dtype=torch.int64)
        phi = torch.zeros(s.numel(), v.numel())
        for i, p in enumerate(s):
            phi[i, (v == p).nonzero(as_tuple=True)[0]] = 1.0
        return phi

    # ───────────── projections & prox (batch-compatible) ─────────────
    @staticmethod
    def _hermitian_proj(X: Tensor) -> Tensor:
        return 0.5 * (X + X.conj().transpose(-2, -1))

    @staticmethod
    def _toeplitz_proj(H: Tensor) -> Tensor:
        B, L, _ = H.shape
        T = H.clone()
        for d in range(-L + 1, L):
            diag = H.diagonal(d, dim1=-2, dim2=-1)        # (B, L-|d|)
            mean = diag.mean(dim=-1, keepdim=True)
            T.diagonal(d, dim1=-2, dim2=-1).copy_(mean.expand_as(diag))
        return T

    @staticmethod
    def _psd_proj(T: Tensor) -> Tensor:
        lam, U = torch.linalg.eigh(T)
        lam.clamp_(min=0.0)
        return (U * lam.unsqueeze(-2)) @ U.transpose(-2, -1).conj()

    @staticmethod
    def _svt(Z: Tensor, tau: float) -> Tensor:
        U, s, Vh = torch.linalg.svd(Z)
        s = torch.clamp(s - tau, min=0.0)
        return (U * s.unsqueeze(-2)) @ Vh
