import cvxpy as cp
import numpy as np
import torch
from typing import Tuple

Tensor = torch.Tensor

class SparseCovariance:
    """Solve eq. (17) from the paper:
    'Structured Nyquist Correlation Reconstruction for DOA Estimation With Sparse Arrays'
     with CVXPY (nuclear-norm relaxation)."""

    def __init__(self, system_model):
        self.sys   = system_model
        self.phi   = self._build_phi().cpu().numpy()        # Φ : numpy for CVXPY
        self.phi_H = self.phi.conj().T
        self.U     = self.phi.shape[1]                      # |U|
        self.S     = self.phi.shape[0]                      # |S|

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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

        # Toeplitz constraint → equality of diagonals
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_phi(self) -> Tensor:
        s = torch.tensor(self.sys.array,          dtype=torch.long)
        v = torch.tensor(self.sys.virtual_array,  dtype=torch.long)
        phi = torch.zeros(s.numel(), v.numel())
        for i, p in enumerate(s):
            idx = (v == p).nonzero(as_tuple=True)[0]
            phi[i, idx] = 1.
        return phi

