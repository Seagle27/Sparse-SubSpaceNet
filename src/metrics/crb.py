import torch
from src.utils import build_phi



# ---------- vec/unvec helpers (column-major) ----------
def _vecF(M: torch.Tensor) -> torch.Tensor:
    return M.transpose(-2, -1).reshape(-1)


def _vecF_batch(M: torch.Tensor) -> torch.Tensor:
    B, n, _ = M.shape
    return M.transpose(1, 2).reshape(B, n * n).transpose(0, 1)


def _unvecF_batch(V: torch.Tensor, n: int) -> torch.Tensor:
    B = V.shape[1]
    return V.transpose(0, 1).view(B, n, n).transpose(1, 2)


# ---------- steering & derivative on filled grid ----------
def _steering(U_pos: torch.Tensor, thetas: torch.Tensor, phase_scale: float) -> torch.Tensor:
    # a_mk = exp(-j * phase_scale * u_m * sin(theta_k))
    v = U_pos.to(dtype=torch.float64).reshape(-1, 1)
    th = thetas.to(dtype=torch.float64).reshape(1, -1)
    phase = -phase_scale * v * torch.sin(th)
    return torch.exp(1j * phase).to(torch.complex128)


def _steering_deriv(U_pos: torch.Tensor, theta_k: torch.Tensor, phase_scale: float) -> tuple[
    torch.Tensor, torch.Tensor]:
    v = U_pos.to(dtype=torch.float64)
    th = theta_k.to(dtype=torch.float64)
    a = torch.exp(1j * (-phase_scale * v * torch.sin(th)))
    da = (1j * -phase_scale * v * torch.cos(th)) * a
    return a.to(torch.complex128), da.to(torch.complex128)

# ---------- main: SNCR CRB ----------
def calculate_sncr_crb(S_positions: torch.Tensor,
                       thetas_deg: torch.Tensor,
                       snr_db: float,
                       L_snapshots: int,
                       d: float = 0.5):
    """
    SNCR stochastic CRB (per-angle variances in rad^2) for a sparse array.

    S_positions : 1D tensor of sparse sensor indices (integers, in units of d)
    thetas_deg  : 1D tensor of K DOAs [deg]
    snr_db      : per-sensor SNR per source [dB]
    L_snapshots : number of snapshots **(not ADMM iters)**
    d           : inter-element spacing in wavelengths; phase_scale = 2π d
    """
    device = S_positions.device if isinstance(S_positions, torch.Tensor) else ("cuda" if torch.cuda.is_available() else "cpu")
    thetas = torch.deg2rad(thetas_deg).to(device=device, dtype=torch.float64)
    S_pos  = S_positions.to(device=device, dtype=torch.long)

    # ---- 1) Phi defines the filled grid size U ----
    # Use YOUR existing builder; it must return Φ ∈ R^{|S|×U}
    Phi = build_phi(S_pos).to(device)
    U   = Phi.shape[1]
    U_pos = torch.arange(0, U, device=device, dtype=torch.float64)

    # ---- 2) Steering on the filled grid with your phase scale ----
    phase_scale = 2.0 * torch.pi * d            # matches exp(-j 2π d * pos * sinθ)
    A = _steering(U_pos, thetas, float(phase_scale))

    # ---- 3) Map SNR -> p using actual column powers ----
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2  = torch.tensor(1.0, device=device, dtype=torch.float64)
    col_power = (A.abs()**2).mean(dim=0).real.clamp_min(1e-12)   # c_k
    p = (snr_lin * sigma2 / col_power).to(torch.float64)         # (K,)

    # ---- 4) Build Jacobian J = [D_theta, D_p, d_sigma] ----
    K  = thetas.numel()
    D_p_cols, D_th_cols = [], []
    for k in range(K):
        a_k, da_k = _steering_deriv(U_pos, thetas[k], float(phase_scale))
        D_p_cols.append(torch.kron(a_k.conj(), a_k))
        D_th_cols.append(p[k] * (torch.kron(da_k.conj(), a_k) + torch.kron(a_k.conj(), da_k)))
    D_p  = torch.stack(D_p_cols,  dim=1)                                     # (U^2, K)
    D_th = torch.stack(D_th_cols, dim=1)                                     # (U^2, K)
    d_sigma = _vecF(torch.eye(U, dtype=torch.complex128, device=device))     # (U^2,)
    J = torch.cat([D_th, D_p, d_sigma[:, None]], dim=1)                      # (U^2, 2K+1)

    # ---- 5) Covariances on filled and sparse arrays ----
    Pdiag = torch.diag(p.to(torch.complex128))
    R_yy  = A @ Pdiag @ A.conj().T + sigma2.to(torch.complex128) * torch.eye(U, dtype=torch.complex128, device=device)
    R_xx  = Phi.to(torch.complex128) @ R_yy @ Phi.to(torch.complex128).conj().T

    # ---- 6) Left/right weights via solves (no explicit inverse) ----
    Y_right = torch.linalg.solve(R_xx, Phi.to(R_xx.dtype))                   # (|S|, |U|)
    M_right = Phi.transpose(0,1).to(R_xx.dtype) @ Y_right
    Y_left  = torch.linalg.solve(R_xx.transpose(0,1), Phi.to(R_xx.dtype))    # (|S|, |U|)
    M_left  = Phi.transpose(0,1).to(R_xx.dtype) @ Y_left

    # ---- 7) Apply (M_left^T ⊗ M_right) to J via vec-trick: vec(M_right X M_left) ----
    Xb = _unvecF_batch(J, U)                # (P, U, U), P=2K+1
    Y  = torch.matmul(M_right, Xb)          # (P, U, U)
    Y  = torch.matmul(Y, M_left)            # (P, U, U)
    KJ = _vecF_batch(Y)                     # (U^2, P)

    # ---- 8) FIM and CRB (proper complex -> scale by L) ----
    L_t = torch.tensor(float(L_snapshots), device=device, dtype=torch.float64)
    F = L_t * (J.conj().transpose(0,1) @ KJ)                                  # (2K+1, 2K+1)

    try:
        CRB_beta = torch.linalg.inv(F)
    except RuntimeError:
        CRB_beta = torch.linalg.pinv(F)

    CRB_theta = CRB_beta[:K, :K].real
    return torch.diag(CRB_theta).clamp_min(0.0)  # (K,) rad^2


