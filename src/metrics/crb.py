import math
import torch
from src.utils import build_phi  # your existing Φ(S) builder

def _vecF_col_major_4d_to_2d(outer_BUUK: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of UxU matrices (last two dims) into column-major vecs.
    Input : (B, U, U, K)
    Output: (B, U*U, K)
    vec_Fcol(M) = vec_column_major(M) = vec((M^T)^T) -> implemented as transpose before reshape.
    """
    B, U, _, K = outer_BUUK.shape
    return outer_BUUK.transpose(1, 2).reshape(B, U * U, K)

def _unvecF_col_major_2d_to_4d(J_BU2P: torch.Tensor, U: int) -> torch.Tensor:
    """
    Inverse of column-major vec, for a batch of vectors stacked along the last dim.
    Input : (B, U*U, P)
    Output: (B, P, U, U)
    """
    B, _, P = J_BU2P.shape
    tmp = J_BU2P.transpose(1, 2).reshape(B, P, U, U)   # this equals M^T
    return tmp.transpose(-1, -2)                       # back to M

def calculate_sncr_crb_batched(
    S_positions: torch.Tensor,
    thetas_deg: torch.Tensor,      # (B, K) or (K,)
    snr_db: float | torch.Tensor,  # scalar or (B,) or (B,K)
    L_snapshots: int | float | torch.Tensor,  # scalar or (B,)
    d: float = 0.5,
    sigma2: float = 1.0,
    return_per_angle: bool = True,  # if False -> returns per-sample RMSE lower bound (rad)
    dtype_complex: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Batched SNCR stochastic CRB (proper complex Gaussian).
    Returns:
      - if return_per_angle=True : (B, K) per-angle variances in rad^2
      - else                     : (B,)   RMSE lower bound per sample (rad), i.e. sqrt(mean diag)
    """
    device = S_positions.device if isinstance(S_positions, torch.Tensor) else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    S = S_positions.to(device=device, dtype=torch.long)

    thetas_deg = torch.as_tensor(thetas_deg, dtype=torch.float64, device=device)
    if thetas_deg.dim() == 1:
        thetas_deg = thetas_deg.unsqueeze(0)  # (1, K)
    B, K = thetas_deg.shape

    # Φ from your sparse array → determines U
    Phi = build_phi(S).to(device)                         # (|S|, U), real
    S_count, U = Phi.shape
    U_pos = torch.arange(0, U, dtype=torch.float64, device=device)

    # Phase-only steering on the filled grid
    phase_scale = 2.0 * math.pi * d                       # matches exp(-j 2π d * u * sinθ)
    th = torch.deg2rad(thetas_deg)                        # (B, K)
    sin_th = torch.sin(th)                                # (B, K)
    cos_th = torch.cos(th)                                # (B, K)

    # A(b,u,k) = exp(-j * phase_scale * u * sin(theta_bk))
    phase = -phase_scale * U_pos.view(1, U, 1) * sin_th.view(B, 1, K)  # (B, U, K)
    A = torch.exp(1j * phase).to(dtype_complex)                          # (B, U, K)

    # dA/dθ(b,u,k) = (j * -phase_scale * u * cosθ_bk) * A(b,u,k)
    coeff = (1j * -phase_scale) * U_pos.view(1, U, 1) * cos_th.view(B, 1, K)  # (B, U, K)
    dA = (coeff.to(dtype_complex) * A)                                         # (B, U, K)

    # SNR -> p(b,k). For phase-only steering, column power is 1; we compute anyway for generality.
    # If you ever add amplitude taper on the PHYSICAL array, prefer c_k from A_S = Φ A.
    snr_db_t = torch.as_tensor(snr_db, dtype=torch.float64, device=device)
    if snr_db_t.dim() == 0:
        snr_db_t = snr_db_t.expand(B, K)
    elif snr_db_t.dim() == 1:
        snr_db_t = snr_db_t.view(-1, 1).expand(B, K)
    snr_lin = 10.0 ** (snr_db_t / 10.0)                                       # (B, K)

    sigma2_t = torch.as_tensor(sigma2, dtype=torch.float64, device=device)
    # Column power on the filled grid (phase-only -> ones)
    col_power = (A.abs() ** 2).mean(dim=1).real.clamp_min(1e-12)              # (B, K)
    p = (snr_lin * sigma2_t) / col_power                                      # (B, K), real64
    p_c = p.to(dtype_complex)                                                 # complex for products

    # ---------- Build J = [D_theta, D_p, d_sigma] (all batched, no kron) ----------
    # D_p(:,k) = vec( a_k * a_k^H ), using column-major vec  ⇒ vec = transpose then reshape
    outer_aa = torch.einsum('buk,bvk->buvk', A, A.conj())                     # (B, U, U, K)
    D_p = _vecF_col_major_4d_to_2d(outer_aa)                                   # (B, U^2, K)

    # D_theta(:,k) = p_k * vec( dA_k a_k^H + a_k dA_k^H )
    outer_da_a = torch.einsum('buk,bvk->buvk', dA, A.conj())                  # (B, U, U, K)
    outer_a_da = torch.einsum('buk,bvk->buvk', A,  dA.conj())                 # (B, U, U, K)
    Dth_4d = (outer_da_a + outer_a_da) * p_c.view(B, 1, 1, K)                 # (B, U, U, K)
    D_theta = _vecF_col_major_4d_to_2d(Dth_4d)                                 # (B, U^2, K)

    # d_sigma = vec(I_U) shared across batch
    d_sigma = torch.eye(U, dtype=dtype_complex, device=device)
    d_sigma_vec = d_sigma.transpose(-2, -1).reshape(U * U)                    # (U^2,)
    d_sigma_B = d_sigma_vec.view(1, -1, 1).expand(B, -1, 1)                   # (B, U^2, 1)

    # J: (B, U^2, 2K+1)
    J = torch.cat([D_theta, D_p, d_sigma_B], dim=2)

    # ---------- Covariances ----------
    # R_yy(b) = A(b) diag(p_b) A(b)^H + sigma2 I
    Ap = A * p_c.view(B, 1, K)                                                # (B, U, K)
    R_yy = Ap @ A.conj().transpose(-2, -1) + sigma2_t.to(dtype_complex) * torch.eye(U, dtype=dtype_complex, device=device)  # (B, U, U)

    # R_xx(b) = Φ R_yy(b) Φ^H
    Phi_c = Phi.to(dtype_complex)
    # Using bmm via einsum to avoid explicit expand:
    R_xx = torch.einsum('su, buv, tv -> bst', Phi_c, R_yy, Phi_c.conj())      # (B, S, S)

    # ---------- Left/right weights via batched solves ----------
    # Solve R_xx(b) Y_right(b) = Φ   → Y_right: (B, S, U)
    Phi_rhs = Phi_c.expand(B, -1, -1)
    Y_right = torch.linalg.solve(R_xx, Phi_rhs)                                # (B, S, U)
    M_right = torch.einsum('us, b s v -> b u v', Phi_c.T, Y_right)            # (B, U, U)

    # Solve R_xx(b)^T Y_left(b) = Φ   → Y_left: (B, S, U)
    Y_left  = torch.linalg.solve(R_xx.transpose(-2, -1), Phi_rhs)              # (B, S, U)
    M_left  = torch.einsum('us, b s v -> b u v', Phi_c.T, Y_left)             # (B, U, U)

    # ---------- Apply (M_left^T ⊗ M_right) to J via vec-trick ----------
    # X_b = unvec(J_b) → (B, P, U, U), then Y = M_right X_b M_left
    Ptot = 2 * K + 1
    Xb = _unvecF_col_major_2d_to_4d(J, U)                                      # (B, Ptot, U, U)

    # Multiply left/right in (B*P, U, U) space to leverage fast bmm
    Xbp = Xb.reshape(B * Ptot, U, U)
    M_right_rep = M_right.unsqueeze(1).expand(B, Ptot, U, U).reshape(B * Ptot, U, U)
    M_left_rep  = M_left .unsqueeze(1).expand(B, Ptot, U, U).reshape(B * Ptot, U, U)
    Y = M_right_rep @ Xbp @ M_left_rep                                         # (B*P, U, U)
    Y = Y.reshape(B, Ptot, U, U)

    # Back to vec (column-major): (B, U^2, Ptot)
    KJ = _vecF_col_major_4d_to_2d(Y.permute(0, 2, 3, 1))  # (B, U^2, Ptot)

    # ---------- FIM and CRB ----------
    # F = L * J^H KJ
    L_t = torch.as_tensor(L_snapshots, dtype=torch.float64, device=device)
    if L_t.dim() == 0:
        L_t = L_t.expand(B)
    F = torch.einsum('bpu, buq -> bpq', J.conj().transpose(-2, -1), KJ)        # (B, Ptot, Ptot)
    F = (F + F.conj().transpose(-2, -1)) * 0.5                                 # Hermitize
    F = F * L_t.view(B, 1, 1)

    # Invert batch
    try:
        CRB = torch.linalg.inv(F)
    except RuntimeError:
        CRB = torch.linalg.pinv(F, rcond=1e-12)

    # θ-block diag → per-angle variances (B, K)
    CRB_theta = CRB[:, :K, :K].real
    per_angle_var = torch.diagonal(CRB_theta, dim1=-2, dim2=-1).clamp_min(0.0) # (B, K)

    if return_per_angle:
        return per_angle_var  # (B, K) in rad^2

    # Else, per-sample RMSE lower bound (rad): sqrt(mean_k variance_k)
    rmse_lb = torch.sqrt(per_angle_var.mean(dim=1))                             # (B,)
    return rmse_lb
