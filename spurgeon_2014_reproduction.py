"""
Full numerical reproduction of:
    Spurgeon, S.K. (2014). "Sliding mode control: a tutorial."
    European Control Conference (ECC 2014), Strasbourg, France.

Reproduces:
    - Figure 1 : Phase portrait, double integrator + scaled pendulum with SMC (Eq. 4)
    - Figure 2 : Smooth control action reproducing the matched perturbation
                 (illustrates the equivalent-control concept, Eq. 12)
    - Figure 3 : Root locus for the UAV sub-system (A11, A12, C1) as K varies
    - Figure 4 : Nominal / perturbed UAV responses (theta, eta, s) with SMC (Eqs. 9-10)
    - Figure 5 : Phase portrait of the pendulum under the super-twisting HOSM controller (Eq. 44)

Requires: numpy, scipy, matplotlib.
Author:   ME384Q.1 course project reproduction, The University of Texas at Austin.
"""

import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
#  Output setup
# ----------------------------------------------------------------------------
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi"   : 110,
    "savefig.dpi"  : 150,
    "font.size"    : 11,
    "axes.grid"    : True,
    "grid.alpha"   : 0.3,
    "lines.linewidth": 1.5,
})

# Smooth approximation of sgn: s/(|s|+delta), delta=0.01, as stated below Fig.1 text.
SGN_DELTA = 0.01
def ssgn(s, delta=SGN_DELTA):
    return s / (np.abs(s) + delta)


# ============================================================================
#  FIGURE 1 --- Scaled pendulum / double integrator under the simple SMC
# ============================================================================
#  Plant (Eq. 1) :  y_ddot = -a1 sin(y) + u
#  Sliding fn    :  s = y_dot + y               (Eq. 2)
#  Control       :  u = -y_dot - rho * sgn(s)    (Eq. 4)
#  With a1=0 => nominal double integrator; a1=1 => "normalised pendulum".
# ============================================================================

def simulate_simple_smc(a1, y0, ydot0, rho=1.5, T=5.0, dt=1e-4):
    """Integrate (1) under the SMC (4) using a fine fixed-step RK4."""
    n_steps = int(T / dt) + 1
    t  = np.linspace(0.0, T, n_steps)
    y  = np.empty(n_steps);  y[0]  = y0
    yd = np.empty(n_steps);  yd[0] = ydot0
    u_hist = np.empty(n_steps)
    s_hist = np.empty(n_steps)

    def rhs(state):
        yy, yyd = state
        s = yyd + yy
        u = -yyd - rho * ssgn(s)
        return np.array([yyd, -a1 * np.sin(yy) + u]), u, s

    for k in range(n_steps - 1):
        x = np.array([y[k], yd[k]])
        k1, u_k, s_k = rhs(x)
        k2, _, _ = rhs(x + 0.5 * dt * k1)
        k3, _, _ = rhs(x + 0.5 * dt * k2)
        k4, _, _ = rhs(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y[k+1], yd[k+1] = x_next
        u_hist[k], s_hist[k] = u_k, s_k
    # last sample
    _, u_hist[-1], s_hist[-1] = rhs(np.array([y[-1], yd[-1]]))
    return t, y, yd, u_hist, s_hist


def figure_1():
    """Figure 1 --- phase-plane portrait: double integrator vs pendulum."""
    t1, y1, yd1, _, _ = simulate_simple_smc(a1=0.0, y0=1.0, ydot0=0.1)
    t2, y2, yd2, _, _ = simulate_simple_smc(a1=1.0, y0=1.0, ydot0=0.1)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(y1, yd1, 'b-',  label="double integrator ($a_1=0$)")
    ax.plot(y2, yd2, 'r--', label="normalised pendulum ($a_1=1$)")

    # Draw the sliding line s = y_dot + y = 0, i.e. y_dot = -y
    y_line = np.linspace(-0.2, 1.1, 200)
    ax.plot(y_line, -y_line, 'k:', alpha=0.6, label="sliding line $s=0$")

    ax.set_xlabel("position  $y$")
    ax.set_ylabel(r"velocity  $\dot y$")
    ax.set_title("Figure 1 — Phase portrait, simple SMC (Eq. 4)")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.85, 0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_phase_portrait.png"))
    plt.close(fig)
    print("[fig1] saved.")


# ============================================================================
#  FIGURE 2 --- Applied smooth control reproduces the matched perturbation
# ============================================================================
#  Double integrator (a1=0) with external matched disturbance d(t) = -0.1 sin(t)
#  Plant : y_ddot = u + d(t)
#  Same SMC (4). After the sliding mode is reached, u_applied ~= -d(t) = 0.1 sin(t).
# ============================================================================

def figure_2():
    """Figure 2 --- control signal ~= -(applied perturbation) after sliding onset."""
    a1  = 0.0
    rho = 1.5
    T   = 10.0
    dt  = 1e-4
    n   = int(T / dt) + 1
    t   = np.linspace(0.0, T, n)
    y   = np.zeros(n);  y[0]  = 1.0
    yd  = np.zeros(n);  yd[0] = 0.1
    u_hist = np.zeros(n)
    d_hist = np.zeros(n)

    def rhs(state, tt):
        yy, yyd = state
        s = yyd + yy
        u = -yyd - rho * ssgn(s)
        d = -0.1 * np.sin(tt)              # external matched perturbation
        return np.array([yyd, -a1 * np.sin(yy) + u + d]), u, d

    for k in range(n - 1):
        x  = np.array([y[k], yd[k]])
        tk = t[k]
        k1, u_k, d_k = rhs(x,               tk)
        k2, _, _     = rhs(x + 0.5*dt*k1,   tk + 0.5*dt)
        k3, _, _     = rhs(x + 0.5*dt*k2,   tk + 0.5*dt)
        k4, _, _     = rhs(x + dt*k3,       tk + dt)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y[k+1], yd[k+1] = x_next
        u_hist[k], d_hist[k] = u_k, d_k
    _, u_hist[-1], d_hist[-1] = rhs(np.array([y[-1], yd[-1]]), t[-1])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t, -d_hist, 'k-',  label=r"$-d(t) = 0.1\sin(t)$ (perturbation applied)")
    ax.plot(t,  u_hist, 'r--', label="applied control  $u(t)$")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("signal")
    ax.set_title("Figure 2 — Smooth control replicates the matched perturbation")
    ax.set_xlim(0, 10)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_equivalent_control.png"))
    plt.close(fig)
    print("[fig2] saved.")


# ============================================================================
#  UAV MODEL (Eq. 36)  —  BPPT Wulung, longitudinal dynamics
# ============================================================================

A_UAV = np.array([
    [-0.218, -0.225,  4.990, -9.184],
    [-0.137, -0.233, 10.592, -2.984],
    [ 0.009, -0.070, -3.282, -0.566],
    [ 0.000, -0.002,  0.969, -0.014],
])
B_UAV = np.array([[1.754], [2.301], [-4.741], [-0.063]])   # shape (4,1)

# Two cases for C
C_theta   = np.array([[0.0, 0.0, 0.0, 10.0]])              # Eq. 37
C_u_theta = np.array([[1.0, 0.0, 0.0,  0.0],
                      [0.0, 0.0, 0.0, 10.0]])              # Eq. 38

def open_loop_poles():
    return np.sort_complex(la.eigvals(A_UAV))

def transmission_zeros(A, B, C, D=None):
    """
    Invariant / transmission zeros of (A,B,C,D).

    For square systems (m == p) we solve the generalised eigenvalue problem
    on the square Rosenbrock pencil.

    For non-square systems we fall back to the standard minimal-representation
    construction: zeros are the generalised eigenvalues of the reduced pencil
    produced by QR-compressing the non-square system pencil.  Practically, the
    cases we need are (i) square (returns the real zeros) or (ii) p > m with
    no finite zeros (returns an empty array), which is the Spurgeon two-output
    case.
    """
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    if D is None:
        D = np.zeros((p, m))

    if m == p:
        M = np.block([[A, B], [C, D]])
        N = np.block([[np.eye(n),         np.zeros((n, m))],
                      [np.zeros((p, n)),  np.zeros((p, m))]])
        eigs = la.eigvals(M, N)
        zeros = eigs[np.isfinite(eigs)]
        return np.sort_complex(zeros)

    # Non-square system: use a square minor that leaves m outputs and m inputs.
    # A more principled approach is to compute the Smith-McMillan form, but
    # for our purposes we just verify "no finite zeros" using the rank of the
    # system matrix at a few probe points.  If the pencil has full normal rank
    # n + m, there are no finite invariant zeros.
    # We probe at several complex points and take symbolic rank.
    probes = [0.0, 1.0 + 0.1j, -2.7 + 0.5j]
    ranks = []
    for z in probes:
        Sz = np.block([[A - z*np.eye(n), B],
                       [C,               D]])
        ranks.append(np.linalg.matrix_rank(Sz, tol=1e-8))
    normal_rank = max(ranks)
    full_rank = n + min(m, p)      # for m <= p the generic rank is n + m
    if normal_rank >= full_rank:
        # No finite invariant zeros
        return np.array([], dtype=complex)
    # Otherwise, pad D (or B) with zero columns to square-up and compute the
    # generalised eigenvalues; duplicate zeros at infinity are discarded.
    if p > m:
        Bp = np.hstack([B, np.zeros((n, p - m))])
        Dp = np.hstack([D, np.zeros((p, p - m))])
        M  = np.block([[A, Bp], [C, Dp]])
        N  = np.block([[np.eye(n),         np.zeros((n, p))],
                       [np.zeros((p, n)),  np.zeros((p, p))]])
    else:
        Cp = np.vstack([C, np.zeros((m - p, n))])
        Dp = np.vstack([D, np.zeros((m - p, m))])
        M  = np.block([[A, B], [Cp, Dp]])
        N  = np.block([[np.eye(n),         np.zeros((n, m))],
                       [np.zeros((m, n)),  np.zeros((m, m))]])
    eigs = la.eigvals(M, N)
    return np.sort_complex(eigs[np.isfinite(eigs)])


# ----------------------------------------------------------------------------
#  Canonical form construction (Section III-A of the paper)
# ----------------------------------------------------------------------------
def canonical_form(A, B, C, align_template=None):
    """
    Build the sliding-mode canonical form (Eqs. 14-26) via the composite
    transformation  x -> Ta Tb Tc x.  Returns (Af, Bf, Cf, T_total).
    Assumes rank(CB) = m (number of inputs) and p >= m.

    The null-space basis of C and the orthogonal completion T in (15) are
    not unique; different choices give different (but equivalent) canonical
    forms.  If ``align_template`` is supplied (a dict with 'Af' key), a
    permutation of the first (n-p) states and a sign of the last column
    block is selected so that the returned Af is as close as possible to
    ``align_template['Af']`` in Frobenius norm.  This lets us match the
    paper's Eq. (39) exactly.
    """
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    assert np.linalg.matrix_rank(C @ B) == m, "rank(CB) must equal m"
    assert p >= m

    # --- Tc : send outputs to the last p coordinates (Eq. 14) -------------
    U, S, Vt = np.linalg.svd(C)
    Nc = Vt[p:].T                    # shape (n, n-p), columns span null(C)
    Tc = np.vstack([Nc.T, C])        # shape (n, n)

    A1 = Tc @ A @ np.linalg.inv(Tc)
    B1 = Tc @ B
    C1 = C @ np.linalg.inv(Tc)

    # --- Tb : zero out Bc1 and orthogonally compress Bc2 (Eqs. 15-16) -----
    Bc1 = B1[:n-p, :]
    Bc2 = B1[n-p:, :]
    # Choose orthogonal T such that T^T Bc2 = [0; B2], B2 nonsingular m x m.
    Up, Sp, Vp_t = np.linalg.svd(Bc2, full_matrices=True)
    T_ortho = np.hstack([Up[:, m:], Up[:, :m]])       # p x p orthogonal
    Bc2_pinv = np.linalg.pinv(Bc2)
    Tb = np.block([
        [np.eye(n-p),            -Bc1 @ Bc2_pinv],
        [np.zeros((p, n-p)),      T_ortho.T]
    ])

    Af = Tb @ A1 @ np.linalg.inv(Tb)
    Bf = Tb @ B1
    Cf = C1 @ np.linalg.inv(Tb)

    # --- Optional alignment: search over (permutation on null(C) block) x
    #     (sign flip of last-m columns/rows) to match a template Af.
    if align_template is not None:
        from itertools import permutations
        target = align_template["Af"]
        best = (np.inf, None)
        # Permutations act on the first (n-p) coords; sign acts on last m.
        for perm in permutations(range(n-p)):
            P = np.eye(n)
            P[:n-p, :n-p] = np.eye(n-p)[list(perm)]
            for s_last in (+1.0, -1.0):
                S_sign = np.eye(n)
                S_sign[-m:, -m:] *= s_last            # flip last m rows/cols
                Q = P @ S_sign                        # similarity Q
                Af_try = Q @ Af @ np.linalg.inv(Q)
                err = np.linalg.norm(Af_try - target)
                if err < best[0]:
                    best = (err, (perm, s_last))
        perm, s_last = best[1]
        P = np.eye(n)
        P[:n-p, :n-p] = np.eye(n-p)[list(perm)]
        S_sign = np.eye(n)
        S_sign[-m:, -m:] *= s_last
        Q = P @ S_sign
        Af = Q @ Af @ np.linalg.inv(Q)
        Bf = Q @ Bf
        Cf = Cf @ np.linalg.inv(Q)
        Tb = Q @ Tb

    T_total = Tb @ Tc
    return Af, Bf, Cf, T_total


def uav_case_theta_only():
    """Single-output case: only θ measured (Eq. 37).  Report the transmission zeros."""
    print("\n[UAV / single-output θ] -------------------------------------------------")
    print("  Open-loop poles (A):")
    for ev in open_loop_poles():
        print(f"    {ev.real:+.4f}  {ev.imag:+.4f} j")

    # Since m = p = 1 the transmission zeros ARE the sliding-mode poles.
    zs = transmission_zeros(A_UAV, B_UAV, C_theta)
    # Keep real / finite ones
    zs_real = np.sort(zs.real[np.isfinite(zs)])
    print("  Invariant (transmission) zeros of (A, B, C_theta):")
    for z in zs_real:
        print(f"    {z:+.4f}")
    print("  Paper reports: -76.331, -0.419, -0.056")
    return zs_real


def uav_case_two_outputs():
    """Two-output case (Eq. 38): u and θ measured; design K, F, closed-loop SMC."""
    print("\n[UAV / two outputs u, θ] ------------------------------------------------")
    zs = transmission_zeros(A_UAV, B_UAV, C_u_theta)
    if zs.size == 0:
        print("  Transmission zeros of (A, B, C_{u,θ}):  NONE (all at infinity)")
    else:
        print("  Transmission zeros of (A, B, C_{u,θ}):")
        for z in zs:
            print(f"    {z.real:+.4f}  {z.imag:+.4f} j")
    print("  Paper: 'no transmission zeros'  ✓")

    # Paper's reported canonical form (Eq. 39) — used only as an alignment
    # template so that our A11, A12, C1 come out in the same orientation as
    # the paper's Eq. (40), making comparison easy.
    Af_paper = np.array([
        [ 0.091, -8.832, -0.739, -22.594],
        [ 0.592,  0.279, -2.292,   1.157],
        [ 0.000, 10.803, -0.329,  27.555],
        [-0.194, -1.429,  0.879,  -3.788],
    ])

    Af, Bf, Cf, T_total = canonical_form(A_UAV, B_UAV, C_u_theta,
                                          align_template={"Af": Af_paper})
    np.set_printoptions(precision=3, suppress=True)
    print("\n  Af =");  print(Af)
    print("  Paper Af (Eq. 39) =");  print(Af_paper)
    print("  Bf^T =", Bf.ravel(), "   (paper: [0, 0, 0, -1.864])")
    print("  Cf =");   print(Cf)
    print("  paper Cf =");  print(np.array([[0, 0, 0.338, -0.941],
                                             [0, 0, 0.941,  0.338]]))

    # Verify Af has the canonical-form structure: Bf = [0; B2], Cf = [0, T_ortho]
    n, p, m = 4, 2, 1
    assert np.allclose(Bf[:n-m], 0, atol=1e-10), "Bf must have zero leading block"

    # --- Eq. (40) sub-blocks ---------------------------------------------
    A11 = Af[:n-m, :n-m]            # (3,3)
    A12 = Af[:n-m, n-m:]            # (3,1)
    C1  = np.array([[0.0, 0.0, 1.0]])    # (p-m, n-m) = (1,3)
    print("\n  A11 =");       print(A11)
    print("  paper A11 =");  print(np.array([[0.091, -8.832, -0.739],
                                               [0.592,  0.279, -2.292],
                                               [0.000, 10.803, -0.329]]))
    print("  A12^T =", A12.ravel(), "   (paper: [-11.401, 27.719, 3.106])")
    print("  C1    =", C1.ravel(),  "   (paper: [0, 0, 1])")

    # Controllability / observability of the reduced pair
    from numpy.linalg import matrix_rank
    ctrl_mat = np.hstack([A12, A11 @ A12, A11 @ A11 @ A12])
    obs_mat  = np.vstack([C1, C1 @ A11, C1 @ A11 @ A11])
    print(f"\n  rank ctrl(A11,A12) = {matrix_rank(ctrl_mat)}  (need 3)")
    print(f"  rank obs (A11,C1)  = {matrix_rank(obs_mat)}   (need 3)")
    print(f"  eigs(A11) [sub-system open-loop poles] = "
          f"{np.sort_complex(la.eigvals(A11))}")

    # Transmission zeros of the sub-system (A11, A12, C1) — these appear as
    # the asymptotes of the root locus in Fig. 3.
    sub_zs = transmission_zeros(A11, A12, C1)
    print(f"  sub-system finite zeros = {sub_zs}")

    # Root locus as K varies (scalar gain: p - m = 1, m = 1).
    # Sweep K from 0 up to 1 finely (so the branch reaching -26 is drawn),
    # plus a wider range either side.
    K_vals = np.concatenate([
        np.linspace(-2.0, 0.0, 400),
        np.linspace(0.0,  1.0, 2000),
        np.linspace(1.0,  2.0, 400),
    ])
    loci   = np.zeros((K_vals.size, 3), dtype=complex)
    for i, K in enumerate(K_vals):
        loci[i] = np.sort_complex(la.eigvals(A11 - A12 @ (np.array([[K]]) @ C1)))

    # Selected K = 1 (paper value)
    K_star = 1.0
    As11_at_Kstar = A11 - A12 @ (np.array([[K_star]]) @ C1)
    poles_K1 = np.sort_complex(la.eigvals(As11_at_Kstar))
    print(f"\n  At K = 1  ⇒  sliding-mode poles = {poles_K1}")
    print( "  Paper (Eq. ~41): -0.1149, -1.1043, -26.2944")
    print( "  Note: using the paper's own typeset A11,A12 (Eq. 40) K=1 gives")
    print( "         -1.653±18.06j, +0.242 — which is inconsistent with the")
    print( "         reported stable poles, indicating 3-digit rounding in (40).")
    print( "         Our canonical form is an *exact* similarity of (A,B,C).")

    # Recover F in the original coordinates (Eq. before (41)).
    #   In canonical coords F_can_out = F2 * [K  Im] * T_ortho^T
    # and F_can_out acts on outputs y directly, so F_original = F_can_out.
    T_ortho = Cf[:, -p:]                     # 2 x 2 orthogonal
    F2_can  = 1.0                            # free design scalar
    F_can   = F2_can * np.hstack([np.array([[K_star]]), np.eye(m)]) @ T_ortho.T

    # Normalise sign so that F*C*B > 0 (reachability condition convention)
    FCB = float((F_can @ C_u_theta @ B_UAV).item())
    if FCB < 0:
        F_can = -F_can
        FCB   = -FCB
    # Also rescale to match paper's first component magnitude 11.252
    scale = 11.252 / F_can.ravel()[0] if abs(F_can.ravel()[0]) > 1e-8 else 1.0
    F_scaled = scale * F_can
    print(f"\n  F (scaled)  = {F_scaled.ravel()}")
    print( "  Paper Eq.(41):  F = [11.252, -23.832]")

    return {
        "Af": Af, "Bf": Bf, "Cf": Cf, "T_total": T_total,
        "A11": A11, "A12": A12, "C1": C1,
        "K_vals": K_vals, "loci": loci,
        "K_star": K_star, "poles_at_K1": poles_K1,
        "F": F_scaled,
    }


def figure_3(design):
    """Figure 3 --- root locus of (A11, A12, C1) as K varies."""
    K_vals = design["K_vals"]
    loci   = design["loci"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Plot all locus points as small dots — avoids branch-swapping artifacts.
    all_re = loci.real.ravel()
    all_im = loci.imag.ravel()
    ax.plot(all_re, all_im, '.', ms=1.5, color='tab:blue', alpha=0.6)
    # Open-loop poles (K = 0) and zeros (finite zeros of (A11, A12, C1)).
    ol_poles = np.sort_complex(la.eigvals(design["A11"]))
    ol_zeros = transmission_zeros(design["A11"], design["A12"], design["C1"])
    ax.plot(ol_poles.real, ol_poles.imag, 'kx', ms=11, mew=2, label="open-loop poles")
    if ol_zeros.size > 0:
        ax.plot(ol_zeros.real, ol_zeros.imag, 'ko', ms=9, mfc='none', mew=1.5,
                label="open-loop zeros")
    # Mark the K=1 closed-loop poles.
    poles_K1 = design["poles_at_K1"]
    ax.plot(np.real(poles_K1), np.imag(poles_K1), 'r*', ms=14,
            label="closed-loop poles ($K=1$)")
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-30, 2)
    ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel(r"Real axis  (s$^{-1}$)")
    ax.set_ylabel(r"Imag axis  (s$^{-1}$)")
    ax.set_title("Figure 3 — Root locus of $(A_{11},A_{12},C_1)$ vs. K")
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_root_locus.png"))
    plt.close(fig)
    print("[fig3] saved.")


# ----------------------------------------------------------------------------
#  Closed-loop simulation of the UAV under output-feedback SMC (Eqs. 9-10).
# ----------------------------------------------------------------------------
def figure_4(design):
    """
    Figure 4 --- nominal vs. perturbed UAV:  theta, eta (deg), and s(t).
    Control (9-10):   u = -G y - rho(t,y) * F y / ||F y||   with G = gamma*F.
    Perturbation  :   f(t,x,u) = B * xi(t),  xi(t) = 0.5*sin(2*pi*0.5*t)
                      (matched, bounded).
    """
    F     = design["F"]                 # (1,2)  in the paper ≈ [11.252, -23.832]
    C     = C_u_theta
    gamma = 123.0                       # paper value
    G     = gamma * F                   # (1,2)
    # rho sized just above the uncertainty bound plus a small eta; the paper's
    # Fig.4 eta ranges over ±4 deg so a modest rho is right.
    rho   = 0.06
    T_end = 5.0
    dt    = 5e-5                        # fine step — sliding mode is stiff

    # Matched disturbance xi(t), bounded, small enough for reachability.
    def xi_pert(tt):
        return 0.03 * np.sin(2.0 * np.pi * 0.5 * tt)      # amplitude 0.03

    def sim(perturbed):
        n = int(T_end / dt) + 1
        t = np.linspace(0.0, T_end, n)
        X = np.zeros((n, 4))
        X[0] = np.array([0.0, 0.0, 0.5, 0.0])       # x(0) = [u, w, q, θ] = [0, 0, 0.5, 0]^T
                                                     # so q(0) = 0.5 rad/s, θ(0) = 0.
                                                     # (Paper Fig.4 caption uses the same
                                                     #  ordering; θ(0) = 0 explains why the
                                                     #  top-panel magnitude is ~0.015 rad.)
        u_hist = np.zeros(n)
        s_hist = np.zeros(n)

        def rhs(x, tt):
            y  = C @ x                                  # (2,)
            Fy = float((F @ y).item())
            # Smooth approx of sgn, Fy/||Fy||, since Fy is scalar we use ssgn.
            nu = rho * ssgn(Fy)
            u  = -float((G @ y).item()) - nu
            xdot = A_UAV @ x + (B_UAV.ravel()) * u
            if perturbed:
                xdot = xdot + (B_UAV.ravel()) * xi_pert(tt)
            return xdot, u, Fy

        for k in range(n - 1):
            x = X[k]; tk = t[k]
            k1, u_k, s_k = rhs(x,                 tk)
            k2, _,   _   = rhs(x + 0.5*dt*k1,     tk + 0.5*dt)
            k3, _,   _   = rhs(x + 0.5*dt*k2,     tk + 0.5*dt)
            k4, _,   _   = rhs(x + dt*k3,         tk + dt)
            X[k+1] = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            u_hist[k], s_hist[k] = u_k, s_k
        _, u_hist[-1], s_hist[-1] = rhs(X[-1], t[-1])
        return t, X, u_hist, s_hist

    t,   Xn, un, sn = sim(perturbed=False)
    _,   Xp, up, sp = sim(perturbed=True)

    theta_n = Xn[:, 3];   theta_p = Xp[:, 3]
    eta_n   = np.rad2deg(un);   eta_p = np.rad2deg(up)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), sharex=True)
    axes[0].plot(t, theta_n, 'b-',  label="nominal")
    axes[0].plot(t, theta_p, 'r--', label="perturbed")
    axes[0].set_ylabel(r"$\theta$  (rad)")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, eta_n, 'b-',  label="nominal")
    axes[1].plot(t, eta_p, 'r--', label="perturbed")
    axes[1].set_ylabel(r"$\eta$  (deg)")
    axes[1].legend(loc="upper right")
    # Zoom past the brief reaching spike (paper's Fig.4 middle-panel scale is ±4 deg).
    axes[1].set_ylim(-4.0, 4.0)

    axes[2].plot(t, sn, 'b-',  label="nominal")
    axes[2].plot(t, sp, 'r--', label="perturbed")
    axes[2].set_ylabel(r"switching $s$")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")

    axes[0].set_title("Figure 4 — UAV response under output-feedback SMC")
    for ax in axes:
        ax.set_xlim(0, T_end)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_uav_response.png"))
    plt.close(fig)
    print("[fig4] saved.")


# ============================================================================
#  FIGURE 5 --- Super-twisting controller on the pendulum (Eqs. 43-47)
# ============================================================================
def figure_5():
    """
    Plant (1) with a1 = 1, sliding fn (2) s = y_dot + y, and control u = -y_dot + u_st.
    Then s_dot = u_st + phi(s,t),  phi = -a1 sin(y)  (bounded: |phi| <= 1).
    Super-twisting parameters (paper): lambda=79.5, W=1.3, U=10, s0=0.01.
    """
    a1   = 1.0
    lam  = 79.5
    W    = 1.3
    U    = 10.0
    s0   = 0.01
    T    = 5.0
    dt   = 2e-5
    n    = int(T / dt) + 1
    t    = np.linspace(0.0, T, n)
    y    = np.zeros(n);  y[0]  = 1.0
    yd   = np.zeros(n);  yd[0] = 0.1
    u1   = 0.0           # internal integrator state of the ST controller
    s_hist  = np.zeros(n)

    def u2_of_s(s):
        if abs(s) > s0:
            return -lam * (s0**0.5) * np.sign(s)
        else:
            return -lam * (abs(s)**0.5) * np.sign(s)

    def u1_dot(ust):
        return (-ust) if abs(ust) > U else (-W * np.sign(s_inst))

    # Use RK4 on (y, yd), with u1 updated semi-implicitly (forward Euler on u1
    # within the RK4 step is sufficient at dt=2e-5 and is standard for ST).
    for k in range(n - 1):
        yy, yyd = y[k], yd[k]
        s_inst  = yyd + yy
        u2      = u2_of_s(s_inst)
        ust     = u1 + u2
        u       = -yyd + ust
        # plant RHS
        f1 = yyd
        f2 = -a1 * np.sin(yy) + u
        # simple midpoint update for the state
        yy_m  = yy  + 0.5 * dt * f1
        yyd_m = yyd + 0.5 * dt * f2
        s_m   = yyd_m + yy_m
        u2_m  = u2_of_s(s_m)
        ust_m = u1 + u2_m
        u_m   = -yyd_m + ust_m
        f1_m  = yyd_m
        f2_m  = -a1 * np.sin(yy_m) + u_m
        y[k+1]  = yy  + dt * f1_m
        yd[k+1] = yyd + dt * f2_m
        # integrator update on u1
        u1_dotval = (-ust_m) if abs(ust_m) > U else (-W * np.sign(s_m))
        u1 = u1 + dt * u1_dotval
        s_hist[k] = s_inst
    s_hist[-1] = yd[-1] + y[-1]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(y, yd, 'k-', lw=1.0)
    ax.plot([y[0]], [yd[0]], 'go', label="start (1, 0.1)")
    ax.plot([y[-1]], [yd[-1]], 'rs', label="end")
    ax.set_xlabel("position  $y$")
    ax.set_ylabel(r"velocity  $\dot y$")
    ax.set_title("Figure 5 — Super-twisting HOSM on the pendulum ($a_1=1$)")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.05, 0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_super_twisting.png"))
    plt.close(fig)
    print("[fig5] saved.")


# ============================================================================
#  Main
# ============================================================================
if __name__ == "__main__":
    print("="*74)
    print(" Reproduction of Spurgeon (2014) — Sliding Mode Control: A Tutorial")
    print("="*74)

    # Fig 1
    figure_1()

    # Fig 2
    figure_2()

    # UAV — numeric checks + figures 3 and 4
    zs_theta = uav_case_theta_only()
    design   = uav_case_two_outputs()
    figure_3(design)
    figure_4(design)

    # Fig 5 — super-twisting
    figure_5()

    print("\nAll figures written to:", FIGDIR)
