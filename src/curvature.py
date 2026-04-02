"""
src/curvature.py
================
Geometric analysis of engagement trajectories via the Frenet-Serret frame.

Given an engagement trajectory

    e(t) = (r(t), l(t), c(t)) ∈ ℝ³

where r = retention rate, l = like-accumulation rate, c = comment rate,
this module computes:

    κ(t)  — curvature   : how sharply the trajectory bends at time t
    τ(t)  — torsion     : how much the trajectory twists out of its
                          osculating plane at time t
    W     — total attention work       : ∫ κ(t) ‖e'(t)‖ dt
    D     — cognitive dissociation     : ∫ |τ(t)| ‖e'(t)‖ dt
    T,N,B — Frenet-Serret frame        : tangent, normal, binormal

Mathematical background
-----------------------
For a smooth curve γ: [0,T] → ℝⁿ parametrised by arc length s,
the curvature and torsion are defined via the Frenet-Serret equations:

    dT/ds = κ N
    dN/ds = -κ T + τ B
    dB/ds = -τ N

where T = γ'(s), N = T'/‖T'‖, B = T × N.

For an arbitrary (not arc-length) parametrisation t:

    κ(t) = ‖e'(t) × e''(t)‖ / ‖e'(t)‖³
    τ(t) = det[e'(t), e''(t), e'''(t)] / ‖e'(t) × e''(t)‖²

These are computed numerically via finite differences.

Interpretation
--------------
High κ(t): the audience's retention, like, and comment signals are all
           changing direction simultaneously — an emotional peak, a
           narrative turn, a surprising moment.

High τ(t): the three signals are diverging from each other — people
           watch but don't like (high τ with positive torsion), or like
           but don't comment (high τ with negative torsion). This is the
           *cognitive dissociation index* D.

The conjecture: 
    - High W, low D  → coherent peak → predicts shares
    - High W, high D → split reaction → predicts comments not shares  
    - Low W, low D   → passive consumption → predicts rewatches

Author: Khadidiatou Cissé
Date:   April 2026
"""

import numpy as np
from numpy.typing import NDArray


def smooth(x: NDArray, window: int = 5) -> NDArray:
    """
    Apply a simple moving average to reduce finite-difference noise.
    
    Uses convolution with a uniform kernel. Edge effects are handled
    by reflecting the signal at the boundaries.
    
    Parameters
    ----------
    x      : 1D array, the signal to smooth
    window : int, smoothing window width (must be odd for symmetry)
    
    Returns
    -------
    1D array of same length as x
    """
    if window < 2:
        return x
    kernel = np.ones(window) / window
    padded = np.pad(x, window // 2, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(x)]


def finite_differences(e: NDArray, dt: float = 1.0) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute first, second, and third derivatives of a trajectory via
    central finite differences.
    
    Parameters
    ----------
    e  : array of shape (T, 3), the engagement trajectory
    dt : time step (defaults to 1 = uniform steps)
    
    Returns
    -------
    (e1, e2, e3) : arrays of shape (T, 3), first/second/third derivatives
    """
    e1 = np.gradient(e, dt, axis=0)      # first derivative
    e2 = np.gradient(e1, dt, axis=0)     # second derivative
    e3 = np.gradient(e2, dt, axis=0)     # third derivative
    return e1, e2, e3


def curvature(e: NDArray, dt: float = 1.0,
              smooth_window: int = 5, eps: float = 1e-8) -> NDArray:
    """
    Compute the curvature κ(t) of the engagement trajectory.
    
    κ(t) = ‖e'(t) × e''(t)‖ / ‖e'(t)‖³
    
    High κ at time t means the trajectory is bending sharply —
    a simultaneous change in multiple engagement dimensions.
    
    Parameters
    ----------
    e             : array of shape (T, 3)
    dt            : time step
    smooth_window : window for pre-smoothing the trajectory
    eps           : small constant to avoid division by zero
    
    Returns
    -------
    kappa : array of shape (T,), curvature at each time step
    """
    # Smooth before differentiating to reduce noise amplification
    e_s = np.column_stack([smooth(e[:, i], smooth_window) for i in range(e.shape[1])])
    e1, e2, _ = finite_differences(e_s, dt)
    
    cross      = np.cross(e1, e2)             # shape (T, 3)
    cross_norm = np.linalg.norm(cross, axis=1)   # ‖e' × e''‖
    speed      = np.linalg.norm(e1, axis=1)      # ‖e'‖
    
    return cross_norm / (speed**3 + eps)


def torsion(e: NDArray, dt: float = 1.0,
            smooth_window: int = 5, eps: float = 1e-8) -> NDArray:
    """
    Compute the torsion τ(t) of the engagement trajectory.
    
    τ(t) = det[e'(t), e''(t), e'''(t)] / ‖e'(t) × e''(t)‖²
    
    High |τ| at time t means the three engagement signals are diverging
    from each other — the curve is twisting out of its osculating plane.
    This is the signature of *cognitive dissociation*: the audience is
    responding differently along each modality dimension.
    
    Note: torsion is signed. Positive τ means the binormal is rotating
    in the standard orientation; negative τ means the opposite. Both
    indicate dissociation; the sign reveals which signals are decoupling.
    
    Parameters
    ----------
    e             : array of shape (T, 3)
    dt            : time step
    smooth_window : window for pre-smoothing
    eps           : small constant to avoid division by zero
    
    Returns
    -------
    tau : array of shape (T,), torsion at each time step
    """
    e_s = np.column_stack([smooth(e[:, i], smooth_window) for i in range(e.shape[1])])
    e1, e2, e3 = finite_differences(e_s, dt)
    
    # Scalar triple product: det[e', e'', e''']
    # For 3D vectors: det = e' · (e'' × e''')
    cross_12 = np.cross(e1, e2)                   # e' × e''
    triple   = np.einsum('ij,ij->i', e3, cross_12) # e''' · (e' × e'')
    denom    = np.linalg.norm(cross_12, axis=1)**2 # ‖e' × e''‖²
    
    return triple / (denom + eps)


def frenet_serret_frame(e: NDArray, dt: float = 1.0,
                        smooth_window: int = 5,
                        eps: float = 1e-8) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute the Frenet-Serret frame (T, N, B) along the trajectory.
    
    T(t) = e'(t) / ‖e'(t)‖          tangent vector
    N(t) = T'(t) / ‖T'(t)‖          principal normal
    B(t) = T(t) × N(t)              binormal
    
    The frame rotates as the curve evolves; the rate of rotation is
    governed by κ (in the T-N plane) and τ (in the N-B plane).
    
    Parameters
    ----------
    e             : array of shape (T, 3)
    dt            : time step
    smooth_window : window for pre-smoothing
    eps           : small constant
    
    Returns
    -------
    (T, N, B) : each of shape (T, 3), the Frenet-Serret frame at each time step
    """
    e_s = np.column_stack([smooth(e[:, i], smooth_window) for i in range(e.shape[1])])
    e1, e2, _ = finite_differences(e_s, dt)
    
    # Tangent
    speed = np.linalg.norm(e1, axis=1, keepdims=True)
    T     = e1 / (speed + eps)
    
    # Principal normal
    T1       = np.gradient(T, dt, axis=0)
    T1_norm  = np.linalg.norm(T1, axis=1, keepdims=True)
    N        = T1 / (T1_norm + eps)
    
    # Binormal
    B = np.cross(T, N)
    
    return T, N, B


def total_attention_work(e: NDArray, dt: float = 1.0,
                         smooth_window: int = 5) -> float:
    """
    Compute the total attention work W.
    
    W = ∫₀ᵀ κ(t) ‖e'(t)‖ dt
    
    This is the length-weighted total curvature of the engagement path.
    It is a coordinate-free, scale-invariant summary of how dynamically
    complex the audience response is over the video's lifetime.
    
    Returns
    -------
    W : float, total attention work
    """
    e_s = np.column_stack([smooth(e[:, i], smooth_window) for i in range(e.shape[1])])
    e1, e2, _ = finite_differences(e_s, dt)
    
    kappa = curvature(e, dt, smooth_window)
    speed = np.linalg.norm(e1, axis=1)
    
    # Numerical integration by the trapezoidal rule
    integrand = kappa * speed
    return float(np.trapezoid(integrand, dx=dt))


def cognitive_dissociation(e: NDArray, dt: float = 1.0,
                           smooth_window: int = 5) -> float:
    """
    Compute the cognitive dissociation index D.
    
    D = ∫₀ᵀ |τ(t)| ‖e'(t)‖ dt
    
    This is the torsion analogue of W. High D means the three engagement
    signals (retention, likes, comments) are systematically diverging
    from each other throughout the video. Low D means they move together
    in a coordinated, coherent manner.
    
    Conjectured interpretation:
        High W, low D  → coherent peak           → predicts shares
        High W, high D → split cognitive response → predicts comments
        Low W, low D   → passive consumption      → predicts rewatches
    
    Returns
    -------
    D : float, cognitive dissociation index
    """
    e_s = np.column_stack([smooth(e[:, i], smooth_window) for i in range(e.shape[1])])
    e1, _, _ = finite_differences(e_s, dt)
    
    tau   = torsion(e, dt, smooth_window)
    speed = np.linalg.norm(e1, axis=1)
    
    integrand = np.abs(tau) * speed
    return float(np.trapezoid(integrand, dx=dt))


def curvature_peaks(kappa: NDArray, threshold: float = 0.5) -> NDArray:
    """
    Identify time indices where curvature exceeds a threshold fraction
    of its maximum value.
    
    These are the 'emotional events' in the video — moments of sharp
    audience response change. Their positions in the video timeline
    are the most interpretable output of the geometric analysis.
    
    Parameters
    ----------
    kappa     : array of shape (T,), curvature time series
    threshold : fraction of max curvature above which a point is a 'peak'
    
    Returns
    -------
    peak_indices : array of indices where κ(t) is a local maximum
                   above the threshold
    """
    cutoff = threshold * kappa.max()
    peaks  = []
    for i in range(1, len(kappa) - 1):
        if kappa[i] > cutoff and kappa[i] >= kappa[i-1] and kappa[i] >= kappa[i+1]:
            peaks.append(i)
    return np.array(peaks)


def engagement_summary(e: NDArray, dt: float = 1.0,
                       smooth_window: int = 5) -> dict:
    """
    Compute all geometric summary statistics for a single engagement trajectory.
    
    This is the main function to call on each video. It returns all the
    features that downstream models (M3 and M4) will use.
    
    Parameters
    ----------
    e             : array of shape (T, 3) — columns: [retention, like_rate, comment_rate]
    dt            : time step
    smooth_window : smoothing window
    
    Returns
    -------
    dict with keys:
        W          : total attention work
        D          : cognitive dissociation index
        W_D_ratio  : W / (D + ε) — coherence ratio
        kappa_max  : maximum curvature
        kappa_mean : mean curvature
        n_peaks    : number of curvature peaks
        tau_mean   : mean torsion
        tau_std    : std of torsion
        arc_length : total path length ∫ ‖e'‖ dt
    """
    kappa = curvature(e, dt, smooth_window)
    tau   = torsion(e, dt, smooth_window)
    
    e_s = np.column_stack([smooth(e[:, i], smooth_window) for i in range(e.shape[1])])
    e1, _, _ = finite_differences(e_s, dt)
    speed = np.linalg.norm(e1, axis=1)
    
    W          = float(np.trapezoid(kappa * speed, dx=dt))
    D          = float(np.trapezoid(np.abs(tau) * speed, dx=dt))
    arc_length = float(np.trapezoid(speed, dx=dt))
    peaks      = curvature_peaks(kappa)
    
    return {
        'W':           W,
        'D':           D,
        'W_D_ratio':   W / (D + 1e-8),
        'kappa_max':   float(kappa.max()),
        'kappa_mean':  float(kappa.mean()),
        'n_peaks':     len(peaks),
        'tau_mean':    float(tau.mean()),
        'tau_std':     float(tau.std()),
        'arc_length':  arc_length,
    }


# ── Demo / sanity check ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Curvature module — sanity check")
    print("=" * 50)

    T = 200
    t = np.linspace(0, 2 * np.pi, T)
    dt = t[1] - t[0]

    # Type 1: Smooth decay — nearly planar (low W, low D expected)
    r1 = 0.9 * np.exp(-0.5 * t / (2 * np.pi)) + 0.05
    l1 = 0.8 * np.exp(-0.6 * t / (2 * np.pi)) + 0.04
    c1 = 0.7 * np.exp(-0.7 * t / (2 * np.pi)) + 0.03
    e1 = np.column_stack([r1, l1, c1])

    # Type 2: Viral with coherent spike — all three signals spike together
    # (high W, low D: the curve bends sharply but stays in-plane)
    s = t / (2 * np.pi)
    spike = 0.5 * np.exp(-80 * (s - 0.55) ** 2)
    r2 = 0.9 * np.exp(-0.5 * s) + 0.05 + spike
    l2 = 0.8 * np.exp(-0.5 * s) + 0.04 + spike * 0.95
    c2 = 0.7 * np.exp(-0.5 * s) + 0.03 + spike * 0.90
    e2 = np.column_stack([r2, l2, c2])

    # Type 3: Controversial — retention spikes early, comments spike late,
    # likes barely move. Signals genuinely decouple → high torsion.
    spike_r = 0.5 * np.exp(-60 * (s - 0.35) ** 2)
    spike_c = 0.5 * np.exp(-60 * (s - 0.75) ** 2)
    r3 = 0.9 * np.exp(-0.4 * s) + 0.05 + spike_r
    l3 = 0.8 * np.exp(-0.4 * s) + 0.04 + spike_r * 0.15   # likes barely respond
    c3 = 0.7 * np.exp(-0.4 * s) + 0.03 + spike_c           # comments respond later
    e3 = np.column_stack([r3, l3, c3])

    for name, e in [
        ('Smooth decay        (expect: low W,  low D)', e1),
        ('Viral coherent      (expect: high W, low D)', e2),
        ('Controversial split (expect: high W, high D)', e3),
    ]:
        stats = engagement_summary(e, dt=dt, smooth_window=3)
        print(f"\n{name}")
        print(f"  W          = {stats['W']:.4f}")
        print(f"  D          = {stats['D']:.4f}")
        print(f"  W/D ratio  = {stats['W_D_ratio']:.4f}")
        print(f"  kappa_max  = {stats['kappa_max']:.6f}")
        print(f"  n_peaks    = {stats['n_peaks']}")
        print(f"  tau_mean   = {stats['tau_mean']:.6f}")
