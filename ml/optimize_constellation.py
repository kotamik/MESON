#!/usr/bin/env python3
# optimize_constellation.py
# offline pipeline that designs a lunar south-pole satellite constellation.
#
# two things working together:
#   1. a genetic algorithm searching the mixed integer/continuous space of
#      {number of sats + their orbital elements}. it wants max constant south-pole
#      coverage with as few sats as possible, low perturbation fragility
#      (longevity), and a tight spread of orbit types (complexity).
#   2. a little neural-net surrogate (a tiny MLP, written by hand in numpy)
#      trained on the genome -> fitness pairs the GA throws off. each generation
#      it pre-screens an over-produced batch of candidates so only the promising
#      ones hit the slow physics sim. the usual "surrogate-assisted evolution".
#
# physics here is the SAME as the live sandbox (src/ephemeris.js + src/coverage.js):
# build the real sun/earth/moon from j2000 elements, integrate them as a
# point-mass n-body system, then propagate each sat (rk4) in that true
# ecliptic-frame field - sun + earth + moon, moon's real eccentric inclined orbit
# and all. earth's third-body pull dominates at useful south-pole altitudes and
# drives kozai cycles that wreck a naive "frozen" orbit, so modelling the whole
# field is exactly what makes the result hold up once it's flying in the sandbox.
#
# the hot loops (the big-body ephemeris and the per-generation sat propagation)
# are numba-jitted, and the population is evaluated one-genome-per-core with
# prange, so a full 2-year search runs in a couple of minutes on all cores
# instead of grinding through it single-threaded.
#
# spits out ml/best_constellation.json (the sandbox can import it) + a history csv.
# needs numpy + numba (see requirements.txt). run:  py ml/optimize_constellation.py
import json
import os
import sys
import time
import math
import winsound
import numpy as np
import config as cf

# numba does the heavy lifting: it JIT-compiles the tight integration loops to
# native code and runs the population evaluation across every cpu core (prange).
try:
    from numba import njit, prange, get_num_threads
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        return args[0] if (args and callable(args[0])) else (lambda fn: fn)

    def prange(*a):
        return range(*a)

    def get_num_threads():
        return 1

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

rng = np.random.default_rng(42)

# real constants, kept in sync with src/constants.js (don't edit)
GM_SUN   = 1.32712440018e20
GM_EARTH = 3.986004418e14
GM_MOON  = 4.9028695e12
GM_JUPITER = 1.26686534e17
M_SUN, M_EARTH, M_MOON, M_JUPITER = 1.98847e30, 5.97219e24, 7.342e22, 1.898125e27
R_MOON   = 1.7374e6
DAY      = 86400.0
YEAR     = 365.25 * DAY
DEG      = np.pi / 180.0
RAD      = 180.0 / np.pi
AU       = 1.495978707e11
MOON_TILT = 1.5424 * DEG                       # lunar equator tilt to ecliptic

EARTH_EL = dict(a=1.00000261 * AU, e=0.01671123, i=-0.00001531 * DEG,
                om=0.0, w=102.93768193 * DEG,
                M=(100.46457166 - 102.93768193) * DEG)
MOON_EL  = dict(a=3.84399e8, e=0.0549, i=5.145 * DEG, om=125.08 * DEG,
                w=318.15 * DEG, M=135.27 * DEG)
JUPITER_EL = dict(a=5.204267 * AU, e=0.048775, i=1.3033 * DEG, om=100.556 * DEG,
                w=14.753 * DEG, M=(20.020) * DEG,)              

# south pole direction in the ecliptic frame (equator tilted about x)
SOUTH_DIR = np.array([0.0, np.sin(MOON_TILT), -np.cos(MOON_TILT)])
# past this a sat has left the system - count it as deorbited 
ESCAPE = 5 * EARTH_EL["a"]

# search bounds for one sat's normalised [0,1] genes.
# apoapsis is capped well inside the strongly-perturbed outer zone on purpose
# (the lunar hill radius is ~60,000 km). very high apoapsis orbits go chaotic
# under earth's pull: a set that scores 100% in one integrator scores way less in
# another, so those "solutions" are numerical artefacts, not real designs.
# staying low keeps the coverage reproducible.
A_MIN = R_MOON + 150e3
A_MAX = R_MOON + 8000e3
E_MAX = cf.EMX
GENES = 6

# GA window is two full years - a realistic mission lifetime. long enough that
# the slow earth-driven kozai cycles play out many times over during the search,
# so the GA can't reward an orbit that just survives a short horizon and then
# falls apart.
DURATION     = cf.DUR* YEAR
DT           = cf.DTS          # 5-min steps. numba makes this cheap, so the GA now
                              # scores at the same fidelity it's verified at

MIN_ELEV     = cf.MEV * DEG
MIN_PERI_ALT = cf.MPA
APO_SOFT     = cf.APS        # apoapsis altitude past which perturbations start to bite


# orbital mechanics
def solve_kepler(M, e, iters=14):
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.full_like(np.asarray(M, float), np.pi))
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def elements_to_state(el, mu):
    """el (K,6) -> state (K,6); generic parent mu.  Perifocal -> frame (3-1-3)."""
    a, e, i, om, w, M = [el[:, k] for k in range(6)]
    E = solve_kepler(M, e)
    nu = np.arctan2(np.sqrt(1 - e**2) * np.sin(E), np.cos(E) - e)
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    xp, yp = r * np.cos(nu), r * np.sin(nu)
    vc = np.sqrt(mu / p)
    vxp, vyp = -vc * np.sin(nu), vc * (e + np.cos(nu))
    co, so = np.cos(om), np.sin(om)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)
    R11 = co * cw - so * sw * ci
    R12 = -co * sw - so * cw * ci
    R21 = so * cw + co * sw * ci
    R22 = -so * sw + co * cw * ci
    R31 = sw * si
    R32 = cw * si
    x = R11 * xp + R12 * yp
    y = R21 * xp + R22 * yp
    z = R31 * xp + R32 * yp
    vx = R11 * vxp + R12 * vyp
    vy = R21 * vxp + R22 * vyp
    vz = R31 * vxp + R32 * vyp
    return np.stack([x, y, z, vx, vy, vz], axis=1)


def _state1(el_dict, mu):
    el = np.array([[el_dict[k] for k in ("a", "e", "i", "om", "w", "M")]])
    return elements_to_state(el, mu)[0]


def rotX(v, t):
    c, s = np.cos(t), np.sin(t)
    return np.array([v[..., 0], c * v[..., 1] - s * v[..., 2],
                     s * v[..., 1] + c * v[..., 2]]).T


# big-body ephemeris (sun + earth + moon), cached
_EPH = {}


def _accel_bodies(P, GMs):
    A = np.zeros_like(P)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = P[j] - P[i]
            A[i] += GMs[j] * d / (np.sqrt(d @ d) ** 3)
    return A


# --- numba kernels: 3-body ephemeris integration -----------------------------
@njit
def _acc3(P, GMs):
    A = np.zeros((4, 3))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            dx = P[j, 0] - P[i, 0]; dy = P[j, 1] - P[i, 1]; dz = P[j, 2] - P[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            inv = GMs[j] / (r2 * math.sqrt(r2))
            A[i, 0] += inv * dx; A[i, 1] += inv * dy; A[i, 2] += inv * dz
    return A


@njit
def _prop_bodies3(n, hdt, P0, V0, GMs):
    sunP = np.empty((n, 3)); earthP = np.empty((n, 3)); moonP = np.empty((n, 3)); jupiterP = np.empty((n, 3))
    P = P0.copy(); V = V0.copy()
    h2 = hdt * 0.5
    for k in range(n):
        sunP[k, 0] = P[0, 0]; sunP[k, 1] = P[0, 1]; sunP[k, 2] = P[0, 2]
        earthP[k, 0] = P[1, 0]; earthP[k, 1] = P[1, 1]; earthP[k, 2] = P[1, 2]
        moonP[k, 0] = P[2, 0]; moonP[k, 1] = P[2, 1]; moonP[k, 2] = P[2, 2]
        jupiterP[k, 0] = P[3, 0]; jupiterP[k, 1] = P[3, 1]; jupiterP[k,2] = P[3, 2]
        if k == n - 1:
            break
        A1 = _acc3(P, GMs)
        P2 = P + V * h2;  V2 = V + A1 * h2
        A2 = _acc3(P2, GMs)
        P3 = P + V2 * h2; V3 = V + A2 * h2
        A3 = _acc3(P3, GMs)
        P4 = P + V3 * hdt; V4 = V + A3 * hdt
        A4 = _acc3(P4, GMs)
        P = P + (hdt / 6.0) * (V + 2.0 * V2 + 2.0 * V3 + V4)
        V = V + (hdt / 6.0) * (A1 + 2.0 * A2 + 2.0 * A3 + A4)
    return sunP, earthP, moonP, jupiterP


def build_ephemeris(duration, hdt):
    key = (duration, hdt)
    if key in _EPH:
        return _EPH[key]
    GMs = np.array([GM_SUN, GM_EARTH, GM_MOON, GM_JUPITER])
    masses = np.array([M_SUN, M_EARTH, M_MOON, M_JUPITER])
    es = _state1(EARTH_EL, GM_SUN)
    ms = _state1(MOON_EL, GM_EARTH)
    js = _state1(JUPITER_EL, GM_SUN)
    P = np.array([
    np.zeros(3),        # Sun
    es[:3],             # Earth
    es[:3] + ms[:3],    # Moon (Earth + Moon relative)
    es[:3] + js[:3]     # Jupiter (Earth + Jupiter relative)
    ])

    V = np.array([
    np.zeros(3),
    es[3:],             # Earth velocity
    es[3:] + ms[3:],    # Moon velocity
    es[3:] + js[3:]     # Jupiter velocity
    ])
    V -= (masses[:, None] * V).sum(0) / masses.sum()        # zero out barycentre drift
    moon0 = dict(pos=P[2].copy(), vel=V[2].copy())

    n = int(round(duration / hdt)) + 1
    if HAVE_NUMBA:
        sunP, earthP, moonP, jupiterP = _prop_bodies3(n, float(hdt), P.copy(), V.copy(), GMs)
    else:
        sunP = np.empty((n, 3)); earthP = np.empty((n, 3)); moonP = np.empty((n, 3)); jupiterP = np.empty((n, 3))
        for k in range(n):
            sunP[k], earthP[k], moonP[k], jupiterP[k] = P[0], P[1], P[2], P[3]
            if k == n - 1:
                break
            a1 = _accel_bodies(P, GMs)
            P2 = P + V * hdt / 2; V2 = V + a1 * hdt / 2
            a2 = _accel_bodies(P2, GMs)
            P3 = P + V2 * hdt / 2; V3 = V + a2 * hdt / 2
            a3 = _accel_bodies(P3, GMs)
            P4 = P + V3 * hdt; V4 = V + a3 * hdt
            a4 = _accel_bodies(P4, GMs)
            P = P + (hdt / 6) * (V + 2 * V2 + 2 * V3 + V4)
            V = V + (hdt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
    eph = dict(hdt=hdt, n=n, sunP=sunP, earthP=earthP, moonP=moonP, jupiterP=jupiterP, moon0=moon0)
    _EPH[key] = eph
    return eph


# sat propagation in the full n-body field + coverage
def decode(genes):
    a = A_MIN + genes[:, 0] * (A_MAX - A_MIN)
    e = genes[:, 1] * E_MAX
    i = genes[:, 2] * np.pi
    om = genes[:, 3] * 2 * np.pi
    w = genes[:, 4] * 2 * np.pi
    M = genes[:, 5] * 2 * np.pi
    return np.stack([a, e, i, om, w, M], axis=1)


def initial_states(el, eph):
    """lunar-equatorial elements -> ecliptic-frame sat states (M,6)."""
    loc = elements_to_state(el, GM_MOON)
    rp = rotX(loc[:, :3], MOON_TILT)
    vp = rotX(loc[:, 3:], MOON_TILT)
    r0 = rp + eph["moon0"]["pos"][None, :]
    v0 = vp + eph["moon0"]["vel"][None, :]
    return np.concatenate([r0, v0], axis=1)


def _accel_sat(R, idx, eph):
    A = np.zeros_like(R)
    for gm, P in ((GM_SUN, eph["sunP"]), (GM_EARTH, eph["earthP"]), (GM_MOON, eph["moonP"]), (GM_JUPITER, eph["jupiterP"])):
        D = P[idx][None, :] - R
        r2 = np.einsum("ij,ij->i", D, D)
        A += (gm / (np.sqrt(r2) * r2))[:, None] * D
    return A


def integrate_visibility(states0, eph, dt, sin_min, steps):
    # pure-numpy propagation, kept for the final verification and as the
    # no-numba fallback. returns per-sat (visibility, crashed-mask).
    R = states0[:, :3].copy()
    V = states0[:, 3:].copy()
    M = R.shape[0]
    vis = np.zeros((M, steps + 1), dtype=bool)
    alive = np.ones(M, dtype=bool)                 # a sat that crashes stops counting
    with np.errstate(all="ignore"):                # dead sats may go inf/nan; masked out anyway
        for k in range(steps + 1):
            ei = 2 * k
            # deorbit check: hit parent object, went non-finite, or got flung out
            Dm = R - eph["moonP"][ei][None, :]
            dm = np.sqrt(np.einsum("ij,ij->i", Dm, Dm))
            r_orig = np.sqrt(np.einsum("ij,ij->i", R, R))
            alive &= np.isfinite(dm) & (dm >= R_MOON) & (r_orig < ESCAPE)
            pole = eph["moonP"][ei] + SOUTH_DIR * R_MOON
            D = R - pole[None, :]
            d = np.sqrt(np.einsum("ij,ij->i", D, D))
            sin_el = (D @ SOUTH_DIR) / np.maximum(d, 1e-3)
            vis[:, k] = (sin_el >= sin_min) & alive
            if k == steps:
                break
            a1 = _accel_sat(R, ei, eph)
            R2 = R + V * dt / 2; V2 = V + a1 * dt / 2
            a2 = _accel_sat(R2, ei + 1, eph)
            R3 = R + V2 * dt / 2; V3 = V + a2 * dt / 2
            a3 = _accel_sat(R3, ei + 1, eph)
            R4 = R + V3 * dt; V4 = V + a3 * dt
            a4 = _accel_sat(R4, ei + 2, eph)
            R = R + (dt / 6) * (V + 2 * V2 + 2 * V3 + V4)
            V = V + (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
    return vis, ~alive                             # (visibility, crashed-mask)


# --- numba kernels: whole-population propagation + coverage -------------------
# one genome per parallel task. each integrates its own satellites over the full
# window and folds down to coverage / max-gap / avg-in-view / deorbit-count, so
# there's no giant M x steps visibility matrix and no cross-task synchronisation.
@njit
def _sat_acc(x, y, z, idx, sunP, earthP, moonP, jupiterP):
    dx = sunP[idx, 0] - x; dy = sunP[idx, 1] - y; dz = sunP[idx, 2] - z
    r2 = dx * dx + dy * dy + dz * dz; g = GM_SUN / (r2 * math.sqrt(r2))
    ax = g * dx; ay = g * dy; az = g * dz
    dx = earthP[idx, 0] - x; dy = earthP[idx, 1] - y; dz = earthP[idx, 2] - z
    r2 = dx * dx + dy * dy + dz * dz; g = GM_EARTH / (r2 * math.sqrt(r2))
    ax += g * dx; ay += g * dy; az += g * dz
    dx = moonP[idx, 0] - x; dy = moonP[idx, 1] - y; dz = moonP[idx, 2] - z
    r2 = dx * dx + dy * dy + dz * dz; g = GM_MOON / (r2 * math.sqrt(r2))
    ax += g * dx; ay += g * dy; az += g * dz
    dx = jupiterP[idx, 0] - x; dy = jupiterP[idx, 1] - y; dz = jupiterP[idx, 2] - z
    r2 = dx * dx + dy * dy + dz * dz; g = GM_JUPITER / (r2 * math.sqrt(r2))
    ax += g * dx; ay += g * dy; az += g * dz
    return ax, ay, az


@njit(parallel=True)
def _eval_pop_kernel(states0, ns, sunP, earthP, moonP, jupiterP, dt, sin_min,
                     R_moon, escape, south, steps):
    P = states0.shape[0]
    max_n = states0.shape[1]
    cov = np.zeros(P); gap = np.zeros(P); avgv = np.zeros(P)
    deo = np.zeros(P, dtype=np.int64)
    s0 = south[0]; s1 = south[1]; s2 = south[2]
    esc2 = escape * escape
    Rm2 = R_moon * R_moon
    hdt = dt * 0.5
    dt6 = dt / 6.0
    T = steps + 1
    for p in prange(P):
        n = ns[p]
        rx = np.empty(max_n); ry = np.empty(max_n); rz = np.empty(max_n)
        vx = np.empty(max_n); vy = np.empty(max_n); vz = np.empty(max_n)
        alive = np.ones(max_n, dtype=np.bool_)
        for j in range(n):
            rx[j] = states0[p, j, 0]; ry[j] = states0[p, j, 1]; rz[j] = states0[p, j, 2]
            vx[j] = states0[p, j, 3]; vy[j] = states0[p, j, 4]; vz[j] = states0[p, j, 5]
        covered = 0; sumv = 0.0; curgap = 0.0; maxgap = 0.0; ndeo = 0
        for k in range(T):
            ei = 2 * k
            mpx = moonP[ei, 0]; mpy = moonP[ei, 1]; mpz = moonP[ei, 2]
            plx = mpx + s0 * R_moon; ply = mpy + s1 * R_moon; plz = mpz + s2 * R_moon
            nvis = 0
            for j in range(n):
                if not alive[j]:
                    continue
                x = rx[j]; y = ry[j]; z = rz[j]
                ddx = x - mpx; ddy = y - mpy; ddz = z - mpz
                dm2 = ddx * ddx + ddy * ddy + ddz * ddz
                ro2 = x * x + y * y + z * z
                if dm2 != dm2 or dm2 <= Rm2 or ro2 > esc2:   # nan / crashed / flung
                    alive[j] = False; ndeo += 1; continue
                ex = x - plx; ey = y - ply; ez = z - plz
                d = math.sqrt(ex * ex + ey * ey + ez * ez)
                if d < 1e-3:
                    d = 1e-3
                if (ex * s0 + ey * s1 + ez * s2) / d >= sin_min:
                    nvis += 1
            sumv += nvis
            if nvis > 0:
                covered += 1; curgap = 0.0
            else:
                curgap += dt
                if curgap > maxgap:
                    maxgap = curgap
            if k == steps:
                break
            for j in range(n):
                if not alive[j]:
                    continue
                x = rx[j]; y = ry[j]; z = rz[j]; ax0 = vx[j]; ay0 = vy[j]; az0 = vz[j]
                a1x, a1y, a1z = _sat_acc(x, y, z, ei, sunP, earthP, moonP, jupiterP)
                x2 = x + ax0 * hdt; y2 = y + ay0 * hdt; z2 = z + az0 * hdt
                v2x = ax0 + a1x * hdt; v2y = ay0 + a1y * hdt; v2z = az0 + a1z * hdt
                a2x, a2y, a2z = _sat_acc(x2, y2, z2, ei + 1, sunP, earthP, moonP, jupiterP)
                x3 = x + v2x * hdt; y3 = y + v2y * hdt; z3 = z + v2z * hdt
                v3x = ax0 + a2x * hdt; v3y = ay0 + a2y * hdt; v3z = az0 + a2z * hdt
                a3x, a3y, a3z = _sat_acc(x3, y3, z3, ei + 1, sunP, earthP, moonP, jupiterP)
                x4 = x + v3x * dt; y4 = y + v3y * dt; z4 = z + v3z * dt
                v4x = ax0 + a3x * dt; v4y = ay0 + a3y * dt; v4z = az0 + a3z * dt
                a4x, a4y, a4z = _sat_acc(x4, y4, z4, ei + 2, sunP, earthP, moonP, jupiterP)
                rx[j] = x + dt6 * (ax0 + 2.0 * v2x + 2.0 * v3x + v4x)
                ry[j] = y + dt6 * (ay0 + 2.0 * v2y + 2.0 * v3y + v4y)
                rz[j] = z + dt6 * (az0 + 2.0 * v2z + 2.0 * v3z + v4z)
                vx[j] = ax0 + dt6 * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
                vy[j] = ay0 + dt6 * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
                vz[j] = az0 + dt6 * (a1z + 2.0 * a2z + 2.0 * a3z + a4z)
        cov[p] = covered / float(T); avgv[p] = sumv / float(T)
        gap[p] = maxgap; deo[p] = ndeo
    return cov, gap, avgv, deo


def _pop_stats_numpy(states0, ns, eph, dt, sin_min, steps, P, max_n):
    # fallback: run the numpy propagator and fold it down the same way the kernel does
    vis, crashed = integrate_visibility(states0, eph, dt, sin_min, steps)
    vis = vis.reshape(P, max_n, -1)
    crashed = crashed.reshape(P, max_n)
    active = (np.arange(max_n)[None, :] < ns[:, None])
    nvis = (vis * active[:, :, None]).sum(axis=1)
    covered = nvis > 0
    cov = covered.mean(axis=1)
    avgv = nvis.mean(axis=1)
    gap = np.array([longest_gap(covered[p], dt) for p in range(P)], dtype=np.float64)
    deo = (crashed & active).sum(axis=1)
    return cov, gap, avgv, deo


def longest_gap(covered, dt):
    gap = cur = 0.0
    for c in covered:
        cur = 0.0 if c else cur + dt
        gap = max(gap, cur)
    return gap


# fitness (kept matching src/optimizer.js)
def fitness_from_stats(el, n, stats):
    gap_h = stats["max_gap"] / 3600.0
    nmm = np.sqrt(GM_MOON / el[:, 0] ** 3)
    p = el[:, 0] * (1 - el[:, 1] ** 2)
    f = 1.5 * nmm * 2.0330530e-4 * (R_MOON / p) ** 2
    dw = f * (2 - 2.5 * np.sin(el[:, 2]) ** 2)
    drift_pen = np.minimum(1.0, np.abs(dw) * RAD * DAY / 5.0)
    peri_alt = el[:, 0] * (1 - el[:, 1]) - R_MOON
    alt_pen = np.where(peri_alt < MIN_PERI_ALT,
                       np.minimum(1.0, (MIN_PERI_ALT - peri_alt) / MIN_PERI_ALT), 0.0)
    apo_alt = el[:, 0] * (1 + el[:, 1]) - R_MOON
    apo_pen = np.maximum(0.0, (apo_alt - APO_SOFT) / APO_SOFT)   # nudge away from chaos
    lon = float(np.mean(drift_pen + alt_pen + apo_pen))
    hard = int(np.sum(peri_alt < MIN_PERI_ALT))
    cmplx = 0.0
    if n > 1:
        cmplx = float(min(2.0, np.std(el[:, 0] / 1e6) / 5 +
                          np.std(el[:, 2] * RAD) / 30 + np.std(el[:, 1]) / 0.3))
    constant = stats["coverage"] >= 0.999 and gap_h < 0.1
    deorb = int(stats.get("deorbits", 0))
    F = (100 * stats["coverage"] - 8 * n - 0.5 * gap_h - 5 * lon - 3 * cmplx +
         (60 if constant else 0) - 300 * hard - 300 * deorb)
    return F


def evaluate_population(genomes, max_n, eph, dt=DT):
    P = len(genomes)
    sin_min = float(np.sin(MIN_ELEV))
    steps = int(round((eph["n"] - 1) / 2))
    all_genes = np.array([g[1].reshape(max_n, GENES) for g in genomes]).reshape(-1, GENES)
    el_all = decode(all_genes)
    states0 = initial_states(el_all, eph)
    ns = np.array([g[0] for g in genomes], dtype=np.int64)
    if HAVE_NUMBA:
        s0 = np.ascontiguousarray(states0.reshape(P, max_n, 6))
        cov, gap, avgv, deo = _eval_pop_kernel(
            s0, ns,
            np.ascontiguousarray(eph["sunP"]), np.ascontiguousarray(eph["earthP"]),
            np.ascontiguousarray(eph["moonP"]), np.ascontiguousarray(eph["jupiterP"]),
            float(dt), sin_min, float(R_MOON), float(ESCAPE),
            np.ascontiguousarray(SOUTH_DIR), int(steps))
    else:
        cov, gap, avgv, deo = _pop_stats_numpy(states0, ns, eph, dt, sin_min, steps, P, max_n)
    el_r = el_all.reshape(P, max_n, GENES)
    out = []
    for p in range(P):
        n = int(ns[p])
        stats = dict(coverage=float(cov[p]), max_gap=float(gap[p]),
                     avg_visible=float(avgv[p]), deorbits=int(deo[p]))
        out.append((fitness_from_stats(el_r[p, :n], n, stats), stats))
    return out


# numpy MLP surrogate
class MLP:
    def __init__(self, n_in, hidden=(64, 64), lr=2e-3, seed=0):
        r = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [r.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.lr = lr
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]; self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _fwd(self, X):
        acts, a = [X], X
        for k in range(len(self.W) - 1):
            a = np.tanh(a @ self.W[k] + self.b[k]); acts.append(a)
        acts.append(a @ self.W[-1] + self.b[-1]); return acts

    def fit(self, X, y, epochs=60, batch=64):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-8
        self.ymu, self.ysd = y.mean(), y.std() + 1e-8
        Xs = (X - self.xmu) / self.xsd; ys = ((y - self.ymu) / self.ysd).reshape(-1, 1)
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                self._step(Xs[idx[s:s + batch]], ys[idx[s:s + batch]])

    def _step(self, X, y):
        acts = self._fwd(X); gW = [None] * len(self.W); gb = [None] * len(self.b)
        delta = (acts[-1] - y) / len(X)
        for k in reversed(range(len(self.W))):
            gW[k] = acts[k].T @ delta; gb[k] = delta.sum(0)
            if k > 0:
                delta = (delta @ self.W[k].T) * (1 - acts[k] ** 2)
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        for k in range(len(self.W)):
            for Mm, Vv, g, Pr in ((self.mW, self.vW, gW, self.W), (self.mb, self.vb, gb, self.b)):
                Mm[k] = b1 * Mm[k] + (1 - b1) * g[k]; Vv[k] = b2 * Vv[k] + (1 - b2) * g[k] ** 2
                Pr[k] -= self.lr * (Mm[k] / (1 - b1 ** self.t)) / (np.sqrt(Vv[k] / (1 - b2 ** self.t)) + eps)

    def predict(self, X):
        return self._fwd((X - self.xmu) / self.xsd)[-1].ravel() * self.ysd + self.ymu


def features(g, max_n):
    return np.concatenate([[g[0] / max_n], g[1]])


# genetic algorithm (surrogate-assisted)
def random_genome(max_n):
    return [int(rng.integers(1, max_n + 1)), rng.random(max_n * GENES)]


def crossover(p1, p2, max_n):
    n = p1[0] if rng.random() < 0.5 else p2[0]
    return [n, np.where(rng.random(max_n * GENES) < 0.5, p1[1], p2[1])]


def mutate(g, max_n, rate=0.18, sigma=0.12):
    n, flat = g
    if rng.random() < 0.12:
        n = int(np.clip(n + (1 if rng.random() < 0.5 else -1), 1, max_n))
    flat = flat.copy(); m = rng.random(flat.size) < rate
    flat[m] = np.clip(flat[m] + rng.standard_normal(m.sum()) * sigma, 0, 1)
    return [n, flat]


def tournament(scored, k=4):
    idx = rng.integers(0, len(scored), size=k)
    return max((scored[i] for i in idx), key=lambda s: s[1])[0]


def optimise(max_n=8, pop_size=140, generations=140, verbose=True):
    eph = build_ephemeris(DURATION, DT / 2)
    pop = [random_genome(max_n) for _ in range(pop_size)]
    ev = evaluate_population(pop, max_n, eph)
    scored = [(pop[i], ev[i][0], ev[i][1]) for i in range(pop_size)]
    surr = MLP(1 + max_n * GENES, seed=1)
    buf_X, buf_y = [], []
    best = max(scored, key=lambda s: s[1])
    history = []
    for gen in range(generations):
        scored.sort(key=lambda s: s[1], reverse=True)
        if scored[0][1] > best[1]:
            best = scored[0]
        for g, F, _ in scored:
            buf_X.append(features(g, max_n)); buf_y.append(F)
        use_surr = len(buf_X) >= 500
        if use_surr:
            surr.fit(np.array(buf_X[-5000:]), np.array(buf_y[-5000:]))
        elite = [s[0] for s in scored[:max(2, pop_size // 10)]]
        need = pop_size - len(elite)
        pool = [mutate(crossover(tournament(scored), tournament(scored), max_n), max_n)
                for _ in range(need * (5 if use_surr else 1))]
        if use_surr:
            pred = surr.predict(np.array([features(c, max_n) for c in pool]))
            pool = [pool[i] for i in np.argsort(pred)[::-1][:need]]
        else:
            pool = pool[:need]
        children = elite + pool
        ev = evaluate_population(children, max_n, eph)
        scored = [(children[i], ev[i][0], ev[i][1]) for i in range(len(children))]
        cov = best[2]["coverage"]
        history.append((gen, best[1], cov, best[0][0]))
        if verbose and gen % 10 == 0:
            print(f"  gen {gen:3d} | best F={best[1]:7.2f} | coverage={cov*100:6.2f}% "
                  f"| sats={best[0][0]} | gap={best[2]['max_gap']/3600:5.2f}h "
                  f"| surrogate={'on' if use_surr else 'off'}")
        if cov >= 0.9999 and best[2]["max_gap"] < DT * 1.5 and gen > 60:
            if verbose:
                print(f"  converged at generation {gen}")
            break
    return best, history


# main
def main():
    print("=" * 72)
    print(" Lunar South-Pole Constellation Optimizer")
    print(" GA + numpy MLP surrogate  ·  Full Sun+Earth+Moon N-body")
    if HAVE_NUMBA:
        print(f" Backend: Numba, {get_num_threads()} threads  (compiling kernels on first call...)")
    else:
        print(" Backend: Numpy fallback (single-core). Run 'pip install numba' for a huge speedup.")
    print("=" * 72)
    t0 = time.time()
    best, history = optimise()
    genome, F, _ = best
    n = genome[0]
    el = decode(genome[1].reshape(-1, GENES)[:n])

    # re-check over the full 2-year window at a finer step (2.5-min) than the
    # GA's 5-min, and with the independent numpy propagator, so the reported
    # numbers aren't just the search's own scoring played back to itself.
    verify_dur = 2 * YEAR
    eph = build_ephemeris(verify_dur, 75.0)
    states0 = initial_states(el, eph)
    steps = int(round((eph["n"] - 1) / 2))
    vis, crashed = integrate_visibility(states0, eph, 150.0, np.sin(MIN_ELEV), steps)
    nvis = vis.sum(axis=0); covered = nvis > 0
    coverage = covered.mean(); max_gap = longest_gap(covered, 150.0); avg_vis = nvis.mean()
    deorbits = int(crashed.sum())

    print("\n" + "-" * 72)
    print(f" BEST CONSTELLATION  ({n} satellites)   fitness = {F:.2f}")
    print(f"   [verified: 2-year full Sun+Earth+Moon propagation, 2.5-min step]")
    print(f"   coverage    : {coverage*100:.3f} %  of the time")
    print(f"   max outage  : {max_gap/3600:.2f} h")
    print(f"   avg in view : {avg_vis:.2f} satellites")
    print(f"   deorbits    : {deorbits}  (satellites lost over the 2 years)")
    print("-" * 72)
    print(f"   {'#':>2} {'a(km)':>9} {'alt_peri':>9} {'alt_apo':>9} {'e':>6} "
          f"{'i(°)':>7} {'Ω(°)':>6} {'ω(°)':>6} {'T(h)':>6}")
    if cf.CNS == True:
        winsound.PlaySound("ding.wav", winsound.SND_FILENAME)
    sats = []
    for k in range(n):
        a, e, i, om, w, M = el[k]
        T = 2 * np.pi * np.sqrt(a ** 3 / GM_MOON)
        print(f"   {k+1:>2} {a/1e3:9.1f} {(a*(1-e)-R_MOON)/1e3:9.1f} "
              f"{(a*(1+e)-R_MOON)/1e3:9.1f} {e:6.3f} {i*RAD:7.2f} {om*RAD:6.1f} "
              f"{w*RAD:6.1f} {T/3600:6.2f}")
        sats.append(dict(a=float(a), e=float(e), i=float(i), om=float(om), w=float(w), M=float(M)))

    out = dict(meta=dict(
        description="Optimised lunar south-pole constellation",
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        frame="Moon-centred lunar-equatorial elements; SI units / radians",
        dynamics="Full Sun+Earth+Moon point-mass N-body, RK4 (matches sandbox)",
        coverage=float(coverage), max_gap_s=float(max_gap), avg_visible=float(avg_vis),
        n_sats=n, fitness=float(F), deorbits=deorbits, min_elevation_deg=MIN_ELEV * RAD,
        verify_days=round(verify_dur / DAY)),
        satellites=sats)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "best_constellation.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(here, "training_history.csv"), "w") as f:
        f.write("generation,best_fitness,coverage,n_sats\n")
        for g, ff, c, ns in history:
            f.write(f"{g},{ff:.4f},{c:.6f},{ns}\n")

    print("-" * 72)
    print(f" Wrote {os.path.join(here, 'best_constellation.json')}")
    print(f" Done in {time.time()-t0:.1f}s - load it with 'Import Python result' in the sandbox.")


if __name__ == "__main__":
    main()
