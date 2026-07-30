// coverage.js - how much of the time the lunar south pole can see a satellite,
// under the full n-body field.
// sats get propagated in the j2000 ecliptic frame under point-mass sun+earth+moon
// gravity - the exact field the live sandbox integrates - off a precomputed
// big-body ephemeris (ephemeris.js). same elements, same forces, so a
// constellation that scores well here holds its coverage once it's deployed.
//
// sat elements come in referenced to the lunar equator (so i = 90 deg is a true
// polar orbit) and get rotated into the ecliptic on the way in, same as the
// sandbox does for user sats. the south-pole ground point
// P(t) = MoonPos(t) + southDir*R_moon rides along with the moving moon.

import { GM_SUN, GM_EARTH, GM_MOON, GM_JUPITER, MOON, EARTH, DAY, JUPITER, DEG } from './constants.js';
import { elementsToState } from './orbital.js';
import { equatorialToEcliptic } from './frames.js';
import { computeEphemeris } from './ephemeris.js';

const moonEqToEcl = equatorialToEcliptic(MOON.axialTilt);
// south pole direction in the ecliptic frame (equator tilted about x)
const SOUTH_DIR = moonEqToEcl([0, 0, -1]);
// beyond this the sat has left the system - treat it as deorbited (matches sandbox)
const ESCAPE = 5 * EARTH.elements.a;

// accel on a sat at ecliptic position r from sun+earth+moon
function accel(r, idx, eph) {
  let ax = 0, ay = 0, az = 0;
  const bodies = [
    [GM_SUN, eph.sunP], [GM_EARTH, eph.earthP], [GM_MOON, eph.moonP] [GM_JUPITER, eph.jupiterP],
  ];
  for (const [gm, P] of bodies) {
    const dx = P[idx * 3] - r[0];
    const dy = P[idx * 3 + 1] - r[1];
    const dz = P[idx * 3 + 2] - r[2];
    const r2 = dx * dx + dy * dy + dz * dz;
    const g = gm / (Math.sqrt(r2) * r2);
    ax += g * dx; ay += g * dy; az += g * dz;
  }
  return [ax, ay, az];
}

// is the sat above the min elevation angle as seen from the moving south pole?
function visible(r, idx, eph, R, sinMin) {
  const mx = eph.moonP[idx * 3], my = eph.moonP[idx * 3 + 1], mz = eph.moonP[idx * 3 + 2];
  const px = mx + SOUTH_DIR[0] * R, py = my + SOUTH_DIR[1] * R, pz = mz + SOUTH_DIR[2] * R;
  const dx = r[0] - px, dy = r[1] - py, dz = r[2] - pz;
  const d = Math.hypot(dx, dy, dz);
  if (d < 1e-3) return false;
  const sinEl = (SOUTH_DIR[0] * dx + SOUTH_DIR[1] * dy + SOUTH_DIR[2] * dz) / d;
  return sinEl >= sinMin;
}

// starting ecliptic state of a sat from its lunar-equatorial elements
function initialState(el, eph) {
  const local = elementsToState(el, GM_MOON);
  const rp = moonEqToEcl(local.pos), vp = moonEqToEcl(local.vel);
  return {
    r: [eph.moon0.pos[0] + rp[0], eph.moon0.pos[1] + rp[1], eph.moon0.pos[2] + rp[2]],
    v: [eph.moon0.vel[0] + vp[0], eph.moon0.vel[1] + vp[1], eph.moon0.vel[2] + vp[2]],
  };
}

// score a constellation. opts: { duration [s], dt [s], minElevDeg }.
// the big-body ephemeris is sampled at dt/2 so the rk4 mid-points land exactly.
export function evaluateConstellation(sats, opts = {}) {
  const duration = opts.duration ?? 21 * DAY;
  const dt = opts.dt ?? 900;
  const sinMin = Math.sin((opts.minElevDeg ?? 5) * DEG);
  const R = MOON.radius;
  const hdt = dt / 2;
  const eph = computeEphemeris(duration, hdt);
  const steps = Math.round(duration / dt);

  const states = sats.map(el => { const s = initialState(el, eph); s.dead = false; return s; });

  let covered = 0, sumVisible = 0, curGap = 0, maxGap = 0, deorbited = 0;
  for (let k = 0; k <= steps; k++) {
    const ei = k * 2;                              // ephemeris index for time k*dt
    let nVis = 0;
    for (const st of states) {
      if (st.dead) continue;                       // crashed sats see nothing
      if (crashedOrGone(st.r, ei, eph, R)) { st.dead = true; deorbited++; continue; }
      if (visible(st.r, ei, eph, R, sinMin)) nVis++;
    }
    sumVisible += nVis;
    if (nVis > 0) { covered++; curGap = 0; }
    else { curGap += dt; if (curGap > maxGap) maxGap = curGap; }
    if (k < steps) for (const st of states) if (!st.dead) rk4(st, ei, eph, dt);
  }

  const n = steps + 1;
  return { coverage: covered / n, maxGap, avgVisible: sumVisible / n, deorbited,
           nSats: sats.length, minElevDeg: opts.minElevDeg ?? 5 };
}

// has this sat hit the lunar surface, gone non-finite, or been flung out of the
// system? if so it's deorbited and can no longer contribute coverage.
function crashedOrGone(r, idx, eph, R) {
  if (!isFinite(r[0]) || !isFinite(r[1]) || !isFinite(r[2])) return true;
  const mx = eph.moonP[idx * 3], my = eph.moonP[idx * 3 + 1], mz = eph.moonP[idx * 3 + 2];
  if (Math.hypot(r[0] - mx, r[1] - my, r[2] - mz) <= R) return true;
  return Math.hypot(r[0], r[1], r[2]) > ESCAPE;
}

// rk4 step. ephemeris indices: ei is t, ei+1 is t+dt/2, ei+2 is t+dt.
function rk4(st, ei, eph, dt) {
  const r = st.r, v = st.v;
  const a1 = accel(r, ei, eph);
  const r2 = [r[0] + v[0] * dt / 2, r[1] + v[1] * dt / 2, r[2] + v[2] * dt / 2];
  const v2 = [v[0] + a1[0] * dt / 2, v[1] + a1[1] * dt / 2, v[2] + a1[2] * dt / 2];
  const a2 = accel(r2, ei + 1, eph);
  const r3 = [r[0] + v2[0] * dt / 2, r[1] + v2[1] * dt / 2, r[2] + v2[2] * dt / 2];
  const v3 = [v[0] + a2[0] * dt / 2, v[1] + a2[1] * dt / 2, v[2] + a2[2] * dt / 2];
  const a3 = accel(r3, ei + 1, eph);
  const r4 = [r[0] + v3[0] * dt, r[1] + v3[1] * dt, r[2] + v3[2] * dt];
  const v4 = [v[0] + a3[0] * dt, v[1] + a3[1] * dt, v[2] + a3[2] * dt];
  const a4 = accel(r4, ei + 2, eph);
  for (let i = 0; i < 3; i++) {
    r[i] += (dt / 6) * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i]);
    v[i] += (dt / 6) * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]);
  }
}

// visible-count over time, for the ui sparkline
export function coverageTimeline(sats, opts = {}) {
  const duration = opts.duration ?? 21 * DAY;
  const dt = opts.dt ?? 1800;
  const sinMin = Math.sin((opts.minElevDeg ?? 5) * DEG);
  const R = MOON.radius;
  const eph = computeEphemeris(duration, dt / 2);
  const steps = Math.round(duration / dt);
  const states = sats.map(el => initialState(el, eph));
  const series = [];
  for (let k = 0; k <= steps; k++) {
    const ei = k * 2;
    let n = 0;
    for (const st of states) if (visible(st.r, ei, eph, R, sinMin)) n++;
    series.push(n);
    if (k < steps) for (const st of states) rk4(st, ei, eph, dt);
  }
  return { series, dt };
}

// secular j2 precession rates. we keep these for the longevity term in the fitness.
export function j2Rates(a, e, i) {
  const n = Math.sqrt(GM_MOON / (a * a * a));
  const p = a * (1 - e * e);
  const f = 1.5 * n * MOON.J2 * (MOON.refRadius / p) ** 2;
  const si2 = Math.sin(i) ** 2;
  return { n, dom: -f * Math.cos(i), dw: f * (2 - 2.5 * si2),
           dM: f * Math.sqrt(1 - e * e) * (1 - 1.5 * si2) };
}
