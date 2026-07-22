// orbital.js - keplerian elements <-> cartesian state vectors.
// pos/vel come back in the parent-centred inertial frame whose xy plane is the
// j2000 ecliptic (same frame the element sets use). radians, metres, mu = GM in SI.

const TAU = Math.PI * 2;

// solve kepler's equation M = E - e*sin(E) for the eccentric anomaly E.
// plain newton-raphson; the starting guess copes with e up to ~0.97.
export function solveKepler(M, e, tol = 1e-12, maxIter = 60) {
  M = ((M % TAU) + TAU) % TAU;
  let E = e < 0.8 ? M : Math.PI;
  for (let k = 0; k < maxIter; k++) {
    const f = E - e * Math.sin(E) - M;
    const fp = 1 - e * Math.cos(E);
    const dE = f / fp;
    E -= dE;
    if (Math.abs(dE) < tol) break;
  }
  return E;
}

// true anomaly from eccentric anomaly
export function trueAnomaly(E, e) {
  const s = Math.sqrt(1 - e * e) * Math.sin(E);
  const c = Math.cos(E) - e;
  return Math.atan2(s, c);
}

// rotate a vector out of the perifocal (pqw) frame into the parent inertial
// frame, 3-1-3 euler sequence (Ω, i, ω).
function perifocalToInertial(vp, om, i, w) {
  const co = Math.cos(om), so = Math.sin(om);
  const ci = Math.cos(i),  si = Math.sin(i);
  const cw = Math.cos(w),  sw = Math.sin(w);

  // combined rotation matrix R = Rz(om) * Rx(i) * Rz(w)
  const R11 = co * cw - so * sw * ci;
  const R12 = -co * sw - so * cw * ci;
  const R13 = so * si;
  const R21 = so * cw + co * sw * ci;
  const R22 = -so * sw + co * cw * ci;
  const R23 = -co * si;
  const R31 = sw * si;
  const R32 = cw * si;
  const R33 = ci;

  return [
    R11 * vp[0] + R12 * vp[1] + R13 * vp[2],
    R21 * vp[0] + R22 * vp[1] + R23 * vp[2],
    R31 * vp[0] + R32 * vp[1] + R33 * vp[2],
  ];
}

// turn a classical element set {a,e,i,om,w,M} + mu into {pos, vel} (m, m/s) in
// the parent-centred ecliptic inertial frame.
export function elementsToState(el, mu) {
  const { a, e, i, om, w, M } = el;
  const E = solveKepler(M, e);
  const nu = trueAnomaly(E, e);

  const p = a * (1 - e * e);             // semi-latus rectum
  const r = p / (1 + e * Math.cos(nu));  // radius

  // pos & vel in the perifocal frame first
  const rp = [r * Math.cos(nu), r * Math.sin(nu), 0];
  const vc = Math.sqrt(mu / p);
  const vp = [-vc * Math.sin(nu), vc * (e + Math.cos(nu)), 0];

  return {
    pos: perifocalToInertial(rp, om, i, w),
    vel: perifocalToInertial(vp, om, i, w),
  };
}

// mean motion n = sqrt(mu / a^3) [rad/s], period T = 2pi/n [s]
export const meanMotion = (a, mu) => Math.sqrt(mu / (a * a * a));
export const orbitalPeriod = (a, mu) => TAU / meanMotion(a, mu);

// push a mean anomaly forward by dt seconds
export function propagateMeanAnomaly(el, mu, dt) {
  const n = meanMotion(el.a, mu);
  return { ...el, M: el.M + n * dt };
}

// go the other way: parent-centred cartesian state back to classical elements
export function stateToElements(pos, vel, mu) {
  const [rx, ry, rz] = pos;
  const [vx, vy, vz] = vel;
  const r = Math.hypot(rx, ry, rz);
  const v2 = vx * vx + vy * vy + vz * vz;

  // specific angular momentum h = r x v
  const hx = ry * vz - rz * vy;
  const hy = rz * vx - rx * vz;
  const hz = rx * vy - ry * vx;
  const h = Math.hypot(hx, hy, hz);

  // node vector n = z_hat x h
  const nx = -hy, ny = hx, nz = 0;
  const nmag = Math.hypot(nx, ny, nz);

  // eccentricity vector
  const rv = rx * vx + ry * vy + rz * vz;
  const ex = (v2 / mu - 1 / r) * rx - (rv / mu) * vx;
  const ey = (v2 / mu - 1 / r) * ry - (rv / mu) * vy;
  const ez = (v2 / mu - 1 / r) * rz - (rv / mu) * vz;
  const e = Math.hypot(ex, ey, ez);

  const energy = v2 / 2 - mu / r;
  const a = -mu / (2 * energy);

  const i = Math.acos(Math.max(-1, Math.min(1, hz / h)));

  let om = nmag > 1e-12 ? Math.acos(Math.max(-1, Math.min(1, nx / nmag))) : 0;
  if (ny < 0) om = TAU - om;

  let w = 0;
  if (nmag > 1e-12 && e > 1e-12) {
    w = Math.acos(Math.max(-1, Math.min(1, (nx * ex + ny * ey + nz * ez) / (nmag * e))));
    if (ez < 0) w = TAU - w;
  }

  let nu = 0;
  if (e > 1e-12) {
    nu = Math.acos(Math.max(-1, Math.min(1, (ex * rx + ey * ry + ez * rz) / (e * r))));
    if (rv < 0) nu = TAU - nu;
  }

  const E = Math.atan2(Math.sqrt(1 - e * e) * Math.sin(nu), e + Math.cos(nu));
  const M = E - e * Math.sin(E);

  return { a, e, i, om, w, M: ((M % TAU) + TAU) % TAU };
}
