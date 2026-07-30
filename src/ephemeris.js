// ephemeris.js - shared sun/earth/moon starting state and trajectory.
// the sandbox and the coverage model both build the system from the SAME real
// j2000 elements and integrate the SAME point-mass field, so an orbit that the
// coverage model likes behaves the same way once it's flying in the live sandbox.

import { SUN, EARTH, MOON, JUPITER } from './constants.js';
import { elementsToState } from './orbital.js';
import { NBody, zeroBarycentreDrift } from './physics.js';

const meta = (b) => ({ GM: b.GM, name: b.name, radius: b.radius, color: b.color,
  rotationPeriod: b.rotationPeriod, axialTilt: b.axialTilt });
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

// build the initial physical state from the real keplerian elements
export function buildSystem() {
  const sim = new NBody();
  const sun = sim.addBody({ ...meta(SUN), pos: [0, 0, 0], vel: [0, 0, 0] });

  const eState = elementsToState(EARTH.elements, SUN.GM);
  const earth = sim.addBody({ ...meta(EARTH),
    pos: eState.pos.slice(), vel: eState.vel.slice() });

  const mState = elementsToState(MOON.elements, EARTH.GM);
  const moon = sim.addBody({ ...meta(MOON),
    pos: add(earth.pos, mState.pos), vel: add(earth.vel, mState.vel) });
  
  const jState = elementsToState(JUPITER.elements, SUN.GM)
  const jupiter = sim.addBody({ ...meta(JUPITER),
    pos: jState.pos.slice(), vel: jState.vel.slice() });

  zeroBarycentreDrift(sim.bodies, [SUN.mass, EARTH.mass, MOON.mass, JUPITER.mass]);
  return { sim, sun, earth, moon, jupiter };
}

// precompute sun/earth/moon positions on a fixed grid (step = hdt), plus the
// moon's start state. cached, because the big-body path is the same for every
// candidate constellation and every GA generation - no point redoing it.
const _cache = new Map();
export function computeEphemeris(duration, hdt) {
  const key = `${duration}|${hdt}`;
  if (_cache.has(key)) return _cache.get(key);

  const { sim, sun, earth, moon, jupiter } = buildSystem();
  const moon0 = { pos: moon.pos.slice(), vel: moon.vel.slice() };
  const n = Math.round(duration / hdt) + 1;
  const sunP = new Float64Array(n * 3);
  const earthP = new Float64Array(n * 3);
  const moonP = new Float64Array(n * 3);
  const jupiterP = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    sunP[k * 3] = sun.pos[0]; sunP[k * 3 + 1] = sun.pos[1]; sunP[k * 3 + 2] = sun.pos[2];
    earthP[k * 3] = earth.pos[0]; earthP[k * 3 + 1] = earth.pos[1]; earthP[k * 3 + 2] = earth.pos[2];
    moonP[k * 3] = moon.pos[0]; moonP[k * 3 + 1] = moon.pos[1]; moonP[k * 3 + 2] = moon.pos[2];
    jupiterP[k * 3] = jupiter.pos[0]; jupiterP[k * 3 + 1] = jupiter.pos[1]; jupiterP[k * 3 + 2] = jupiter.pos[2];
    if (k < n - 1) sim.step(hdt);
  }
  const eph = { duration, hdt, n, sunP, earthP, moonP, jupiterP, moon0 };
  _cache.set(key, eph);
  return eph;
}
