// optimizer.js - the in-browser genetic algorithm for lunar constellations.
// evolves a fleet of moon-orbiting sats toward CONSTANT view of the south pole
// while keeping down:
//   - number of sats     (cost / how simple it is)
//   - orbit drift         (longevity, station-keeping fuel)
//   - spread of orbits    (operational hassle)
// the genome length is itself a gene, so the search finds the smallest fleet
// that still covers the pole. runs one generation per call so the ui can play
// the training back live.
//
// evolutionary search is the usual ml tool for this kind of thing (big,
// non-convex, mixed integer/continuous space). the python pipeline does the same
// but adds a neural surrogate trained on this exact fitness landscape.

import { GM_MOON, MOON, RAD } from './constants.js';
import { evaluateConstellation, j2Rates } from './coverage.js';

const TAU = Math.PI * 2;

// decode bounds for one sat's normalised [0,1] genes.
// apoapsis is capped well inside the perturbed outer zone on purpose: high
// apoapsis lunar orbits go chaotic under earth's pull, so their "coverage" is
// really just an integrator artefact, not a design you can trust. staying low
// keeps the result reproducible back in the live sandbox.
const A_MIN = MOON.radius + 150e3;        // >=150 km altitude
const A_MAX = MOON.radius + 8000e3;
const E_MAX = 0.55;

function decodeSat(g) {
  return {
    a:  A_MIN + g[0] * (A_MAX - A_MIN),
    e:  g[1] * E_MAX,
    i:  g[2] * Math.PI,                    // 0..180°
    om: g[3] * TAU,
    w:  g[4] * TAU,
    M:  g[5] * TAU,
  };
}

const GENES_PER_SAT = 6;

// fitness. higher is better. tuned so adding another sat only pays off when it
// actually buys coverage or closes a real gap.
export function fitness(genome, opts) {
  const sats = genome.sats.slice(0, genome.n).map(decodeSat);
  const stats = evaluateConstellation(sats, opts.coverage);

  const gapHours = stats.maxGap / 3600;

  // longevity + safety penalty (averaged over the sats), plus a HARD reject for
  // any orbit whose periapsis dips down into / below the lunar surface.
  let lon = 0, hard = 0;
  for (const s of sats) {
    const r = j2Rates(s.a, s.e, s.i);
    const dwDegPerDay = Math.abs(r.dw) * RAD * 86400;
    const driftPen = Math.min(1, dwDegPerDay / 5);          // frozen orbit -> 0
    const periAlt = s.a * (1 - s.e) - MOON.radius;
    const altPen  = periAlt < 100e3 ? Math.min(1, (100e3 - periAlt) / 100e3) : 0;
    const apoAlt  = s.a * (1 + s.e) - MOON.radius;
    const apoPen  = Math.max(0, (apoAlt - 9000e3) / 9000e3);
    if (periAlt < 100e3) hard++;
    lon += driftPen + altPen + apoPen;
  }
  lon /= Math.max(1, sats.length);

  // complexity penalty: how spread out the orbit shapes are. a walker-style set
  // of identical orbits is the cheapest to build, launch and fly.
  let cmplx = 0;
  if (sats.length > 1) {
    const mean = (f) => sats.reduce((s, x) => s + f(x), 0) / sats.length;
    const std  = (f, m) => Math.sqrt(sats.reduce((s, x) => s + (f(x) - m) ** 2, 0) / sats.length);
    const ma = mean(s => s.a / 1e6), mi = mean(s => s.i * RAD), me = mean(s => s.e);
    cmplx = std(s => s.a / 1e6, ma) / 5 + std(s => s.i * RAD, mi) / 30 + std(s => s.e, me) / 0.3;
    cmplx = Math.min(2, cmplx);
  }

  const constant = stats.coverage >= 0.999 && gapHours < 0.1;

  const F =
      100 * stats.coverage          // the main thing: fraction of time the pole is covered
    - 8   * genome.n                 // cost: fewer sats is better
    - 0.5 * gapHours                 // shave off outages
    - 5   * lon                      // longevity + don't crash into the moon
    - 3   * cmplx                    // keep it simple to operate
    + (constant ? 60 : 0)            // little bonus for genuine 24/7 coverage
    - 300 * hard                     // hard no on sub-surface periapsis
    - 300 * (stats.deorbited || 0);  // hard no on anything that actually deorbits mid-run

  return { F, stats, lon, cmplx };
}

// genome helpers
function randomGenome(maxN) {
  const n = 1 + Math.floor(Math.random() * maxN);
  const sats = [];
  for (let i = 0; i < maxN; i++) {
    const g = new Array(GENES_PER_SAT);
    for (let k = 0; k < GENES_PER_SAT; k++) g[k] = Math.random();
    sats.push(g);
  }
  return { n, sats };
}

function cloneGenome(gn) {
  return { n: gn.n, sats: gn.sats.map(s => s.slice()) };
}

export class GAOptimizer {
  constructor(opts = {}) {
    this.maxN       = opts.maxN ?? 8;
    this.popSize    = opts.popSize ?? 48;
    this.eliteFrac  = opts.eliteFrac ?? 0.12;
    this.mutRate    = opts.mutRate ?? 0.18;
    this.mutSigma   = opts.mutSigma ?? 0.12;
    this.opts       = {
      coverage: {
        // full sun+earth+moon rk4 propagation (src/coverage.js). this live
        // trainer uses an 18-day window so it stays snappy; the offline python
        // run does a full 2-year window and is the one to actually trust.
        duration:   opts.duration   ?? 18 * 86400,
        dt:         opts.dt         ?? 1500,
        minElevDeg: opts.minElevDeg ?? 5,
      },
    };
    this.generation = 0;
    this.population = Array.from({ length: this.popSize }, () => randomGenome(this.maxN));
    this.history    = [];
    this.best       = null;
    this._score();
  }

  _score() {
    for (const gn of this.population) {
      const r = fitness(gn, this.opts);
      gn._f = r.F; gn._stats = r.stats;
    }
    this.population.sort((a, b) => b._f - a._f);
    const top = this.population[0];
    if (!this.best || top._f > this.best._f) this.best = cloneGenomeWithMeta(top);
  }

  // run one generation, hand back a summary of the current best
  step() {
    const elite = Math.max(1, Math.floor(this.popSize * this.eliteFrac));
    const next = this.population.slice(0, elite).map(cloneGenome);

    while (next.length < this.popSize) {
      const p1 = this._tournament(), p2 = this._tournament();
      next.push(this._mutate(this._crossover(p1, p2)));
    }
    this.population = next;
    this._score();
    this.generation++;

    const b = this.population[0];
    this.history.push({ gen: this.generation, fitness: this.best._f, coverage: this.best._stats.coverage, n: this.best.n });
    return this.summary();
  }

  _tournament(k = 4) {
    let best = null;
    for (let i = 0; i < k; i++) {
      const c = this.population[Math.floor(Math.random() * this.population.length)];
      if (!best || c._f > best._f) best = c;
    }
    return best;
  }

  _crossover(p1, p2) {
    const child = { n: Math.random() < 0.5 ? p1.n : p2.n, sats: [] };
    for (let i = 0; i < this.maxN; i++) {
      const a = p1.sats[i], b = p2.sats[i], g = new Array(GENES_PER_SAT);
      for (let k = 0; k < GENES_PER_SAT; k++) g[k] = Math.random() < 0.5 ? a[k] : b[k];
      child.sats.push(g);
    }
    return child;
  }

  _mutate(gn) {
    // sometimes bump the sat count up or down
    if (Math.random() < 0.12) {
      gn.n = Math.max(1, Math.min(this.maxN, gn.n + (Math.random() < 0.5 ? -1 : 1)));
    }
    for (const s of gn.sats) {
      for (let k = 0; k < GENES_PER_SAT; k++) {
        if (Math.random() < this.mutRate) {
          s[k] += gaussian() * this.mutSigma;
          s[k] = Math.max(0, Math.min(1, s[k]));
        }
      }
    }
    return gn;
  }

  // best constellation, decoded to real orbital elements
  bestConstellation() {
    return this.best.sats.slice(0, this.best.n).map(decodeSat);
  }

  summary() {
    return {
      generation: this.generation,
      fitness: this.best._f,
      stats: this.best._stats,
      nSats: this.best.n,
      elements: this.bestConstellation(),
    };
  }
}

function cloneGenomeWithMeta(gn) {
  const c = cloneGenome(gn);
  c._f = gn._f; c._stats = gn._stats;
  return c;
}

// box-muller standard normal
function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
}
