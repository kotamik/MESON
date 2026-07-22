// physics.js - the n-body gravity integrator (rk4, fixed step, SI units).
// the big bodies (sun/earth/moon) pull on each other. satellites are massless
// test particles: they feel every big body but push back on nothing. keeps the
// solar system stable while any number of spacecraft ride the real field.

export class NBody {
  constructor() {
    this.bodies = [];   // { GM, pos:[x,y,z], vel:[x,y,z], ...meta }
    this.sats   = [];   // { pos, vel, ...meta }
    this.time   = 0;    // seconds since epoch
  }

  addBody(b)      { this.bodies.push(b); return b; }
  addSatellite(s) { this.sats.push(s);   return s; }

  // accel of every body + satellite for one snapshot.
  // bp / sp are flat position arrays (length 3N). returns flat accel arrays.
  _accel(bp, sp) {
    const N = this.bodies.length, S = sp.length / 3;
    const ab = new Float64Array(bp.length);
    const as = new Float64Array(sp.length);

    // big bodies pulling on each other
    for (let i = 0; i < N; i++) {
      const ix = i * 3;
      for (let j = i + 1; j < N; j++) {
        const jx = j * 3;
        const dx = bp[jx] - bp[ix];
        const dy = bp[jx + 1] - bp[ix + 1];
        const dz = bp[jx + 2] - bp[ix + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        const inv = 1 / (Math.sqrt(r2) * r2);     // 1 / r^3
        const gi = this.bodies[i].GM * inv;
        const gj = this.bodies[j].GM * inv;
        ab[ix]     += gj * dx; ab[ix + 1] += gj * dy; ab[ix + 2] += gj * dz;
        ab[jx]     -= gi * dx; ab[jx + 1] -= gi * dy; ab[jx + 2] -= gi * dz;
      }
    }

    // satellites get pulled by every big body
    for (let s = 0; s < S; s++) {
      const sx = s * 3;
      let axx = 0, ayy = 0, azz = 0;
      for (let i = 0; i < N; i++) {
        const ix = i * 3;
        const dx = bp[ix] - sp[sx];
        const dy = bp[ix + 1] - sp[sx + 1];
        const dz = bp[ix + 2] - sp[sx + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        const g = this.bodies[i].GM / (Math.sqrt(r2) * r2);
        axx += g * dx; ayy += g * dy; azz += g * dz;
      }
      as[sx] = axx; as[sx + 1] = ayy; as[sx + 2] = azz;
    }
    return { ab, as };
  }

  // squash the current positions / velocities into flat arrays
  _pack() {
    const N = this.bodies.length, S = this.sats.length;
    const bp = new Float64Array(N * 3), bv = new Float64Array(N * 3);
    const sp = new Float64Array(S * 3), sv = new Float64Array(S * 3);
    for (let i = 0; i < N; i++) {
      const b = this.bodies[i], x = i * 3;
      bp[x] = b.pos[0]; bp[x + 1] = b.pos[1]; bp[x + 2] = b.pos[2];
      bv[x] = b.vel[0]; bv[x + 1] = b.vel[1]; bv[x + 2] = b.vel[2];
    }
    for (let i = 0; i < S; i++) {
      const s = this.sats[i], x = i * 3;
      sp[x] = s.pos[0]; sp[x + 1] = s.pos[1]; sp[x + 2] = s.pos[2];
      sv[x] = s.vel[0]; sv[x + 1] = s.vel[1]; sv[x + 2] = s.vel[2];
    }
    return { bp, bv, sp, sv };
  }

  _unpack(bp, bv, sp, sv) {
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i], x = i * 3;
      b.pos[0] = bp[x]; b.pos[1] = bp[x + 1]; b.pos[2] = bp[x + 2];
      b.vel[0] = bv[x]; b.vel[1] = bv[x + 1]; b.vel[2] = bv[x + 2];
    }
    for (let i = 0; i < this.sats.length; i++) {
      const s = this.sats[i], x = i * 3;
      s.pos[0] = sp[x]; s.pos[1] = sp[x + 1]; s.pos[2] = sp[x + 2];
      s.vel[0] = sv[x]; s.vel[1] = sv[x + 1]; s.vel[2] = sv[x + 2];
    }
  }

  // one plain rk4 step of dt seconds
  step(dt) {
    const { bp, bv, sp, sv } = this._pack();
    const add = (a, b, h) => { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i] * h; return o; };

    // the four rk4 stages
    const a1 = this._accel(bp, sp);
    // k2
    const bp2 = add(bp, bv, dt / 2), sp2 = add(sp, sv, dt / 2);
    const bv2 = add(bv, a1.ab, dt / 2), sv2 = add(sv, a1.as, dt / 2);
    const a2 = this._accel(bp2, sp2);
    // k3
    const bp3 = add(bp, bv2, dt / 2), sp3 = add(sp, sv2, dt / 2);
    const bv3 = add(bv, a2.ab, dt / 2), sv3 = add(sv, a2.as, dt / 2);
    const a3 = this._accel(bp3, sp3);
    // k4
    const bp4 = add(bp, bv3, dt), sp4 = add(sp, sv3, dt);
    const bv4 = add(bv, a3.ab, dt), sv4 = add(sv, a3.as, dt);
    const a4 = this._accel(bp4, sp4);

    const comb = (y, k1, k2, k3, k4) => {
      const o = new Float64Array(y.length);
      for (let i = 0; i < y.length; i++)
        o[i] = y[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      return o;
    };

    const nbp = comb(bp, bv, bv2, bv3, bv4);
    const nbv = comb(bv, a1.ab, a2.ab, a3.ab, a4.ab);
    const nsp = comb(sp, sv, sv2, sv3, sv4);
    const nsv = comb(sv, a1.as, a2.as, a3.as, a4.as);

    this._unpack(nbp, nbv, nsp, nsv);
    this.time += dt;
  }

  // total mechanical energy of the big bodies - handy as a sanity/conservation check
  energy() {
    let ke = 0, pe = 0;
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      const m = b.GM;                                  // stands in for mass (proportional)
      ke += 0.5 * m * (b.vel[0] ** 2 + b.vel[1] ** 2 + b.vel[2] ** 2);
      for (let j = i + 1; j < this.bodies.length; j++) {
        const c = this.bodies[j];
        const d = Math.hypot(b.pos[0] - c.pos[0], b.pos[1] - c.pos[1], b.pos[2] - c.pos[2]);
        pe -= (b.GM * c.GM) / d;
      }
    }
    return ke + pe;
  }
}

// kill the net momentum so the whole system doesn't slide off across the screen
export function zeroBarycentreDrift(bodies, masses) {
  let px = 0, py = 0, pz = 0, M = 0;
  for (let i = 0; i < bodies.length; i++) {
    const m = masses[i];
    px += m * bodies[i].vel[0];
    py += m * bodies[i].vel[1];
    pz += m * bodies[i].vel[2];
    M += m;
  }
  for (const b of bodies) {
    b.vel[0] -= px / M; b.vel[1] -= py / M; b.vel[2] -= pz / M;
  }
}
