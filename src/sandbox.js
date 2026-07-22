// sandbox.js - the 3d sun/earth/moon view, built on three.js

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import {
  SUN, EARTH, MOON, DAY, m2u, RADIUS_GAIN,
} from './constants.js';
import { elementsToState } from './orbital.js';
import { equatorialToEcliptic, identity } from './frames.js';
import { buildSystem } from './ephemeris.js';

const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);   // epoch as a js timestamp
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

export function createSandbox(container) {
  const { sim, sun, earth, moon } = buildSystem();
  const bodyByName = { Sun: sun, Earth: earth, Moon: moon };

  // renderer / scene / camera
  const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0c);

  const camera = new THREE.PerspectiveCamera(55,
    container.clientWidth / container.clientHeight, 0.01, 5e7);
  camera.up.set(0, 0, 1);                 // ecliptic z is up
  camera.position.set(0, -800, 400);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // stars + a sun light
  scene.add(makeStarfield());
  const sunLight = new THREE.PointLight(0xffffff, 4, 0, 0.0);
  scene.add(sunLight);
  scene.add(new THREE.AmbientLight(0x223044, 0.6));

  // body visuals
  const visuals = {};
  for (const def of [SUN, EARTH, MOON]) {
    const b = bodyByName[def.name];
    visuals[def.name] = makeBodyVisual(def, scene, b === sun);
  }

  // south pole marker + spin axis on the moon
  const poleGroup = new THREE.Group();
  const poleDot = new THREE.Mesh(
    new THREE.SphereGeometry(m2u(MOON.radius) * 0.08, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0x5ac66f }));
  const axis = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, m2u(MOON.radius) * 1.6),
      new THREE.Vector3(0, 0, -m2u(MOON.radius) * 1.6)]),
    new THREE.LineBasicMaterial({ color: 0x335577 }));
  poleGroup.add(poleDot, axis);
  scene.add(poleGroup);

  // moon equator tilt (mean obliquity to the ecliptic)
  const moonEqToEcl = equatorialToEcliptic(MOON.axialTilt);
  const earthEqToEcl = equatorialToEcliptic(EARTH.axialTilt);

  // orbit trails
  const trails = {
    Earth: makeTrail(scene, 0x3a6fd0, 1500, 400 * DAY),   // ~a full heliocentric orbit
    Moon:  makeTrail(scene, 0x888888, 1200, 30 * DAY),    // ~a full lunar orbit
  };

  // satellites: { body, mesh, trail, def }
  const satellites = [];
  const satGroup = new THREE.Group();
  scene.add(satGroup);

  // time control
  let playing = true;
  let secPerSec = DAY * 5;      // sim seconds per real second
  let lastT = performance.now();
  let energy0 = sim.energy();

  // camera follow. focusObj is whatever the camera should ride along with
  // (null = free / whole-system view). focusPrev is where it sat last frame,
  // so we can shift the camera by however far the body moved this frame.
  let focusObj = null;
  const focusPrev = new THREE.Vector3();

  // public api
  const api = {
    sim, bodyByName,
    scene, camera, controls, visuals,     // exposed for the console / verification
    get time() { return sim.time; },
    get playing() { return playing; },
    setPlaying(v) { playing = v; },
    togglePlay() { playing = !playing; return playing; },
    setSpeed(secondsPerSecond) { secPerSec = secondsPerSecond; },
    getSpeed() { return secPerSec; },
    focusOn(name) { api.snapTo(name); },
    energyDrift() { return (sim.energy() - energy0) / Math.abs(energy0); },
    simDate() { return new Date(J2000_MS + sim.time * 1000); },
    moonState: () => moon,
    moonPoleDir,
    satellites,
    onSatellitesChanged: null,      // ui hook, fired when a sat is auto-destroyed

    // drop in a satellite from keplerian elements around a parent body.
    // el: {a,e,i,om,w,M} in radians/metres; parent 'Earth'|'Moon';
    // reference: 'equator'|'ecliptic'; color optional.
    addSatellite(el, parentName = 'Moon', reference = 'equator', color = 0xffc451) {
      const parent = bodyByName[parentName];
      const mu = parentName === 'Earth' ? EARTH.GM : MOON.GM;
      const local = elementsToState(el, mu);
      const xf = reference === 'equator'
        ? (parentName === 'Earth' ? earthEqToEcl : moonEqToEcl)
        : identity;
      const pos = add(parent.pos, xf(local.pos));
      const vel = add(parent.vel, xf(local.vel));
      const body = sim.addSatellite({ pos: pos.slice(), vel: vel.slice() });

      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(m2u(MOON.radius) * 0.05, 0.6), 8, 8),
        new THREE.MeshBasicMaterial({ color }));
      satGroup.add(mesh);
      const trail = makeTrail(satGroup, color, 500, 5 * DAY);   // last few days of the orbit
      const rec = { body, mesh, trail, def: { el, parentName, reference, color } };
      satellites.push(rec);
      return rec;
    },

    clearSatellites() {
      while (satellites.length) removeSat(satellites[satellites.length - 1]);
    },

    // true if the south pole can see at least one sat above minElevDeg right now
    poleCoveredNow(minElevDeg = 5) {
      const pole = moonSouthPolePoint();
      const up = moonPoleDir(-1);
      const sinMin = Math.sin(minElevDeg * Math.PI / 180);
      for (const s of satellites) {
        const dx = s.body.pos[0] - pole[0];
        const dy = s.body.pos[1] - pole[1];
        const dz = s.body.pos[2] - pole[2];
        const d = Math.hypot(dx, dy, dz);
        if ((up[0] * dx + up[1] * dy + up[2] * dz) / d >= sinMin) return true;
      }
      return false;
    },

    resize() {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    },
  };

  // moon spin axis direction in ecliptic coords (sign +1 north, -1 south)
  function moonPoleDir(sign = 1) {
    return moonEqToEcl([0, 0, sign]);
  }
  function moonSouthPolePoint() {
    const d = moonPoleDir(-1);
    return [moon.pos[0] + d[0] * MOON.radius,
            moon.pos[1] + d[1] * MOON.radius,
            moon.pos[2] + d[2] * MOON.radius];
  }

  // yank one satellite out of the sim + scene and free its gpu bits
  function removeSat(rec) {
    if (focusObj === rec.mesh) focusObj = null;   // stop following a dead sat
    satGroup.remove(rec.mesh);
    rec.mesh.geometry.dispose();
    rec.mesh.material.dispose();
    rec.trail.dispose();
    const si = sim.sats.indexOf(rec.body);
    if (si >= 0) sim.sats.splice(si, 1);
    const ai = satellites.indexOf(rec);
    if (ai >= 0) satellites.splice(ai, 1);
  }

  // kill off any sat that has crashed into a body or been flung out of the
  // system, so a decayed orbit just vanishes instead of rocketing to infinity.
  const ESCAPE_DIST = 5 * EARTH.elements.a;
  function pruneCrashed() {
    let removed = 0;
    for (let i = satellites.length - 1; i >= 0; i--) {
      const p = satellites[i].body.pos;
      let dead = !isFinite(p[0]) || !isFinite(p[1]) || !isFinite(p[2]);
      if (!dead) {
        for (const def of [SUN, EARTH, MOON]) {        // hit the surface of a body?
          const b = bodyByName[def.name];
          if (Math.hypot(p[0] - b.pos[0], p[1] - b.pos[1], p[2] - b.pos[2]) <= def.radius) {
            dead = true; break;
          }
        }
      }
      if (!dead && Math.hypot(p[0], p[1], p[2]) > ESCAPE_DIST) dead = true;   // flung away
      if (dead) { removeSat(satellites[i]); removed++; }
    }
    if (removed && typeof api.onSatellitesChanged === 'function') api.onSatellitesChanged();
  }

  // main loop
  function frame(now) {
    const dtReal = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;

    if (playing) {
      // chop the step up so each rk4 step stays small enough to be accurate
      const target = secPerSec * dtReal;
      const maxStep = 600;                      // 10 min max per rk4 step
      let remaining = target, guard = 0;
      while (remaining > 1e-6 && guard++ < 4000) {
        const h = Math.min(maxStep, remaining);
        sim.step(h);
        remaining -= h;
      }
      if (satellites.length) pruneCrashed();
      pushTrails();
    }

    updateVisuals(now);
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  function pushTrails() {
    // the trail decides for itself whether enough sim-time has passed to sample
    trails.Earth.push(rel(earth.pos, sun.pos), sim.time);
    trails.Moon.push(rel(moon.pos, earth.pos), sim.time);   // geocentric trail
    for (const s of satellites) {
      const parent = s.def.parentName === 'Earth' ? earth : moon;
      s.trail.push(rel(s.body.pos, parent.pos), sim.time);
    }
  }

  function updateVisuals(now) {
    // meshes track the physics, and spin about their tilted axes
    for (const def of [SUN, EARTH, MOON]) {
      const b = bodyByName[def.name];
      const v = visuals[def.name];
      v.group.position.set(m2u(b.pos[0]), m2u(b.pos[1]), m2u(b.pos[2]));
      const spin = def.rotationPeriod ? (sim.time / def.rotationPeriod) * Math.PI * 2 : 0;
      v.spin.rotation.y = spin;
      // keep the marker roughly the same size on screen no matter the zoom
      const dist = camera.position.distanceTo(v.group.position);
      v.marker.scale.setScalar(Math.max(dist * 0.012, m2u(def.radius) * 1.4));
    }
    sunLight.position.copy(visuals.Sun.group.position);

    // trails live in each parent's frame, so move their origins along
    trails.Earth.group.position.copy(visuals.Sun.group.position);
    trails.Moon.group.position.copy(visuals.Earth.group.position);

    // pole marker + axis ride the moon and go green when we have coverage
    poleGroup.position.copy(visuals.Moon.group.position);
    poleGroup.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(...moonPoleDir(1)).normalize());
    poleDot.position.set(0, 0, -m2u(MOON.radius));
    poleDot.material.color.setHex(api.poleCoveredNow() ? 0x5ac66f : 0xe05a5a);

    // satellites track physics in their parent's frame
    for (const s of satellites) {
      const parent = s.def.parentName === 'Earth' ? earth : moon;
      const pv = visuals[s.def.parentName].group.position;
      s.mesh.position.set(
        pv.x + m2u(s.body.pos[0] - parent.pos[0]),
        pv.y + m2u(s.body.pos[1] - parent.pos[1]),
        pv.z + m2u(s.body.pos[2] - parent.pos[2]));
      s.trail.group.position.copy(pv);
    }

    applyFocus();
  }

  function applyFocus() {
    if (!focusObj) return;
    // ride along with the focused body: shove the camera AND the look-at target
    // by however far the body moved this frame. orbit + zoom offset is preserved,
    // so the user can still spin/zoom around it while it drifts through space.
    const p = focusObj.position;
    const dx = p.x - focusPrev.x, dy = p.y - focusPrev.y, dz = p.z - focusPrev.z;
    camera.position.x += dx; camera.position.y += dy; camera.position.z += dz;
    controls.target.x += dx; controls.target.y += dy; controls.target.z += dz;
    focusPrev.copy(p);
  }

  const rel = (p, o) => [m2u(p[0] - o[0]), m2u(p[1] - o[1]), m2u(p[2] - o[2])];

  // rewind sun/earth/moon back to the j2000 epoch and zero the clock. deploying
  // an optimized constellation calls this first, so the sats start in exactly
  // the state the optimizer scored them in. without it they'd be inserted at
  // whatever time the sim has drifted to - a different sun/earth/moon geometry,
  // a different perturbation phase - and behave nothing like the reported result.
  api.resetEpoch = function () {
    const fresh = buildSystem();
    const src = { Sun: fresh.sun, Earth: fresh.earth, Moon: fresh.moon };
    for (const name of ['Sun', 'Earth', 'Moon']) {
      const s = src[name], d = bodyByName[name];
      d.pos[0] = s.pos[0]; d.pos[1] = s.pos[1]; d.pos[2] = s.pos[2];
      d.vel[0] = s.vel[0]; d.vel[1] = s.vel[1]; d.vel[2] = s.vel[2];
    }
    sim.time = 0;
    energy0 = sim.energy();
    trails.Earth.clear();
    trails.Moon.clear();
  };

  // lock the camera onto obj and back off `dist` scene units
  function focusTo(obj, dist) {
    focusObj = obj;
    focusPrev.copy(obj.position);
    controls.target.copy(obj.position);
    camera.position.set(obj.position.x + dist, obj.position.y - dist, obj.position.z + dist * 0.5);
  }

  // jump the camera to a body (or the whole system) and start following it
  api.snapTo = function (name) {
    if (name === 'System') {
      focusObj = null;                            // free view, system stays near origin
      controls.target.set(0, 0, 0);
      camera.position.set(0, -m2u(EARTH.elements.a) * 1.4, m2u(EARTH.elements.a) * 0.7);
      return;
    }
    const def = name === 'Sun' ? SUN : name === 'Earth' ? EARTH : MOON;
    const v = visuals[name];
    const r = m2u(def.radius);
    const d = name === 'Moon' ? r * 8 : name === 'Earth' ? r * 10 : r * 4;
    focusTo(v.group, d);
  };

  // click a body or satellite to follow it. dragging is just orbiting, so skip those.
  let downAt = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!downAt) return;
    const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
    downAt = null;
    if (moved > 5) return;                        // that was a drag, leave focus alone
    const hit = pickAt(e.clientX, e.clientY);
    if (!hit) return;                             // missed everything, keep current view
    if (hit.name) api.snapTo(hit.name);
    else focusTo(hit.obj, m2u(MOON.radius) * 4);  // a satellite - hug it close
  });

  // whichever body/sat marker is closest on screen to the click (within a few px)
  function pickAt(clientX, clientY) {
    camera.updateMatrixWorld();
    const rect = renderer.domElement.getBoundingClientRect();
    const cx = clientX - rect.left, cy = clientY - rect.top;
    let best = null, bestD = 26;                  // px search radius
    const consider = (obj, name) => {
      const p = obj.position.clone().project(camera);
      if (p.z > 1) return;                        // behind the camera
      const sx = (p.x * 0.5 + 0.5) * rect.width;
      const sy = (-p.y * 0.5 + 0.5) * rect.height;
      const d = Math.hypot(sx - cx, sy - cy);
      if (d < bestD) { bestD = d; best = { obj, name }; }
    };
    consider(visuals.Sun.group, 'Sun');
    consider(visuals.Earth.group, 'Earth');
    consider(visuals.Moon.group, 'Moon');
    for (const s of satellites) consider(s.mesh, null);
    return best;
  }

  requestAnimationFrame(frame);
  api.snapTo('Earth');
  return api;
}

// visual helpers
function makeBodyVisual(def, scene, emissive) {
  const group = new THREE.Group();
  const r = m2u(def.radius) * RADIUS_GAIN;
  const mat = emissive
    ? new THREE.MeshBasicMaterial({ color: def.color })
    : new THREE.MeshStandardMaterial({ color: def.color, roughness: 0.9, metalness: 0.0 });
  const spin = new THREE.Group();
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), mat);
  // tilt the spin axis by the body's obliquity (about x)
  spin.rotation.x = def.axialTilt || 0;
  spin.add(sphere);
  group.add(spin);

  // a little screen-space dot so you can find the body at any zoom
  const marker = new THREE.Sprite(new THREE.SpriteMaterial({
    color: def.color, transparent: true, opacity: 0.9,
    map: dotTexture(), depthWrite: false }));
  group.add(marker);

  scene.add(group);
  return { group, spin, sphere, marker };
}

// orbit trail. `ring`/`ringT` hold the raw samples; `draw` is a SEPARATE buffer
// that three actually renders (they must not alias, or reordering corrupts it).
// points are sampled on sim-time cadence and anything older than maxAgeSec is
// dropped, so trails don't grow forever and stay sane at any time-warp.
function makeTrail(parent, color, maxPoints, maxAgeSec) {
  const group = new THREE.Group();
  const ring  = new Float32Array(maxPoints * 3);   // raw positions, ring buffer
  const ringT = new Float64Array(maxPoints);        // sim-time each point was taken
  const draw  = new Float32Array(maxPoints * 3);    // contiguous buffer three draws
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(draw, 3));
  geo.setDrawRange(0, 0);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
  line.frustumCulled = false;
  group.add(line);
  parent.add(group);
  const sampleDt = maxAgeSec / maxPoints;           // min sim-seconds between samples
  let count = 0, head = 0, lastT = -Infinity;
  return {
    group,
    push(p, t) {
      if (t - lastT < sampleDt) return;             // not time for a new sample yet
      lastT = t;
      ring[head * 3] = p[0]; ring[head * 3 + 1] = p[1]; ring[head * 3 + 2] = p[2];
      ringT[head] = t;
      head = (head + 1) % maxPoints;
      count = Math.min(count + 1, maxPoints);
      // rebuild the draw buffer oldest->newest, skipping anything too old
      let n = 0;
      for (let i = 0; i < count; i++) {
        const idx = (head - count + i + maxPoints) % maxPoints;
        if (t - ringT[idx] > maxAgeSec) continue;
        draw[n * 3] = ring[idx * 3]; draw[n * 3 + 1] = ring[idx * 3 + 1]; draw[n * 3 + 2] = ring[idx * 3 + 2];
        n++;
      }
      geo.setDrawRange(0, n);
      geo.attributes.position.needsUpdate = true;
    },
    clear() { count = 0; head = 0; lastT = -Infinity; geo.setDrawRange(0, 0); },
    dispose() { parent.remove(group); geo.dispose(); },
  };
}

function makeStarfield() {
  const n = 4000, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 2e6 + Math.random() * 1e6;
    const th = Math.acos(2 * Math.random() - 1), ph = Math.random() * Math.PI * 2;
    pos[i * 3] = r * Math.sin(th) * Math.cos(ph);
    pos[i * 3 + 1] = r * Math.sin(th) * Math.sin(ph);
    pos[i * 3 + 2] = r * Math.cos(th);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xaaaacc, size: 6000, sizeAttenuation: true }));
}

let _dotTex;
function dotTexture() {
  if (_dotTex) return _dotTex;
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.4, 'rgba(255,255,255,0.5)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  _dotTex = new THREE.CanvasTexture(c);
  return _dotTex;
}
