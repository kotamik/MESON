// ui.js - dom wiring. time controls, the satellite form, and the live GA trainer.

import { createSandbox } from './sandbox.js';
import { GAOptimizer } from './optimizer.js';
import { DEG, RAD, MOON, DAY } from './constants.js';
import { orbitalPeriod } from './orbital.js';
import { GM_MOON } from './constants.js';

const $ = (id) => document.getElementById(id);
const sandbox = createSandbox($('scene'));
window.sandbox = sandbox;            // handy handle for the console
window.addEventListener('resize', () => sandbox.resize());
sandbox.onSatellitesChanged = () => renderSatList();   // redraw the list when one crashes

// time controls
$('play').onclick = () => {
  const p = sandbox.togglePlay();
  $('play').textContent = p ? 'Pause' : 'Play';
};
document.querySelectorAll('[data-focus]').forEach(b =>
  b.onclick = () => sandbox.snapTo(b.dataset.focus));

const speed = $('speed');
function applySpeed() {
  const sps = Math.pow(10, parseFloat(speed.value)) * 60;   // 10^v minutes per sec
  sandbox.setSpeed(sps);
  $('speedLbl').textContent = `${fmtDur(sps)} of sim per real second`;
}
speed.oninput = applySpeed; applySpeed();

// add satellite
function readSatForm() {
  const parent = $('sParent').value;
  const R = parent === 'Earth' ? 6.371e6 : MOON.radius;
  const alt = parseFloat($('sAlt').value) * 1e3;
  const e = parseFloat($('sEcc').value);
  // altitude field is periapsis altitude, so a = rp / (1-e)
  const rp = R + alt;
  const a = rp / (1 - e);
  return {
    parent, reference: $('sRef').value,
    el: {
      a, e,
      i:  parseFloat($('sInc').value) * DEG,
      om: parseFloat($('sRaan').value) * DEG,
      w:  parseFloat($('sArgp').value) * DEG,
      M:  parseFloat($('sMean').value) * DEG,
    },
  };
}
$('addSat').onclick = () => {
  const f = readSatForm();
  const colors = [0xffc451, 0x6ee7ff, 0xff7ad9, 0x9bff6e, 0xffa14f, 0xc792ff];
  sandbox.addSatellite(f.el, f.parent, f.reference, colors[sandbox.satellites.length % colors.length]);
  renderSatList();
};
$('clearSat').onclick = () => { sandbox.clearSatellites(); renderSatList(); };

// dump the current sats into the little text list under the form
function renderSatList() {
  $('satList').innerHTML = sandbox.satellites.map((s, i) => {
    const el = s.def.el;
    const R = s.def.parentName === 'Earth' ? 6.371e6 : MOON.radius;
    const altPeri = (el.a * (1 - el.e) - R) / 1e3;
    return `#${i + 1} ${s.def.parentName} · ${(altPeri).toFixed(0)}km · ` +
           `i=${(el.i * RAD).toFixed(0)}° e=${el.e.toFixed(2)}`;
  }).join('<br>') || '<span style="opacity:.5">none</span>';
}

// optimizer
let ga = null, running = false;
const sparkCtx = $('oSpark').getContext('2d');
const covHist = [];

$('oRun').onclick = () => {
  ga = new GAOptimizer({
    maxN: Math.max(1, parseInt($('oMax').value) || 8),
    minElevDeg: parseFloat($('oElev').value) || 5,
  });
  covHist.length = 0;
  running = true;
  loopGA();
};
$('oStop').onclick = () => { running = false; };

$('oDeploy').onclick = () => {
  if (!ga) return;
  sandbox.resetEpoch();               // start at the epoch the optimizer scored, or it won't match
  sandbox.clearSatellites();
  const colors = [0x33ff88, 0x6ee7ff, 0xffc451, 0xff7ad9, 0x9bff6e, 0xffa14f, 0xc792ff, 0xff5e5e];
  ga.bestConstellation().forEach((el, i) =>
    sandbox.addSatellite(el, 'Moon', 'equator', colors[i % colors.length]));
  sandbox.snapTo('Moon');
  renderSatList();
};

// pull in whatever the offline python run spat out
$('oImport').onclick = async () => {
  try {
    const res = await fetch('./ml/best_constellation.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    sandbox.resetEpoch();             // deploy at the epoch it was designed for
    sandbox.clearSatellites();
    const colors = [0x33ff88, 0x6ee7ff, 0xffc451, 0xff7ad9, 0x9bff6e, 0xffa14f, 0xc792ff, 0xff5e5e];
    data.satellites.forEach((el, i) =>
      sandbox.addSatellite(el, 'Moon', 'equator', colors[i % colors.length]));
    sandbox.snapTo('Moon');
    renderSatList();
    $('oCov').textContent = (data.meta.coverage * 100).toFixed(2) + '%';
    $('oN').textContent = data.meta.n_sats;
    $('oGap').textContent = fmtDur(data.meta.max_gap_s);
    $('oElems').innerHTML = `<b>Imported from Python</b> (${data.meta.generated})<br>` +
      data.satellites.map((el, i) =>
        `#${i + 1} a=${(el.a / 1e3).toFixed(0)}km e=${el.e.toFixed(3)} i=${(el.i * RAD).toFixed(1)}°`).join('<br>');
  } catch (e) {
    $('oElems').textContent = 'could not load ml/best_constellation.json - run the python optimizer first (and serve over http).';
  }
};

function loopGA() {
  if (!running || !ga) return;
  // one generation per frame so the ui doesn't freeze
  const s = ga.step();

  $('oGen').textContent = s.generation;
  $('oFit').textContent = s.fitness.toFixed(1);
  $('oCov').textContent = (s.stats.coverage * 100).toFixed(2) + '%';
  $('oN').textContent = s.nSats;
  $('oGap').textContent = fmtDur(s.stats.maxGap);

  covHist.push(s.stats.coverage);
  drawSpark();

  $('oElems').innerHTML = s.elements.map((el, i) => {
    const T = orbitalPeriod(el.a, GM_MOON);
    return `#${i + 1} a=${(el.a / 1e3).toFixed(0)}km e=${el.e.toFixed(3)} ` +
           `i=${(el.i * RAD).toFixed(1)}° Ω=${(el.om * RAD).toFixed(0)}° ` +
           `ω=${(el.w * RAD).toFixed(0)}° T=${(T / 3600).toFixed(1)}h`;
  }).join('<br>');

  // stop once we've locked in solid coverage and it's been running a while
  if (s.stats.coverage >= 0.9995 && s.stats.maxGap < 60 && s.generation > 40) running = false;
  requestAnimationFrame(loopGA);
}

function drawSpark() {
  const w = $('oSpark').width = $('oSpark').clientWidth;
  const h = $('oSpark').height = 54;
  sparkCtx.clearRect(0, 0, w, h);
  // the 100% line up top
  sparkCtx.strokeStyle = '#333'; sparkCtx.beginPath();
  sparkCtx.moveTo(0, 4); sparkCtx.lineTo(w, 4); sparkCtx.stroke();
  sparkCtx.strokeStyle = '#5ac66f'; sparkCtx.lineWidth = 1.5; sparkCtx.beginPath();
  covHist.forEach((c, i) => {
    const x = (i / Math.max(1, covHist.length - 1)) * w;
    const y = 4 + (1 - c) * (h - 8);
    i ? sparkCtx.lineTo(x, y) : sparkCtx.moveTo(x, y);
  });
  sparkCtx.stroke();
}

// hud - refresh the corner readout a few times a second
setInterval(() => {
  $('hDate').textContent = sandbox.simDate().toISOString().slice(0, 10);
  $('hT').textContent = fmtDur(sandbox.time);
  $('hE').textContent = (sandbox.energyDrift() * 100).toExponential(1) + '%';
  $('hSat').textContent = sandbox.satellites.length;
  const ok = sandbox.poleCoveredNow(parseFloat($('oElev').value) || 5);
  const p = $('hPole');
  p.textContent = ok ? 'LINK OK' : 'NO LINK';
  p.className = 'pill ' + (ok ? 'up' : 'down');
}, 250);

// helpers
function fmtDur(sec) {
  const a = Math.abs(sec);
  if (a < 90) return sec.toFixed(0) + ' s';
  if (a < 5400) return (sec / 60).toFixed(1) + ' min';
  if (a < 2 * DAY) return (sec / 3600).toFixed(1) + ' h';
  if (a < 400 * DAY) return (sec / DAY).toFixed(1) + ' d';
  return (sec / (365.25 * DAY)).toFixed(2) + ' yr';
}

renderSatList();
