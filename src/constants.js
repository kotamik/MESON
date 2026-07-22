// constants.js - real astronomical data at the j2000 epoch (2000-01-01 12:00 TT).
// everything's SI (metres, kg, seconds, radians) unless the name says otherwise.
// numbers come from the usual places: iau 2009 constants, jpl de430, standish's
// mean planetary elements, and nasa's lunar constants doc.

export const G = 6.67430e-11;            // gravitational constant  [m^3 kg^-1 s^-2]
export const DEG = Math.PI / 180;        // deg -> rad
export const RAD = 180 / Math.PI;        // rad -> deg
export const AU  = 1.495978707e11;       // astronomical unit       [m]
export const DAY = 86400;                // seconds in a day
export const YEAR = 365.25 * DAY;        // julian year             [s]

// gravitational parameters GM = G*mass. we know GM way better than G and mass separately.
export const GM_SUN   = 1.32712440018e20;
export const GM_EARTH = 3.986004418e14;
export const GM_MOON  = 4.9028695e12;

// the bodies. keplerian elements are osculating mean elements at j2000, in the
// j2000 ecliptic frame:
//   a   semi-major axis            [m]
//   e   eccentricity               [-]
//   i   inclination                [rad]  (to the ecliptic)
//   om  long. of ascending node Ω  [rad]
//   w   arg of periapsis ω         [rad]
//   M   mean anomaly at epoch      [rad]
//   parent - what it orbits ('sun', 'earth', null)

export const SUN = {
  name: 'Sun',
  mass: 1.98847e30,
  GM: GM_SUN,
  radius: 6.9634e8,
  color: 0xffd23b,
  parent: null,
  // sun is basically at the barycentre. start it at rest at the origin and let
  // the n-body integrator nudge it around on its own.
  rotationPeriod: 25.05 * DAY,           // equatorial sidereal rotation
  axialTilt: 7.25 * DEG,                 // to the ecliptic
};

export const EARTH = {
  name: 'Earth',
  mass: 5.97219e24,
  GM: GM_EARTH,
  radius: 6.371e6,                       // volumetric mean radius
  equatorialRadius: 6.378137e6,
  color: 0x2a6fdb,
  parent: 'sun',
  // heliocentric ecliptic mean elements (earth-moon barycentre, standish j2000)
  elements: {
    a:  1.00000261 * AU,
    e:  0.01671123,
    i: -0.00001531 * DEG,
    om: 0.0 * DEG,                       // Ω is undefined at ~0 inclination, just use 0
    w:  102.93768193 * DEG,              // = longitude of perihelion since Ω=0
    M: (100.46457166 - 102.93768193) * DEG,  // L - long.perihelion at epoch
  },
  rotationPeriod: 86164.0905,            // sidereal day [s]
  axialTilt: 23.43928 * DEG,             // obliquity of the ecliptic
  J2: 1.08263e-3,
};

export const MOON = {
  name: 'Moon',
  mass: 7.342e22,
  GM: GM_MOON,
  radius: 1.7374e6,
  color: 0xbfbfbf,
  parent: 'earth',
  // geocentric ecliptic mean elements at j2000 (the classic mean lunar elements)
  elements: {
    a:  3.84399e8,
    e:  0.0549,
    i:  5.145 * DEG,
    om: 125.08 * DEG,                    // this node regresses on an 18.6-year cycle
    w:  318.15 * DEG,
    M:  135.27 * DEG,
  },
  rotationPeriod: 27.321661 * DAY,       // synchronous - tidally locked
  axialTilt: 1.5424 * DEG,               // spin axis tilt to the ecliptic
  // selenodetic bits we use for frozen-orbit / longevity checks
  J2: 2.0330530e-4,                      // biggest zonal harmonic
  refRadius: 1.7374e6,                   // reference radius for those harmonics
};

export const BODIES = [SUN, EARTH, MOON];

// render scale. the real system spans ~1.5e11 m and webgl is happier with O(1)
// numbers, so 1 scene unit = 1e6 m (1000 km). RADIUS_GAIN can fatten the bodies
// so the moon isn't a sub-pixel speck next to a 150-million-km orbit - the
// physics always runs on the true SI values regardless.
export const SCENE_UNIT = 1e6;           // metres per scene unit
export const RADIUS_GAIN = 1;            // true scale; bodies also get screen-space markers

export const m2u = (m) => m / SCENE_UNIT;          // metres -> scene units
export const u2m = (u) => u * SCENE_UNIT;          // scene units -> metres
