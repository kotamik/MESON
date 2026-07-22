// frames.js - rotations between a body's equatorial plane and the ecliptic.
// it's natural to give a sat's elements relative to the equator of whatever it
// orbits (so i = 90 deg is a real polar orbit), but the sim runs in the j2000
// ecliptic frame. these helpers spin a state vector from equator into ecliptic.
//
// each body's equator is just its mean obliquity tilted about the ecliptic x
// axis (the j2000 equinox line). standard first-order thing. the moon really
// rides a slowly precessing cassini state; we cheat with its fixed 1.54 deg
// mean tilt and say so in the ui.

// rotate vector v by angle t about the x axis
export function rotX(v, t) {
  const c = Math.cos(t), s = Math.sin(t);
  return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]];
}

// build an equator->ecliptic transform for a given obliquity [rad]
export function equatorialToEcliptic(obliquity) {
  return (v) => rotX(v, obliquity);
}

// identity, for elements that are already in the ecliptic
export const identity = (v) => v;
