# Sun Earth Moon Sandbox

A 3D N-body sandbox of the Sun–Earth–Moon system with live gravity, custom
satellites, and an optimizer that designs the smallest satellite constellation
giving constant coverage of the lunar south pole.

## Setup

Needs Python 3

```bash
py serve.py                        # serves http://localhost:8000
```

Then open http://localhost:8000 in a browser.

To (optionally) run the offline constellation optimizer (can take up to 2 hours to finish):

```bash
pip install -r ml/requirements.txt
py ml/optimize_constellation.py    # writes ml/best_constellation.json
```

## Guide

- **Camera:** click any body (or satellite) to follow it; drag to orbit, scroll
  to zoom. The focus buttons (System / Sun / Earth / Moon) do the same.
- **Time:** Pause/Play and the speed slider (minutes to years per second).
- **Add satellite:** pick a parent (Earth or Moon) and orbital elements, then
  *Add satellite*. Satellites feel the real gravity of all three bodies; one that
  decays into a body is destroyed.
- **Optimizer:** *Train* runs the in-browser genetic algorithm live; *Deploy
  best* flies its current best around the Moon. *Import Python result* loads the
  design from the offline pipeline (`ml/best_constellation.json`). The south-pole
  marker turns green whenever a satellite has line of sight.
