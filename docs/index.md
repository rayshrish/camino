# camino
[![PyPI version](https://badge.fury.io/py/jwst-camino.svg)](https://badge.fury.io/py/jwst-camino)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-Zensical-blue)](https://rayshrish.github.io/camino/)
[![CI](https://github.com/rayshrish/camino/actions/workflows/documentation.yml/badge.svg)](https://github.com/rayshrish/camino/actions/workflows/documentation.yml)

`camino` is a JAX-based package for modelling and inferring JWST wavefront aberrations from NIRCam defocused imaging.

Contributors: [Shrishmoy Ray](https://github.com/rayshrish), [Benjamin Pope](https://github.com/benjaminpope)

## What is camino?

`camino` is a Python package for modelling optical propagation, wavefront aberrations, and JWST/NIRCam-style inference workflows using JAX autodiff and dLux optics primitives.

## Installation

The package is distributed under `jwst-camino`, while the import name remains `camino`:

```bash
pip install jwst-camino
```

You can also build from source:

```bash
git clone https://github.com/your-org/camino.git
cd camino
python -m pip install -e .
```

We recommend using a virtual environment to avoid dependency conflicts.

## Collaboration & Development

We welcome collaboration on the modelling, testing, and documentation work. If you want to contribute, please work from a branch and keep the code formatted with the configured pre-commit Black checks.
