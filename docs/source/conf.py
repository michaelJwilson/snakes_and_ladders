"""Sphinx configuration for snakes_and_ladders's API docs."""

from __future__ import annotations

project = "snakes_and_ladders"
copyright = "2026, M. J. Wilson"
author = "M. J. Wilson"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # parses the NumPy-style docstrings used in this repo
    "sphinx.ext.viewcode",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "alabaster"
