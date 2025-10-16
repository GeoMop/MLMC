# Configuration file for the Sphinx documentation builder.
#
# For a full list of configuration options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime

# -- Path setup --------------------------------------------------------------

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

curr_year = datetime.datetime.now().year
project = "mlmc"
copyright = f"{curr_year}, Martin Špetlík, Jan Březina"
author = "Martin Špetlík, Jan Březina"

# The full version, including alpha/beta/rc tags
release = "1.0.2"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

# Autodoc settings
autosummary_generate = True           # Generate autosummary files
autoclass_content = "class"           # Don't repeat __init__ docstring
autodoc_member_order = "groupwise"    # Grouped members in docs
autodoc_typehints = "description"     # Show type hints in docstring

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Exclude build files and system junk
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Optional: uncomment if you have static files like custom CSS
# html_static_path = ["_static"]

# This tells Sphinx which file is the master doc (entry point)
master_doc = "contents"

# -- Optional convenience: print path info on build --------------------------
print(f"[conf.py] Using sys.path[0]: {sys.path[0]}")
