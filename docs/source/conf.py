# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../src'))
sys.path.append(os.path.abspath('../../src/mlmc'))
sys.path.append(os.path.abspath('../../src/mlmc/plot'))
sys.path.append(os.path.abspath('../../src/mlmc/quantity'))
sys.path.append(os.path.abspath('../../src/mlmc/tool'))
sys.path.append(os.path.abspath('../../src/mlmc/sim'))
sys.path.append(os.path.abspath('../../src/mlmc/random'))


# -- Project information -----------------------------------------------------

project = 'MLMC'
copyright = '2021, Jan Brezina, Martin Spetlik'
author = 'Jan Brezina, Martin Spetlik'

# The full version, including alpha/beta/rc tags
release = 'daf'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    ]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme" #'alabaster'

html_theme_options = {
    #    'canonical_url': '',
    #    'analytics_id': '',
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
    #    'style_external_links': False,
    #    'vcs_pageview_mode': '',
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

autosummary_generate = True

# autodoc_default_options = {
#     'members': True,
#     # The ones below should be optional but work nicely together with
#     # example_package/autodoctest/doc/source/_templates/autosummary/class.rst
#     # and other defaults in sphinx-autodoc.
#     'show-inheritance': True,
#     'inherited-members': True,
#     'no-special-members': True,
# }
master_doc = "contents"


# General information about the project.
curr_year = datetime.datetime.now().year
project = "mlmc"
copyright = "{}, Jan Březina, Martin Špetlík".format(curr_year)
author = "Jan Březina, Martin Špetlík"
