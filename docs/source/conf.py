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
from pathlib import Path
import sys

HERE_PATH = Path(__file__).parent
SOURCE_PATH = HERE_PATH.parent.parent
sys.path.insert(0, str(SOURCE_PATH))


# -- Project information -----------------------------------------------------

project = 'TorchOk'
copyright = 'Copyright (c) 2021-2022, Vlad Vinogradov, Rashit Bayazitov, Vladislav Patrushev, Vyacheslav Shults'
author = 'Vlad Vinogradov, Rashit Bayazitov, Vladislav Patrushev, Vyacheslav Shults'

# The full version, including alpha/beta/rc tags
release = '0.4.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.todo',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.linkcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.imgmath',
    'myst_parser',
    # 'sphinx.ext.autosectionlabel',
    'nbsphinx',
    # 'sphinx_autodoc_typehints',
    # 'sphinx_paramlinks',
    # 'sphinx.ext.githubpages',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".txt", ".md", ".ipynb"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autosummary_generate = True

autoclass_content = 'class'
autodoc_inherit_docstrings = False

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    # 'methods': True,
    'special-members': '__call__,__init__,__getitem__',
    'exclude-members': '_abc_impl',
    'show-inheritance': True,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
