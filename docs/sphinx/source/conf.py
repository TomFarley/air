# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
project_path = os.path.abspath('../../..')
sys.path.insert(0, project_path)
#print(f'Adding project root directory to python path for sphinx documentation: {project_path}')

# -- Project information -----------------------------------------------------

project = 'Fusion Infra-Red Experiments'
copyright = '2019, Tom Farley, Scott Silburn'
author = 'Tom Farley, Scott Silburn'

# The full version, including alpha/beta/rc tags
release = 'v2.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # ['_static']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = 'images/fire_logo.jpg'

# Output file base name for HTML help builder.
htmlhelp_basename = 'FIREdoc'

latex_documents = [
    (master_doc, 'fire.tex',
     u'FIRE Documentation',
     u'Tom Farley, Scott Silburn et. al.',
     'manual'),
]