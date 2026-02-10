# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../indago'))
import indago
# import sphinx_fontawesome


project = 'Indago'
copyright = '2023, Department of fluid mechanics and computational engineering, Faculty of Engineering, University of Rijeka'
author = 'Department of fluid mechanics and computational engineering, Faculty of Engineering, University of Rijeka'
version = indago.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'myst_parser',
    #'sphinx_mdinclude',
    'sphinx.ext.napoleon',
    #'sphinx_fontawesome',
    # 'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.jsmath',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# html_theme_options = {
#     'cssfiles': ["http://netdna.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]
# }

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True
autoapi_python_class_content = 'both'


def setup(app):
    app.add_css_file('style.css')
