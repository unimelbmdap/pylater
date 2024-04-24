import pylater

project = "LATER modelling in Python"
copyright = "2024, MDAP"
author = "MDAP"
version = pylater.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

maximum_signature_line_length = 88

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_show_copyright = False
html_show_sphinx = False

html_theme_options = {}
