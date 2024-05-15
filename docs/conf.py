import pylater

project = "pylater"
copyright = "2024, MDAP"
author = "MDAP"
version = pylater.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_nb",
    "sphinx_autodoc_typehints",
]

nb_execution_timeout = -1

templates_path = ["_templates"]
exclude_patterns = []

maximum_signature_line_length = 88

html_theme = "furo"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_show_copyright = False
html_show_sphinx = False

html_theme_options = {}

autodoc_typehints = "both"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autoclass_content = "init"

typehints_use_signature = True

def setup(app):
    app.add_css_file("types_fix.css")
