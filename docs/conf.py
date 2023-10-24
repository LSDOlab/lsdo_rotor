# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../lsdo_project_template/core'))     # for autodoc

# -- Project information -----------------------------------------------------

project = 'LSDO Rotor'
copyright = '2023, Marius Ruh'
author = 'Marius Ruh'
version = '0.1'
# release = 0.1.0rtc


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "numpydoc",                 
    "sphinx_copybutton",            # allows copying code embedded in the docs rendered from .md or .ipynb files
    "myst_nb",                      # renders .md, .myst, .ipynb files
    "sphinx.ext.viewcode",          # adds the source code for classes and functions in auto generated api ref
    "sphinxcontrib.collections",    # adds files from outside src and executes functions before Sphinx builds
    "sphinxcontrib.bibtex",         # for references and citations
]

# sphinxcontrib.bibtex options
bibtex_bibfiles = ['src/references.bib']

myst_title_to_header = True
myst_enable_extensions = ["dollarmath", "amsmath", "tasklist"]
nb_execution_mode = 'off'

# autodoc options
# autodoc_typehints = 'description'

# autoapi options
autoapi_dirs = ["../lsdo_rotor/core"]
autoapi_root = 'src/autoapi'
autoapi_type = 'python'
autoapi_file_patterns = ['*.py', '*.pyi']
autoapi_options = [ 'members', 'undoc-members', 'private-members', 'show-inheritance', 
                   'show-module-summary', 'special-members', 'imported-members', ]
autoapi_add_toctree_entry = False
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'class' # 'both' or '__init'

root_doc = 'index'

# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'myst-nb',
#     '.myst': 'myst-nb',
#     '.ipynb': 'myst-nb',
#     }

# source_parsers = {'.md': 'myst-nb',
#                 '.ipynb': 'myst-nb',
#                 }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['README.md', '_build', 'Thumbs.db', '.DS_Store', 'src/welcome.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme' # other theme options: 'sphinx_book_theme', 'sphinx_rtd_theme', 
                                # 'alabaster', 'classic', 'sphinxdoc', 'nature', 'bizstyle', ...

# html_theme_options for sphinx_rtd_theme
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',   # other valid colors: 'white', ...
    # toc options
    'collapse_navigation': False,   # default: True
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': True     # default: False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


import glob
# Function used by collections for converting .py files from examples
# to .md and writing those into `_temp/target/` directory before Sphinx builds
def py2md(config):
    # root_dir needs a trailing slash (i.e. /root/dir/)
    for ex in glob.iglob(config['target'] + '**/ex_*.py', recursive=True):
        with open(ex) as f:
            code = f.read()
            no_line_breaks = ' '.join(code.splitlines())

            if code[0:3] == "'''":      
                title, desc = split_first_string_between_quotes(no_line_breaks, "'")
            elif code[0:3] == '"""':
                title, desc = split_first_string_between_quotes(no_line_breaks, '"')
            else:
                raise SyntaxError('Docstring for title and description is not declared correctly')

        with open(ex[:-3]+'.md', 'w') as g:
            g.write('# ' + title + '\n')
            g.write(desc + '\n\n')
            g.write('```python\n')
            g.write(code)
            g.write('\n```')

    return

import re

def split_first_string_between_quotes(code_string, quotes):
    if quotes == "'":
      check = re.search("'''(.+?)'''", code_string)
    elif quotes == '"':
      check = re.search('"""(.+?)"""', code_string)
    
    if check:
      docstring = check.group(1)
      out_strings = docstring.split(':', 1)
      if len(out_strings)==2:
        title, desc = out_strings[0].strip(), out_strings[1].strip()
      else:
        title, desc = out_strings[0].strip(), ''

      return title, desc
    
    else:
        raise SyntaxError('Docstring for title and description is not declared correctly')

collections = {
    
  #   # copy_tutorials collection copies the contents inside `/tutorials` 
  #   # directory into `/src/_temp/tutorials`
  #  'copy_tutorials': {
  #     'driver': 'copy_folder',
  #     'source': '../tutorials', # source relative to path of makefile, not wrt /src
  #     'target': 'tutorials/',
  #     'ignore': [],
  #   #   'active': True,         # default: True. If False, this collection is ignored during doc build.
  #   #   'safe': True,           # default: True. If True, any problem will raise an exception and stops the build.
  #     'clean': True,            # default: True. If False, no cleanup is done before collections get executed.
  #     'final_clean': True,      # default: True. If True, a final cleanup is done at the end of a Sphinx build.
  #   #   'tags': ['my_collection', 'dummy'],     # List of tags, which trigger an activation of the collection.
  #                                       # Should be used together with active set to False, 
  #                                       # otherwise the collection gets always executed.
  #                                       # Use -t tag option of sphinx-build command to trigger related collections.
  #                                       # e.g. : `sphinx-build -b html -t dummy . _build/html`
  #  },

   'copy_examples': {
      'driver': 'copy_folder',
      'source': '../examples',  # source relative to path of makefile, not wrt /src
      'target': 'examples/',
      'ignore': [],
      'clean': True,            # default: True. If False, no cleanup is done before collections get executed.
      'final_clean': True,      # default: True. If True, a final cleanup is done at the end of a Sphinx build.
   },

    # convert_examples collection converts all .py files to .md files recursively inside `_temp/examples` 
    # directory and also extracts the docstrings from the .py files to generate title and descriptions
    # for those examples
   'convert_examples': {
      'driver': 'writer_function',  # uses custom WriterFunctionDriver written by Anugrah
      'from'  : '_temp/examples/',  # source relative to path of makefile, not wrt /src
      'source': py2md,              # custom function written above in `conf.py`
      'target': 'examples/',        # target was a file for original FunctionDriver, e.g., 'target': 'examples/temp.txt'
                                    # the original FunctionDriver was supposed to write only 1 file.
      'clean': True,       
      'final_clean': True,      
    #   'write_result': True,   # this prevents original FunctionDriver from writing to the target file
   },
}

collections_target = 'src/_temp'    # default : '_collections', the default storage location for all collections
collections_clean  = True           # default : True, all configured target locations get wiped out at the beginning
                                    # can be overwritten for individual collection by setting value for the 'clean' key
collections_final_clean  = True     # default : True, all collections start their clean-up routine after a Sphinx build is done
                                    # can be overwritten for individual collection by setting value for the 'final_clean' key



