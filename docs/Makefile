# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD = poetry run sphinx-build
SPHINXAPIDOC = poetry run sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).

# Generate API rst files (mirrors src/ package tree)
.PHONY: api
api:
	@poetry run python "$(SOURCEDIR)/generate_api.py"

.PHONY: user-guide
user-guide:
	@poetry run python "$(SOURCEDIR)/generate_user_guide.py"

# Build HTML after regenerating API
html: api user-guide
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ -d "$(SOURCEDIR)/user_guide/iframe_figures" ]; then mkdir -p "$(BUILDDIR)/html/user_guide" && cp -R "$(SOURCEDIR)/user_guide/iframe_figures" "$(BUILDDIR)/html/user_guide/"; fi

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
