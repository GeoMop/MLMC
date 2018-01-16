# The build directory
build:
	mkdir build
	
# Build HTML rendering of the ReST readme file using docutils.
readme: build
	rst2html README.rst build/README.html

# Build HTML rendering	of CHANGES
readme: build
	rst2html CHANGES.rst build/CHANGES.html
	
# Build MANIFEST only
manifest:
	python3 setup.py sdist --manifest-only
