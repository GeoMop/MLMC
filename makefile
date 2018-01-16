# The build directory
build:
	mkdir build
	
# Build HTML rendering of the ReST readme file using docutils.
readme: build
	rst2html README.rst build/README.html
	