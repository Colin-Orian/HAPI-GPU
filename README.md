HAPI - The Holography Library

This is a simple library for computing holograms.  It supports multiple algorithms 
the most interesting of which is ray tracing.  The program in the ray directory is a basic 
test of this.  The current system only ray traces spheres, since it is used for testing 
the underlying algorithm.  The HAPI directory contains the library, which is statically 
linked with the test programs.

The two external dependencies are jansson and freeimage, you may need to change project 
settings to correctly link to them.

Each program needs a display.halo file which must be in the same directory as the .exe 
file.  A sample one is included.