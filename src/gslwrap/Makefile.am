EXTRA_DIST = generate matrix_source.cc matrix_source.h vector_source.cc vector_source.h
INCLUDES = -I$(top_srcdir) @GSL_CFLAGS@
lib_LIBRARIES   = libgslwrap.a
libgslwrapincludedir = $(includedir)/gslwrap

BUILT_SOURCES =  matrix_float.h matrix_double.h matrix_int.h \
vector_double.h  vector_float.h  vector_int.h \
matrix_float.cc matrix_double.cc matrix_int.cc \
vector_float.cc vector_double.cc vector_int.cc 

$(BUILT_SOURCES): vector_source.h matrix_source.h vector_source.cc matrix_source.cc
	./generate

libgslwrapinclude_HEADERS = matrix_float.h matrix_double.h matrix_int.h \
vector_double.h  vector_float.h  vector_int.h \
permutation.h \
histogram.h \
matrix_vector_operators.h \
min_fminimizer.h multimin_fdfminimizer.h \
random_generator.h \
random_number_distribution.h

libgslwrap_a_SOURCES = matrix_float.cc matrix_double.cc matrix_int.cc \
vector_float.cc vector_double.cc vector_int.cc \
random_generator.cc 

