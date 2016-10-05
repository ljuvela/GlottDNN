AUTOMAKE_OPTIONS = subdir-objects
AM_CXXFLAGS = -std=c++11 -Wall -Wextra

noinst_LIBRARIES = libgslwrap.a
libgslwrap_a_SOURCES = gslwrap/matrix_float.h gslwrap/matrix_double.h gslwrap/matrix_int.h \
gslwrap/vector_double.h  gslwrap/vector_float.h  gslwrap/vector_int.h \
gslwrap/permutation.h \
gslwrap/histogram.h \
gslwrap/matrix_vector_operators.h \
gslwrap/min_fminimizer.h gslwrap/multimin_fdfminimizer.h \
gslwrap/random_generator.h \
gslwrap/random_number_distribution.h \
gslwrap/matrix_float.cc gslwrap/matrix_double.cc gslwrap/matrix_int.cc \
gslwrap/vector_float.cc gslwrap/vector_double.cc gslwrap/vector_int.cc \
gslwrap/random_generator.cc 

bin_PROGRAMS = Analysis

Analysis_SOURCES = glott/Analysis.cpp Filters.h \
					glott/definitions.h glott/definitions.cpp \
					glott/ReadConfig.h glott/ReadConfig.cpp \
					glott/FileIo.h glott/FileIo.cpp \
					glott/InverseFiltering.h glott/InverseFiltering.cpp \
					glott/SpFunctions.h glott/SpFunctions.cpp \
					glott/AnalysisFunctions.h glott/AnalysisFunctions.cpp \
					glott/DebugUtils.h glott/DebugUtils.cpp
					
Analysis_LDADD = libgslwrap.a
