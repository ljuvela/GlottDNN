#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([Glott Vocoder], [1.2], [BUG-REPORT-ADDRESS])
AC_CONFIG_SRCDIR([src/glott/Analysis.cpp])
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_RANLIB

# Checks for libraries.
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])
AC_CHECK_LIB([config++],[config_init])
AC_CHECK_LIB([sndfile],[sf_write_double])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([floor pow rint sqrt strerror])

AC_CONFIG_FILES([Makefile
		 src/Makefile
                 src/glott/Makefile])
AC_OUTPUT
