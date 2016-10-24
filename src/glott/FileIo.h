#ifndef FILEIO_H_
#define FILEIO_H_

#include "definitions.h"
#include <gslwrap/vector_float.h>


int ReadWavFile(const char *fname, gsl::vector *signal, Param *params);
int WriteWavFile(const char *basename, const char *extension, const gsl::vector &signal, const int &fs);
int ReadGslVector(const char *filename, const DataType format, gsl::vector *vector_ptr);
int ReadGslVector(const char *basename, const char *extension, const DataType format, gsl::vector *vector_ptr);
int ReadGslMatrix(const char *filename, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr);
int ReadGslMatrix(const char *basename, const char *extension, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr);
int WriteGslVector(const char *basename, const char *extension, const DataType &format, const gsl::vector &vector);
int WriteGslVectorFloat(const char *basename, const char *extension, const DataType &format, const gsl::vector_float &vector) ;
int WriteGslMatrix(const char *basename, const char *extension, const DataType &format, const gsl::matrix &mat);
int ReadSynthesisData(const char *basename, Param *params, SynthesisData *data);


#endif /* FILEIO_H_ */
