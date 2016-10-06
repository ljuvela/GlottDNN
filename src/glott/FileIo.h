#ifndef FILEIO_H_
#define FILEIO_H_

#include "definitions.h"

void ReadWavFile(const char *fname, gsl::vector *signal, Param *params);
int ReadGslVector(const char *filename, const DataType format, gsl::vector *vector_ptr);
int WriteGslVector(const char *filename, const DataType &format, const gsl::vector &vector);

#endif /* FILEIO_H_ */
