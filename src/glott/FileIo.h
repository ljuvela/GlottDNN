#ifndef FILEIO_H_
#define FILEIO_H_

#include "definitions.h"

void ReadFile(const char *fname, gsl::vector *signal, Param *params);
int ReadGslVector(const char *filename, const DataType format, gsl::vector *vector_ptr);

#endif /* FILEIO_H_ */
