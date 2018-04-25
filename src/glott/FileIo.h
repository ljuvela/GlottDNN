// Copyright 2016-2018 Lauri Juvela and Manu Airaksinen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FILEIO_H_
#define FILEIO_H_

#include "definitions.h"

int ParseArguments(int argc, char **argv, Param *params);
void PrintUsageAnalysis();

int ReadExternalExcitation(const std::string &filename, gsl::vector *source_signal);

int ReadWavFile(const char *fname, gsl::vector *signal, Param *params);
int ReadWavFile(const char *fname, gsl::vector *signal);
int ReadWavFile(const std::string &fname, gsl::vector *signal);
int WriteWavFile(const std::string &filename, const gsl::vector &signal, const int &fs);
int ReadGslVector(const std::string &filename, const DataType format, gsl::vector *vector_ptr);
int ReadGslMatrix(const std::string &filename, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr);
int WriteGslVector(const std::string &filename, const DataType &format, const gsl::vector &vector);
int WriteGslMatrix(const std::string &filename, const DataType &format, const gsl::matrix &mat);
int ReadSynthesisData(const char *basename, Param *params, SynthesisData *data);

int ReadFileFloat(const std::string &fname_str, float **file_data, size_t *n_read);
int WriteFileFloat(const std::string &fname_str, const float *data, const size_t &n_values);

int FilePathBasename(const char *filename, std::string *filepath, std::string *basename);
std::string GetParamPath(const std::string &default_dir, const std::string &extension, const std::string &custom_dir, const Param &params);

#endif /* FILEIO_H_ */
