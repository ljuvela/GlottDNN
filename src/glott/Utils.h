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

#ifndef UTILS_H_
#define UTILS_H_
#include <vector>

gsl::vector StdVector2GslVector(const std::vector<double> &stdvec);
gsl::matrix ElementProduct(const gsl::matrix &A, const gsl::matrix &B);
gsl::matrix ElementDivision(const gsl::matrix &A, const gsl::matrix &B);


int CheckCommandLineAnalysis(int argc);
int CheckCommandLineSynthesis(int argc);

void CheckNanInf(gsl::vector &vec);

/* Debug functions */
void VPrint1(const gsl::vector &vector);
void VPrint2(const gsl::vector &vector);
void VPrint3(const gsl::vector &vector);
void VPrint4(const gsl::vector &vector);
void VPrint5(const gsl::vector &vector);

void MPrint1(const gsl::matrix &matrix);

#endif /* UTILS_H_ */
