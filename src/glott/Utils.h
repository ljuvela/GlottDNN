#ifndef UTILS_H_
#define UTILS_H_
#include <vector>

gsl::vector StdVector2GslVector(const std::vector<double> &stdvec);
gsl::matrix ElementProduct(const gsl::matrix &A, const gsl::matrix &B);
gsl::matrix ElementDivision(const gsl::matrix &A, const gsl::matrix &B);


int CheckCommandLineAnalysis(int argc);

void CheckNanInf(gsl::vector &vec);

/* Debug functions */
void VPrint1(const gsl::vector &vector);
void VPrint2(const gsl::vector &vector);
void VPrint3(const gsl::vector &vector);
void VPrint4(const gsl::vector &vector);
void VPrint5(const gsl::vector &vector);

void MPrint1(const gsl::matrix &matrix);

#endif /* UTILS_H_ */
