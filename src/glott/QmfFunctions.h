#ifndef SRC_GLOTT_QMFFUNCTIONS_H_
#define SRC_GLOTT_QMFFUNCTIONS_H_

#include <vector>
#include "ComplexVector.h"
#include "definitions.h"

namespace Qmf {

gsl::vector GetMatchingFilter(const gsl::vector &H0);

void GetSubBands(const gsl::vector &frame, const gsl::vector &H0, const gsl::vector &H1,
                  gsl::vector *frame_qmf1, gsl::vector *frame_qmf2);

void Decimate(const gsl::vector &frame_orig, const int skip, gsl::vector *frame_decimated);

void CombinePoly(const gsl::vector &a_qmf1, const gsl::vector &a_qmf2,
               const double &qmf_gain, const int &Nsub, gsl::vector *a_combined);

}


#endif // QMFFUNCTIONS_H_INCLUDED
