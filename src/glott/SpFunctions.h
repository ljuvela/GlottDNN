/*
 * SpFunctions.h
 *
 *  Created on: 3 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_SPFUNCTIONS_H_
#define SRC_GLOTT_SPFUNCTIONS_H_
#include <vector>
#include "definitions.h"

void InterpolateNearest(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void InterpolateLinear(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void Filter(const gsl::vector &b, const gsl::vector &a, const gsl::vector &x, gsl::vector *y);
void Filter(const std::vector<double> &b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y);
void Filter(const gsl::vector b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y);
void Filter(const std::vector<double> &b, const gsl::vector a, const gsl::vector &x, gsl::vector *y);
void ApplyWindowingFunction(const WindowingFunctionType &window_function, gsl::vector *frame);
void Autocorrelation(const gsl::vector &frame, const int &order, gsl::vector *r);
void Levinson(const gsl::vector &r, gsl::vector *A);

#endif /* SRC_GLOTT_SPFUNCTIONS_H_ */
