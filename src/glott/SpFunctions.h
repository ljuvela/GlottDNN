/*
 * SpFunctions.h
 *
 *  Created on: 3 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_SPFUNCTIONS_H_
#define SRC_GLOTT_SPFUNCTIONS_H_
#include <vector>
#include "ComplexVector.h"
#include "definitions.h"

void InterpolateNearest(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void InterpolateLinear(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void InterpolateSpline(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void Filter(const gsl::vector &b, const gsl::vector &a, const gsl::vector &x, gsl::vector *y);
void Filter(const std::vector<double> &b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y);
void Filter(const gsl::vector b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y);
void Filter(const std::vector<double> &b, const gsl::vector a, const gsl::vector &x, gsl::vector *y);
void ApplyWindowingFunction(const WindowingFunctionType &window_function, gsl::vector *frame);
void Autocorrelation(const gsl::vector &frame, const int &order, gsl::vector *r);
void Levinson(const gsl::vector &r, gsl::vector *A);
void Poly2Lsf(const gsl::vector &a, gsl::vector *lsf);
void Poly2Lsf(const gsl::matrix &a_mat, gsl::matrix *lsf_mat);
void Roots(const gsl::vector &x, ComplexVector *r);
void Roots(const gsl::vector &x, const size_t ncoef, ComplexVector *r);
void AllPassDelay(const double &lambda, gsl::vector *signal);
void ConcatenateFrames(const gsl::vector &frame1, const gsl::vector &frame2, gsl::vector *frame_result);
int NextPow2(int n);
bool IsPow2(int n);
void FFTRadix2(const gsl::vector &x, ComplexVector *X);
void FFTRadix2(const gsl::vector &x, const size_t nfft, ComplexVector *X);
void IFFTRadix2(const ComplexVector &X, gsl::vector *x);
void WFilter(const gsl::vector &A, const gsl::vector &B,const gsl::vector &signal,const double &lambda, gsl::vector *result);
void WarpingAlphas2Sigmas(double *alp, double *sigm, double lambda, int dim);
void OverlapAdd(const gsl::vector &frame, const size_t center_index, gsl::vector *target);
double getMean(const gsl::vector &vec);
double getEnergy(const gsl::vector &vec);
double LogEnergy2FrameEnergy(const double &log_energy, const size_t frame_size);
double Skewness(const gsl::vector &data);


#endif /* SRC_GLOTT_SPFUNCTIONS_H_ */
