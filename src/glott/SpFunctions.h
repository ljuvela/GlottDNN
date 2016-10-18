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

void Interpolate(const gsl::vector &vector, gsl::vector *i_vector);
void InterpolateNearest(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
double InterpolateLinear(const double &val1, const double &val2, double interp_x);
void InterpolateLinear(const gsl::matrix &mat, const double frame_index, gsl::vector *i_vector);
void InterpolateLinear(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void InterpolateLinear(const gsl::vector &x_orig, const gsl::vector &y_orig, const gsl::vector &x_interp, gsl::vector *y_interp);
void InterpolateSpline(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector);
void Filter(const gsl::vector &b, const gsl::vector &a, const gsl::vector &x, gsl::vector *y);
void Filter(const std::vector<double> &b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y);
void Filter(const gsl::vector b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y);
void Filter(const std::vector<double> &b, const gsl::vector a, const gsl::vector &x, gsl::vector *y);
gsl::vector Conv(const gsl::vector &conv1, const gsl::vector &conv2);
void ApplyWindowingFunction(const WindowingFunctionType &window_function, gsl::vector *frame);
void Autocorrelation(const gsl::vector &frame, const int &order, gsl::vector *r);
void Levinson(const gsl::vector &r, gsl::vector *A);
void Lsf2Poly(const gsl::vector &lsf_vec, gsl::vector *poly_vec);
void Lsf2Poly(const gsl::matrix &lsf_mat, gsl::matrix *poly_mat);
void Poly2Lsf(const gsl::vector &a, gsl::vector *lsf);
void Poly2Lsf(const gsl::matrix &a_mat, gsl::matrix *lsf_mat);
void Roots(const gsl::vector &x, ComplexVector *r);
void Roots(const gsl::vector &x, const size_t ncoef, ComplexVector *r);
void AllPassDelay(const double &lambda, gsl::vector *signal);
void ConcatenateFrames(const gsl::vector &frame1, const gsl::vector &frame2, gsl::vector *frame_result);
int NextPow2(int n);
bool IsPow2(int n);
void FFTRadix2(const gsl::vector &x, ComplexVector *X);
void FFTRadix2(const gsl::vector &x, size_t nfft, ComplexVector *X);
void IFFTRadix2(const ComplexVector &X, gsl::vector *x);
void WFilter(const gsl::vector &A, const gsl::vector &B,const gsl::vector &signal,const double &lambda, gsl::vector *result);
void WarpingAlphas2Sigmas(const gsl::vector &alp, const double &lambda, gsl::vector *sigm);
void OverlapAdd(const gsl::vector &frame, const size_t center_index, gsl::vector *target);
double getMean(const gsl::vector &vec);
double getMeanF0(const gsl::vector &fundf);
double getEnergy(const gsl::vector &vec);
double getSquareSum(const gsl::vector &vec);
double LogEnergy2FrameEnergy(const double &log_energy, const size_t frame_size);
double FrameEnergy2LogEnergy(const double &frame_energy, const size_t frame_size);
double Skewness(const gsl::vector &data);
int FindPeaks(const gsl::vector &vec, const double &threshold, gsl::vector_int *index, gsl::vector *value);
gsl::vector_int FindHarmonicPeaks(const gsl::vector &fft_mag, const double &f0, const int &fs);
void StabilizePoly(const int &fft_length, gsl::vector *A);
gsl::vector_int LinspaceInt(const int &start_val, const int &hop_val,const int &end_val);
void Linear2Erb(const gsl::vector &linvec, const int &fs, gsl::vector *erbvec);
void Erb2Linear(const gsl::vector &vector_erb, const int &fs,  gsl::vector *vector_lin);
int GetFrame(const gsl::vector &signal, const int &frame_index, const int &frame_shift,gsl::vector *frame, gsl::vector *pre_frame);
double GetFilteringGain(const gsl::vector &b, const gsl::vector &a,
                        const gsl::vector &signal, const double &target_gain_db, const size_t &center_index,
                        const size_t &frame_length, const double &warping_lambda);
void SharpenPowerSpectrumPeaks(const gsl::vector_int &peak_indices, const double &gamma, const int &power_spectrum_win, gsl::vector *fft_pow);
void StabilizeLsf(gsl::matrix *lsf);
void StabilizeLsf(gsl::vector *lsf);
void MedianFilter(const gsl::vector &x, const size_t &filterlen, gsl::vector *y);
void MedianFilter(const size_t &filterlen, gsl::vector *x);
void MedianFilter(const size_t &filterlen, gsl::matrix *mat);
void MovingAverageFilter(const gsl::vector &x, const size_t &filterlen, gsl::vector *y);
void MovingAverageFilter(const size_t &filterlen, gsl::vector *x);
void MovingAverageFilter(const size_t &filterlen, gsl::matrix *mat);

#endif /* SRC_GLOTT_SPFUNCTIONS_H_ */
