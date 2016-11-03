/*
 * PitchEstimation.h
 *
 *  Created on: 11 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_PITCHESTIMATION_H_
#define SRC_GLOTT_PITCHESTIMATION_H_

#define NUMBER_OF_F0_CANDIDATES 2
#define F0_INTERP_SAMPLES 7
#define FFT_LENGTH 4096

void FundamentalFrequency(const Param &params, const gsl::vector &glottal_frame,
      const gsl::vector &signal_frame, double *fundf, gsl::vector *fundf_candidates);

void FillF0Gaps(gsl::vector *fundf_ptr);
void FundfPostProcessing(const Param &params, const gsl::vector &fundf_orig, const gsl::matrix &fundf_candidates, gsl::vector *fundf_ptr);

#endif /* SRC_GLOTT_PITCHESTIMATION_H_ */
