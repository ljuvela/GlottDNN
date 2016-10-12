/*
 * PitchEstimation.h
 *
 *  Created on: 11 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_PITCHESTIMATION_H_
#define SRC_GLOTT_PITCHESTIMATION_H_

#define NUMBER_OF_F0_CANDIDATES 3
#define F0_INTERP_SAMPLES 7
#define FFT_LENGTH 4096

void FundamentalFrequency(const Param &params, const gsl::vector &glottal_frame,
      const gsl::vector &signal_frame, double *fundf, gsl::vector *fundf_candidates);

#endif /* SRC_GLOTT_PITCHESTIMATION_H_ */
