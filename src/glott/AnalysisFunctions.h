/*
 * AnalysisFunctions.h
 *
 *  Created on: 30 Sep 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_ANALYSISFUNCTIONS_H_
#define SRC_GLOTT_ANALYSISFUNCTIONS_H_

int GetF0(const Param &params, const gsl::vector &signal, gsl::vector *fundf);
int GetGci(const Param &params, const gsl::vector &signal, gsl::vector_int *gci_inds);
int GetGain(const Param &params, const gsl::vector &signal, gsl::vector *gain);
int GetFrame(const Param &params, const gsl::vector &signal, const int frame_index,gsl::vector *frame, gsl::vector *pre_frame);
int SpectralAnalysis(const Param &params, const AnalysisData &data, gsl::matrix *vocal_tract_poly);
void GetPulses(const Param &params, const gsl::vector &source_signal, const gsl::vector_int &gci_inds, gsl::vector &fundf, gsl::matrix *pulses_mat);


#endif /* SRC_GLOTT_ANALYSISFUNCTIONS_H_ */
