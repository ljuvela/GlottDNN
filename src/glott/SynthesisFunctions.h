/*
 * SynthesisFunctions.h
 *
 *  Created on: 13 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_SYNTHESISFUNCTIONS_H_
#define SRC_GLOTT_SYNTHESISFUNCTIONS_H_

void ParameterSmoothing(const Param &params, SynthesisData *data);
void PostFilter(const double &postfilter_coefficient, const int &fs, const gsl::vector &fundf, gsl::matrix *lsf);
int CreateExcitation(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal);
void HarmonicModification(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal);
void SpectralMatchExcitation(const Param &params,const SynthesisData &data, gsl::vector *excitation_signal);
void GenerateUnvoicedSignal(const Param &params, const SynthesisData &data, gsl::vector *signal);
void FilterExcitation(const Param &params, const SynthesisData &data, gsl::vector *signal);
void FftFilterExcitation(const Param &params, const SynthesisData &data, gsl::vector *signal);
void NoiseGating(const Param &params, gsl::vector *frame_energy);
#endif /* SRC_GLOTT_SYNTHESISFUNCTIONS_H_ */
