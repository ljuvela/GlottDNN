/*
 * SynthesisFunctions.h
 *
 *  Created on: 13 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_SYNTHESISFUNCTIONS_H_
#define SRC_GLOTT_SYNTHESISFUNCTIONS_H_

void CreateExcitation(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal);
void HarmonicModification(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal);
void SpectralMatchExcitation(const Param &params,const SynthesisData &data, gsl::vector *excitation_signal);
void FilterExcitation(const Param &params, const SynthesisData &data, gsl::vector *signal);

#endif /* SRC_GLOTT_SYNTHESISFUNCTIONS_H_ */
