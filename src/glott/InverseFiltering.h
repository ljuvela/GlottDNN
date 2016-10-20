/*
 * InverseFiltering.h
 *
 *  Created on: 3 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_INVERSEFILTERING_H_
#define SRC_GLOTT_INVERSEFILTERING_H_

void GetLpWeight(const Param &params, const LpWeightingFunction &weight_type,
						const gsl::vector_int &gci_inds, const gsl::vector &frame,
						const size_t &frame_index, gsl::vector *weight_function);

void ArAnalysis(const int &lp_order,const double &warping_lambda, const LpWeightingFunction &weight_type,
                  const gsl::vector &lp_weight, const gsl::vector &frame, gsl::vector *A);

void WWLP(const gsl::vector &weight_function, const double &warping_lambda, const LpWeightingFunction weight_type,
		const int &lp_order, const gsl::vector &frame, gsl::vector *A);

void LPC(const gsl::vector &frame, const int &lpc_order, gsl::vector *A);

void MeanBasedSignal(const gsl::vector &signal, const int &fs, const double &mean_f0, gsl::vector *mean_based_signal);

void SedreamsGciDetection(const gsl::vector &residual, const gsl::vector &mean_based_signal, gsl::vector_int *gci_inds);

#endif /* SRC_GLOTT_INVERSEFILTERING_H_ */
