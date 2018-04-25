// Copyright 2016-2018 Lauri Juvela and Manu Airaksinen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
