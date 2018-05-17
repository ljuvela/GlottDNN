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
