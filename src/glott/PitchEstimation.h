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
