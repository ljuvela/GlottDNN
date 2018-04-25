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

#ifndef SRC_GLOTT_ANALYSISFUNCTIONS_H_
#define SRC_GLOTT_ANALYSISFUNCTIONS_H_

void HighPassFiltering(const Param &params, gsl::vector *signal);
int PolarityDetection(const Param &params, gsl::vector *signal, gsl::vector *source_signal_iaif);
int GetF0(const Param &params, const gsl::vector &signal, const gsl::vector &source_signal_iaif, gsl::vector *fundf);
int GetGci(const Param &params, const gsl::vector &signal, const gsl::vector &source_signal_iaif, const gsl::vector &fundf, gsl::vector_int *gci_inds);
int GetGain(const Param &params, const gsl::vector &fundf, const gsl::vector &signal, gsl::vector *gain);
int SpectralAnalysis(const Param &params, const AnalysisData &data, gsl::matrix *vocal_tract_poly);
int SpectralAnalysisQmf(const Param &params, const AnalysisData &data, gsl::matrix *poly_vocal_tract);
int InverseFilter(const Param &params, const AnalysisData &data, gsl::matrix *poly_glott, gsl::vector *source_signal);
void GetPulses(const Param &params, const gsl::vector &source_signal, const gsl::vector_int &gci_inds, gsl::vector &fundf, gsl::matrix *pulses_mat);
void GetIaifResidual(const Param &params, const gsl::vector &signal, gsl::vector *residual);
void HnrAnalysis(const Param &params, const gsl::vector &source_signal, const gsl::vector &fundf, gsl::matrix *hnr_glott);
int GetPitchSynchFrame(const Param &params, const gsl::vector &signal, const gsl::vector_int &gci_inds,
                        const int &frame_index, const int &frame_shift, const double &f0,
                        gsl::vector *frame, gsl::vector *pre_frame);
#endif /* SRC_GLOTT_ANALYSISFUNCTIONS_H_ */
