#include "Workflow.h"

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "definitions.h"
#include "AnalysisFunctions.h"
#include "DnnClass.h"
#include "FileIo.h"
#include "InverseFiltering.h"
#include "ReadConfig.h"
#include "SpFunctions.h"
#include "SynthesisFunctions.h"
#include "Utils.h"

int RunAnalysis(const std::string &wav_filename,
                const std::string &default_config_filename,
                const std::string &user_config_filename) {
   Param params;
   if (ReadConfig(default_config_filename.c_str(), true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (!user_config_filename.empty() &&
       ReadConfig(user_config_filename.c_str(), false, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;

   AnalysisData data;
   if (ReadWavFile(wav_filename.c_str(), &(data.signal), &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   data.AllocateData(params);
   data.signal = HighPassFiltering(params, data.signal);

   if (!params.use_external_f0 || !params.use_external_gci ||
       params.signal_polarity == POLARITY_DETECT)
      GetIaifResidual(params, data.signal, &(data.source_signal_iaif));
   PolarityDetection(params, &(data.signal), &(data.source_signal_iaif));
   if (GetF0(params, data.signal, data.source_signal_iaif, &(data.fundf)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (GetGci(params, data.signal, data.source_signal_iaif, data.fundf,
              &(data.gci_inds)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   GetGain(params, data.fundf, data.signal, &(data.frame_energy));
   if (params.qmf_subband_analysis)
      SpectralAnalysisQmf(params, data, &(data.poly_vocal_tract));
   else
      SpectralAnalysis(params, data.signal, data.fundf, data.gci_inds,
                       &(data.poly_vocal_tract));
   Poly2Lsf(data.poly_vocal_tract, &data.lsf_vocal_tract);
   MedianFilter(5, &data.lsf_vocal_tract);
   MovingAverageFilter(3, &data.lsf_vocal_tract);
   Lsf2Poly(data.lsf_vocal_tract, &data.poly_vocal_tract);
   InverseFilter(params, data.signal, data.gci_inds, data.fundf,
                 data.frame_energy, data.poly_vocal_tract,
                 &(data.poly_glot), &(data.source_signal));
   if (params.use_waveforms_directly)
      GetPulses(params, data.signal, data.gci_inds, data.fundf, &(data.excitation_pulses));
   else
      GetPulses(params, data.source_signal, data.gci_inds, data.fundf, &(data.excitation_pulses));
   HnrAnalysis(params, data.source_signal, data.fundf, &(data.hnr_glot));
   Poly2Lsf(data.poly_vocal_tract, &(data.lsf_vocal_tract));
   Poly2Lsf(data.poly_glot, &(data.lsf_glot));
   data.SaveData(params);
   std::cout << "Finished analysis." << std::endl << std::endl;
   return EXIT_SUCCESS;
}

int AnalyzeSignal(const std::string &default_config_filename,
                  const std::string &user_config_filename,
                  const gsl::vector &signal, AnalysisData *data) {
   Param params;
   if (ReadConfig(default_config_filename.c_str(), true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (!user_config_filename.empty() &&
       ReadConfig(user_config_filename.c_str(), false, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   data->signal = signal;
   params.signal_length = signal.size();
   params.number_of_frames =
      static_cast<int>(ceil(static_cast<double>(signal.size()) /
                            static_cast<double>(params.frame_shift)));
   data->AllocateData(params);
   data->signal = HighPassFiltering(params, data->signal);
   if (!params.use_external_f0 || !params.use_external_gci ||
       params.signal_polarity == POLARITY_DETECT)
      GetIaifResidual(params, data->signal, &(data->source_signal_iaif));
   PolarityDetection(params, &(data->signal), &(data->source_signal_iaif));
   if (GetF0(params, data->signal, data->source_signal_iaif,
             &(data->fundf)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (GetGci(params, data->signal, data->source_signal_iaif, data->fundf,
              &(data->gci_inds)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   GetGain(params, data->fundf, data->signal, &(data->frame_energy));
   if (params.qmf_subband_analysis)
      SpectralAnalysisQmf(params, *data, &(data->poly_vocal_tract));
   else
      SpectralAnalysis(params, data->signal, data->fundf, data->gci_inds,
                       &(data->poly_vocal_tract));
   Poly2Lsf(data->poly_vocal_tract, &data->lsf_vocal_tract);
   MedianFilter(5, &data->lsf_vocal_tract);
   MovingAverageFilter(3, &data->lsf_vocal_tract);
   Lsf2Poly(data->lsf_vocal_tract, &data->poly_vocal_tract);
   InverseFilter(params, data->signal, data->gci_inds, data->fundf,
                 data->frame_energy, data->poly_vocal_tract,
                 &(data->poly_glot), &(data->source_signal));
   if (params.use_waveforms_directly)
      GetPulses(params, data->signal, data->gci_inds, data->fundf,
                &(data->excitation_pulses));
   else
      GetPulses(params, data->source_signal, data->gci_inds, data->fundf,
                &(data->excitation_pulses));
   HnrAnalysis(params, data->source_signal, data->fundf, &(data->hnr_glot));
   Poly2Lsf(data->poly_vocal_tract, &(data->lsf_vocal_tract));
   Poly2Lsf(data->poly_glot, &(data->lsf_glot));
   return EXIT_SUCCESS;
}

int RunSynthesis(const std::string &filename,
                 const std::string &default_config_filename,
                 const std::string &user_config_filename) {
   std::cout << "Synthesis of " << filename << std::endl;
   Param params;
   if (ReadConfig(default_config_filename.c_str(), true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (!user_config_filename.empty() &&
       ReadConfig(user_config_filename.c_str(), false, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;

   SynthesisData data;
   if (ReadSynthesisData(filename.c_str(), &params, &data) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (params.noise_gated_synthesis)
      NoiseGating(params, &(data.frame_energy));
   if (params.use_postfiltering)
      PostFilter(params.postfilter_coefficient, params.fs, data.fundf,
                 &(data.lsf_vocal_tract));
   if (params.use_postfiltering || params.use_spectral_matching)
      PostFilter(params.postfilter_coefficient_glot, params.fs, data.fundf,
                 &(data.lsf_glot));
   if (params.use_trajectory_smoothing)
      ParameterSmoothing(params, &data);
   StabilizeLsf(&(data.lsf_vocal_tract));
   if (params.use_spectral_matching)
      StabilizeLsf(&(data.lsf_glot));
   if (CreateExcitation(params, data.fundf, data.frame_energy,
                        data.excitation_pulses,
                        data.lsf_vocal_tract, data.lsf_glot, data.hnr_glot,
                        &(data.excitation_signal)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (params.noise_gain_voiced > 0.0)
      HarmonicModification(params, data, &(data.excitation_signal));
   FftFilterExcitation(params, data, &(data.signal));
   GenerateUnvoicedSignal(params, data, &(data.signal));
   std::string out_fname = GetParamPath("exc", ".exc.wav", params.dir_exc, params);
   if (WriteWavFile(out_fname, data.excitation_signal, params.fs) == EXIT_FAILURE)
      return EXIT_FAILURE;
   out_fname = GetParamPath("syn", ".syn.wav", params.dir_syn, params);
   std::cout << out_fname << std::endl;
   if (WriteWavFile(out_fname, data.signal, params.fs) == EXIT_FAILURE)
      return EXIT_FAILURE;
   std::cout << "Finished synthesis" << std::endl;
   return EXIT_SUCCESS;
}

int SynthesizeData(const std::string &default_config_filename,
                   const std::string &user_config_filename,
                   SynthesisData *data, gsl::vector *signal,
                   gsl::vector *excitation) {
   Param params;
   if (ReadConfig(default_config_filename.c_str(), true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (!user_config_filename.empty() &&
       ReadConfig(user_config_filename.c_str(), false, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   params.number_of_frames = static_cast<int>(data->fundf.size());
   params.signal_length = static_cast<int>(
      rint(params.number_of_frames * params.frame_shift / params.speed_scale));
   data->signal = gsl::vector(params.signal_length, true);
   data->excitation_signal = gsl::vector(params.signal_length, true);
   if (params.noise_gated_synthesis)
      NoiseGating(params, &(data->frame_energy));
   if (params.use_postfiltering)
      PostFilter(params.postfilter_coefficient, params.fs, data->fundf,
                 &(data->lsf_vocal_tract));
   if (params.use_postfiltering || params.use_spectral_matching)
      PostFilter(params.postfilter_coefficient_glot, params.fs, data->fundf,
                 &(data->lsf_glot));
   if (params.use_trajectory_smoothing)
      ParameterSmoothing(params, data);
   StabilizeLsf(&(data->lsf_vocal_tract));
   if (params.use_spectral_matching)
      StabilizeLsf(&(data->lsf_glot));
   if (CreateExcitation(params, data->fundf, data->frame_energy,
                        data->excitation_pulses, data->lsf_vocal_tract,
                        data->lsf_glot, data->hnr_glot, excitation) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (params.noise_gain_voiced > 0.0)
      HarmonicModification(params, *data, excitation);
   FftFilterExcitation(params, *data, signal);
   GenerateUnvoicedSignal(params, *data, signal);
   return EXIT_SUCCESS;
}
