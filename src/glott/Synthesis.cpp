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

/***********************************************/
/*                 INCLUDE                     */
/***********************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include <iostream>
#include <iomanip>

#include <vector>
#include <gslwrap/vector_double.h>
#include <gslwrap/vector_int.h>

#include "Filters.h"
#include "definitions.h"
#include "FileIo.h"
#include "ReadConfig.h"
#include "SpFunctions.h"
#include "DnnClass.h"
#include "SynthesisFunctions.h"

#include "Utils.h"


/*******************************************************************/
/*                          MAIN                                   */
/*******************************************************************/

int main(int argc, char *argv[]) {


    if (CheckCommandLineSynthesis(argc) == EXIT_FAILURE) {
      return EXIT_FAILURE;
    }

   if (argc < 3 || argc > 4) {
      std::cout << "Usage: Synthesis <basename.wav> <config_default.cfg> (<config_usr.cfg>)" << std::endl;
   }

   const char *filename = argv[1];
   const char *default_config_filename = argv[2];
   const char *user_config_filename = argv[3];

   std::cout << "Synthesis of " << filename << std::endl;

   /* Read configuration file */
   Param params;
   if (ReadConfig(default_config_filename, true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (argc > 3) {
      if (ReadConfig(user_config_filename, false, &params) == EXIT_FAILURE)
         return EXIT_FAILURE;
   }

   SynthesisData data;
   if(ReadSynthesisData(filename, &params, &data) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if(params.noise_gated_synthesis)
      NoiseGating(params, &(data.frame_energy));
   
   if(params.use_postfiltering) {
      PostFilter(params.postfilter_coefficient, params.fs, data.fundf, &(data.lsf_vocal_tract));
   }

   if(params.use_postfiltering || params.use_spectral_matching) {
      PostFilter(params.postfilter_coefficient_glot, params.fs, data.fundf, &(data.lsf_glot));
   }

   if(params.use_trajectory_smoothing)
      ParameterSmoothing(params, &data);

   /* Check LSF stability and fix if needed */
   StabilizeLsf(&(data.lsf_vocal_tract));
   if (params.use_spectral_matching)
      StabilizeLsf(&(data.lsf_glot));

   /* Create excitation with overlap-add or read external excitation file */
   if (CreateExcitation(params, data, &(data.excitation_signal)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   /* Add noise to excitation to satisfy Harmonic-to-noise ratio*/
   if(params.noise_gain_voiced > 0.0)
      HarmonicModification(params, data, &(data.excitation_signal));

   /* Excitation spectral matching */
   //if(params.use_spectral_matching)
   //   SpectralMatchExcitation(params, data, &(data.excitation_signal)); // Use only with direct form filtering
   
   //FilterExcitation(params, data, &(data.signal));

   /* FFT based filtering includes spectral matching */ 
   FftFilterExcitation(params, data, &(data.signal)); 
   GenerateUnvoicedSignal(params, data, &(data.signal));
   
   std::string out_fname;
   out_fname = GetParamPath("exc", ".exc.wav", params.dir_exc, params);
   if(WriteWavFile(out_fname, data.excitation_signal, params.fs) == EXIT_FAILURE)
       return EXIT_FAILURE;

   out_fname = GetParamPath("syn", ".syn.wav", params.dir_syn, params);
   std::cout << out_fname << std::endl;
   if(WriteWavFile(out_fname, data.signal, params.fs) == EXIT_FAILURE)
       return EXIT_FAILURE;

   std::cout << "Finished synthesis" << std::endl;

   return EXIT_SUCCESS;
}
