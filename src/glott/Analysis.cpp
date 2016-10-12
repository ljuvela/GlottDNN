//   MIT License
//
//   Copyright (c) 2016 ljuvela
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
//
//
//  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//               GlottDNN Speech Parameter Extractor
//  <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//
//  This program reads a speech file and extracts speech
//  parameters using glottal inverse filtering.
//
//  This program has been written in Aalto University,
//  Department of Signal Processing and Acoustics, Espoo, Finland
//
//  This program uses some code from the original GlottHMM vocoder program
//  written by Tuomo Raitio, now re-factored and re-written in C++
//
//  Authors: Lauri Juvela, Manu Airaksinen
//  Acknowledgements: Tuomo Raitio, Paavo Alku
//
//  File Analysis.cpp
//  Version: 1.0


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

#include "Filters.h"
#include "definitions.h"
#include "FileIo.h"
#include "ReadConfig.h"
#include "SpFunctions.h"
#include "AnalysisFunctions.h"

#include "DebugUtils.h"


/*******************************************************************/
/*                          MAIN                                   */
/*******************************************************************/

int main(int argc, char *argv[]) {

   const char *wav_filename = argv[1];
   const char *default_config_filename = argv[2];
   const char *user_config_filename = argv[3];

   /* Read configuration file */
   Param params;
   if (ReadConfig(default_config_filename, true, &params) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (argc > 3) {
      if (ReadConfig(user_config_filename, false, &params) == EXIT_FAILURE)
         return EXIT_FAILURE;
   }

   //create_file(fname, SF_FORMAT_WAV | SF_FORMAT_PCM_16) ;

   /* Read sound file and allocate data */
   AnalysisData data;

   if(ReadWavFile(wav_filename, &(data.signal), &params) == EXIT_FAILURE)
      return EXIT_FAILURE;

   data.AllocateData(params);

   /* High-pass filter signal to eliminate low frequency "rumble" */
   HighPassFiltering(params, &(data.signal));

   // IAIF analysis if( PolarityDetection || GetF0 || GetGci)

   /* Read or estimate signal polarity */
   PolarityDetection(params, &(data.signal), &(data.source_signal_iaif));

   /* Read or estimate fundamental frequency (F0)  */
   GetF0(params, data.signal, &(data.fundf), &(data.source_signal_iaif));

   /* Read or estimate glottal closure instants (GCIs)*/
   if(GetGci(params, data.signal, data.fundf, &(data.gci_inds), &(data.source_signal_iaif)) == EXIT_FAILURE)
      return EXIT_FAILURE;


   /* Estimate frame log-energy (Gain) */
   GetGain(params, data.signal, &(data.frame_energy));



   /* Spectral analysis for vocal tract transfer function*/
   if(params.qmf_subband_analysis) {
      SpectralAnalysisQmf(params, data, &(data.poly_vocal_tract));
   } else {
      SpectralAnalysis(params, data, &(data.poly_vocal_tract));
   }

   /* Convert vocal tract AR polynomials to LSF */
   Poly2Lsf(data.poly_vocal_tract, &(data.lsf_vocal_tract));

   /* Perform glottal inverse filtering with the estimated VT AR polynomials */
   InverseFilter(params, data, &(data.poly_glott), &(data.source_signal));

   /* Convert glottal source AR polynomials to LSF */
   Poly2Lsf(data.poly_glott, &(data.lsf_glott));

   /* Extract pitch synchronous (excitation) waveforms at each frame */
   if (params.use_waveforms_directly)
      GetPulses(params, data.signal, data.gci_inds, data.fundf, &(data.excitation_pulses));
   else
      GetPulses(params, data.source_signal, data.gci_inds, data.fundf, &(data.excitation_pulses));

   /* Write analyzed features to files */
   data.SaveData(params);

   /* Finish */
   std::cout << "Finished analysis." << std::endl << std::endl;
   return EXIT_SUCCESS;

}

/***********/
/*   EOF   */
/***********/

