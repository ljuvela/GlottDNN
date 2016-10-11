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

   /* Read or estimate signal polarity */
   PolarityDetection(params, &(data.signal), &(data.source_signal_iaif));

   /* Read or estimate fundamental frequency (F0)  */
   GetF0(params, data.signal, &(data.fundf), &(data.source_signal_iaif));

   /* Read or estimate glottal closure instants (GCIs)*/
   GetGci(params, data.signal, &(data.gci_inds), &(data.source_signal_iaif));

   /* Estimate frame log-energy (Gain) */
   GetGain(params, data.signal, &(data.frame_energy));

   /* Spectral analysis for vocal tract transfer function*/
   SpectralAnalysis(params, data, &(data.poly_vocal_tract));

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

