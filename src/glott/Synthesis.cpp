/*
 * Synthesis.cpp
 *
 *  Created on: 13 Oct 2016
 *      Author: ljuvela
 */

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

   /* Check LSF stability and fix if needed */
   StabilizeLsf(&data.lsf_vocal_tract);
   StabilizeLsf(&data.lsf_glot);

   if(params.use_postfiltering)
      PostFilter(params.postfilter_coefficient, params.fs, &data.lsf_vocal_tract);

   if(params.use_trajectory_smoothing)
      ParameterSmoothing(params, &data);


   CreateExcitation(params, data, &(data.excitation_signal));

   SpectralMatchExcitation(params, data, (&data.excitation_signal));

  HarmonicModification(params, data, (&data.excitation_signal));

   FilterExcitation(params, data, &(data.signal));




   if(WriteWavFile(filename, ".exc.wav", data.excitation_signal, params.fs) == EXIT_FAILURE)
       return EXIT_FAILURE;

   if(WriteWavFile(filename, ".syn.wav", data.signal, params.fs) == EXIT_FAILURE)
       return EXIT_FAILURE;

   std::cout << "Finished synthesis" << std::endl;

}
