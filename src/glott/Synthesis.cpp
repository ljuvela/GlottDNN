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

#include "Filters.h"
#include "definitions.h"
#include "FileIo.h"
#include "ReadConfig.h"
#include "SpFunctions.h"
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

   //ParameterSmoothing()

   //PostFiltering

   //ConvertLSFs

   CreateExcitation(params, data, &(data.excitation_signal));

   VPrint1(data.excitation_signal);


   std::cout << "Finished synthesis" << std::endl;

}
