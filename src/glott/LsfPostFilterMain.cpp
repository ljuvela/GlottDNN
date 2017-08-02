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

   std::string usage_string = "Usage: LsfPostFilter <config.cfg> <in_filename> <out_filename> ";

   if (argc < 4) {
      std::cout << usage_string << std::endl;
      return EXIT_FAILURE;
   }

   const char *config_filename = argv[1];
   const std::string in_filename = argv[2];
   const std::string out_filename = argv[3];

   /* Read configuration file */
   Param params;
   if (ReadConfig(config_filename, false, &params) == EXIT_FAILURE) {
      std::cout << usage_string << std::endl;
      //std::cerr << "Error: could not read config file" <<  std::endl;
      return EXIT_FAILURE;
   }

   gsl::matrix lsf;
   if (ReadGslMatrix(in_filename, params.data_type, params.lpc_order_vt, &lsf) == EXIT_FAILURE) {
      std::cout << "Error: Could not read " << in_filename << ", check the file and correct LPC order";
      return EXIT_FAILURE;
   }

   // dummy F0 indicating all frames are voiced
   gsl::vector fundf(lsf.get_cols());
   fundf.set_all(1.0);

   PostFilter(params.postfilter_coefficient, params.fs, fundf, &lsf);

   /* Check LSF stability and fix if needed */
   StabilizeLsf(&lsf);

   if (WriteGslMatrix(out_filename, params.data_type, lsf) == EXIT_FAILURE) {
      std::cerr << "Error: could not write " << out_filename << std::endl;
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
