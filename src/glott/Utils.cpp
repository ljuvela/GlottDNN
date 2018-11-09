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

#include <gslwrap/vector_double.h>
#include <vector>
#include <cassert>
#include <iostream>
#include "definitions.h"
#include "Utils.h"

gsl::vector StdVector2GslVector(const std::vector<double> &stdvec) {
   gsl::vector gslvec(stdvec.size());
   size_t i;
   for(i=0;i<gslvec.size();i++)
      gslvec(i) = stdvec[i];

   return gslvec;
}


gsl::matrix ElementProduct(const gsl::matrix &A, const gsl::matrix &B) {
   assert(A.size1() == B.size1());
   assert(A.size2() == B.size2());
   gsl::matrix C(A);

   for(size_t i=0; i<A.size1();i++)
      for(size_t j=0; j<A.size2(); j++)
         C(i,j) = A(i,j)*B(i,j);

   return C;
}

gsl::matrix ElementDivision(const gsl::matrix &A, const gsl::matrix &B) {
   assert(A.size1() == B.size1());
   assert(A.size2() == B.size2());
   gsl::matrix C(A);

   for(size_t i=0; i<A.size1();i++)
      for(size_t j=0; j<A.size2(); j++)
         C(i,j) = A(i,j)/B(i,j);

   return C;
}

/**
 * Replace Nan and Inf values in vector
 * Prints a warning message if invalid values are found
 */
void CheckNanInf(gsl::vector &vec) {
   size_t nan_count = 0;
   size_t inf_count = 0;

   for (size_t i=0; i<vec.size(); i++) {
      if (isinf(vec(i))) {
         vec(i) = 0.0;
         inf_count++;
      }
      if (isnan(vec(i))) {
         vec(i) = 0.0;
         nan_count++;
      }
   }

   if (nan_count > 0 || inf_count > 0)
      std::cerr << "Warning: NaN values (" << nan_count << ") " <<
      "and Inf values (" << inf_count << ") found, replaced with zeros" << std::endl;

}

/**
 * Function CheckCommandLineAnalysis
 *
 * Check command line format and print instructions
 *
 * @param argc number of input arguments
 */
int CheckCommandLineAnalysis(int argc) {

   /* Check command line format */
   if (argc < 3 || argc > 4) {

      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      std::cout << "            GlottDNN - Speech Parameter Extractor "                << std::endl;
      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      std::cout << "Description:\n"                                                    << std::endl;
      std::cout << "    Extraction of speech signal into vocal tract filter and voice" << std::endl;
      std::cout << "    source parameters using glottal inverse filtering.\n"          << std::endl;
      std::cout << "Usage:\n"                                                          << std::endl;
      std::cout << "    Analysis wav_file.wav config_default.cfg (config_user.cfg)\n"  << std::endl;
      std::cout << " wav_file.wav        - Name of the audio file to be analysed"      << std::endl;
      std::cout << " config_default.cfg  - Name of the default config file"            << std::endl;
      std::cout << " config_user.cfg     - Name of the user config file (OPTIONAL)"    << std::endl;
      //std::cout << "Version:\n"  << std::endl;
      //std::cout << "    %s (%s)\n\n",VERSION,DATE);
      return EXIT_FAILURE;
   } else {

      /* Print program description */
      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      std::cout << "            GlottDNN - Speech Parameter Extractor "                << std::endl;
      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      return EXIT_SUCCESS;
   }
}

int CheckCommandLineSynthesis(int argc) {

   /* Check command line format */
   if (argc < 3 || argc > 4) {

      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      std::cout << "            GlottDNN - Speech Waveform Synthesis                "  << std::endl;
      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      std::cout << "Description:\n"                                                    << std::endl;
      std::cout << "    Synthesis of speech signal from glottal vocoder parameters \n" << std::endl;
      std::cout << "Usage:\n"                                                          << std::endl;
      std::cout << "    Synthesis wav_file.wav config_default.cfg (config_user.cfg)\n" << std::endl;
      std::cout << " basename            - Basename for synthesized audio           "  << std::endl;
      std::cout << " config_default.cfg  - Name of the default config file          "  << std::endl;
      std::cout << " config_user.cfg     - Name of the user config file (OPTIONAL)"    << std::endl;
      //std::cout << "Version:\n"  << std::endl;
      //std::cout << "    %s (%s)\n\n",VERSION,DATE);
      return EXIT_FAILURE;
   } else {

      /* Print program description */
      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      std::cout << "            GlottDNN - Speech Waveform Synthesis                "  << std::endl;
      std::cout << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << std::endl;
      return EXIT_SUCCESS;
   }
}


/*********************************************************************/
/*                       TEST FUNCTIONS                              */
/*********************************************************************/

// TEST FUNCTION
// PRINT VECTOR TO FILE: p1.dat
void VPrint1(const gsl::vector &vector) {
	FILE *fid = fopen("p1.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint2(const gsl::vector &vector) {
	FILE *fid = fopen("p2.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint3(const gsl::vector &vector) {
	FILE *fid = fopen("p3.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint4(const gsl::vector &vector) {
	FILE *fid = fopen("p4.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint5(const gsl::vector &vector) {
	FILE *fid = fopen("p5.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void MPrint1(const gsl::matrix &matrix) {
   FILE *fid = fopen("m1.dat", "w");
   matrix.fprintf(fid,"%.30f");
   fclose(fid);
}
