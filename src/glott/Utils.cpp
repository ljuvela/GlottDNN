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
      std::cout << "    Analysis wav_file config_default config_user\n"                << std::endl;
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
