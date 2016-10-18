/*
 * DnnClass.cpp
 *
 *  Created on: 18 Oct 2016
 *      Author: ljuvela
 */


#include <gslwrap/vector_double.h>
#include <gslwrap/matrix_double.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "ReadConfig.h"
#include "DnnClass.h"



DnnLayer::DnnLayer(const gsl::matrix &W_init, const gsl::vector &b_init, const DnnActivationFunction &af) {
   this->W.copy(W_init);
   this->b.copy(b_init);
   this->activation_function = af;
}

DnnLayer::~DnnLayer() {};

void Dnn::addLayer(const DnnLayer &layer) {
   layer_list.push_back(layer);
}

DnnActivationFunction Dnn::ActivationParse(std::string &str) {
   if (!str.compare("S"))
      return SIGMOID;
   else if (!str.compare("L"))
      return LINEAR;
   else if (!str.compare("R"))
      return RELU;
   else if (!str.compare("T"))
      return TANH;
   else
      std::cout << "Warning: invalid activation function \"" << str << "\", using linear activation" << std::endl;

   // Default value
   return LINEAR;
}

int Dnn::ReadInfo(const char *basename) {

   /* Filename processing */
   std::string fname_str;
   fname_str += basename;
   fname_str += ".dnnInfo";
   std::cout << "Reading file " << fname_str  << std::endl;

   libconfig::Config cfg;

   /* Read the file. If there is an error, report it and exit. */
   try
   {
      cfg.readFile(fname_str.c_str());
   }
   catch(const libconfig::FileIOException &fioex)
   {
      std::cerr << "I/O error while reading file: " << fname_str << std::endl;
      return(EXIT_FAILURE);
   }
   catch(const libconfig::ParseException &pex)
   {
      std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                                      << " - " << pex.getError() << std::endl;
      return(EXIT_FAILURE);
   }


   const libconfig::Setting &layers = cfg.lookup("LAYERS");
   num_layers = layers.getLength();
   for (libconfig::Setting const &layer : layers)
      this->layer_sizes.push_back(layer);


   const libconfig::Setting &activations = cfg.lookup("ACTIVATIONS");
   std::vector<std::string> strs;
   for (libconfig::Setting const &af : activations) {
      strs.push_back(af);
      this->activation_functions.push_back(ActivationParse(strs.back()));
   }


/*
   size_t i;
   for(i=0;i<num_layers-1;i++) {
         gsl::matrix W(layer_sizes[i+1],layer_sizes[i]);
         gsl::vector b(layer_sizes[i]);
         DnnActivationFunction af = ActivationParse(activation_functions[i+1]);
         layer_list.push_back(DnnLayer(W,b,af));
   }
*/

  // std::cout << layer_sizes[0] << std::endl;
   //if (!layer_sizes.isArray())
     // return EXIT_FAILURE;


   //num_layers = layer_sizes.getLength();




   return EXIT_SUCCESS;
}

int Dnn::ReadData(const char *basename) {

   /* Filename processing */
   std::string fname_str;
   fname_str += basename;
   //fname_str += ".dnnData";
   fname_str += ".LSF";
   std::cout << "Reading file " << fname_str  << std::endl;

   std::ifstream file(fname_str, std::ios::in | std::ios::binary);
   if (!file)
      EXIT_FAILURE;

   std::streampos file_size;
   size_t n_values;

   // File size
   file.seekg(0, std::ios::end);
   file_size = file.tellg();
   file.seekg(0, std::ios::beg);

   n_values = file_size / sizeof(double);
   double *file_data = new double[n_values];
   file.read(reinterpret_cast<char*>(file_data), file_size);

   size_t i,j, ind=0;
   size_t row_size=30;
   size_t colum_size = 3;
   for(j=0;j<colum_size;j++)
      for(i=0;i<row_size;i++){
         std::cout << file_data[ind++] << std::endl;
      }
   delete[] file_data;


   return EXIT_SUCCESS;


}
