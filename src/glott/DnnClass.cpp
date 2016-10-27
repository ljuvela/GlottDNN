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
#include <cassert>
#include <cmath>
#include <libconfig.h++>
#include "ReadConfig.h"
#include "Utils.h"
#include "DnnClass.h"



DnnLayer::DnnLayer(const gsl::matrix &W_init, const gsl::matrix &b_init, const DnnActivationFunction &af) {
   this->W.copy(W_init);
   this->b.copy(b_init);
   this->activation_function = af;
   this->layer_output = gsl::matrix(W_init.get_rows(), 1, true);
}

DnnLayer::~DnnLayer() {}

void DnnLayer::ForwardPass(const gsl::matrix &input) {

   // Check correct size
   size_t in_size = W.size2();
   //size_t out_size = W.size1();
   assert(in_size == input.size1());


   // Matrix multiplication
   layer_output = (this->W * input) + this->b;

   // Activation Function
   switch(this->activation_function) {
   case SIGMOID:
      ApplySigmoid(&layer_output);
      break;
   case RELU:
     // this->ApplyRelu(output);
      break;
   case TANH:
     // this->ApplyTanh(output);
      break;
   case LINEAR:
      break;
   }
}

void DnnLayer::ApplySigmoid(gsl::matrix *mat) {
   size_t i,j;
   for (i=0;i<mat->size1();i++)
      for(j=0;j<mat->size2();j++)
         (*mat)(i,j) = 1.0/(1.0+exp(-1.0*(*mat)(i,j)));
}


DnnParams::DnnParams() {
   lpc_order_vt = 0;
   lpc_order_glot = 0;
   f0_order= 0;
   gain_order= 0;
   hnr_order= 0;
   warping_lambda_vt = 0.0;
   fs= 0;
}


size_t DnnParams::getInputDimension() {
     size_t input_dim = 0;
     input_dim += lpc_order_vt;
     input_dim += lpc_order_glot;
     input_dim += hnr_order;
     input_dim += f0_order;
     input_dim += gain_order;

     return input_dim;
}

Dnn::Dnn() {
   input_min_value = 0.1;
   input_max_value = 0.9;
}

const gsl::vector & Dnn::getOutput() {


   // input normalization
  // 0.1 + 0.8*(gsl_vector_get(inputdata,i+stack*NPAR) - min)/(max-min);
   input_matrix = input_min_value + (input_max_value-input_min_value)*ElementDivision(input_matrix - input_data_min, input_data_max - input_data_min);

   gsl::matrix &input_ref = input_matrix;
   for(DnnLayer layer : this->layer_list) {
      layer.ForwardPass(input_ref);
      input_ref = layer.getOutput();
   }

   size_t i;
   for(i=0;i<output_vector.size();i++)
      output_vector(i) = input_ref(i,0);

   // output scaling

   return output_vector;

}

void Dnn::setInput(const SynthesisData &data, const size_t &frame_index) {

   input_matrix.set_dimensions(this->input_params.getInputDimension(),1);

   int i,ind=0;
   //f0
   if (this->input_params.f0_order > 0)
      for (i=0;i<input_params.f0_order;i++)
         input_matrix(ind++,0) = data.fundf(frame_index);

   //gain
   if (this->input_params.gain_order > 0)
      for (i=0;i<input_params.gain_order;i++)
         input_matrix(ind++,0) = data.frame_energy(frame_index);

   //hnr
   if (this->input_params.hnr_order > 0)
      for (i=0;i<input_params.hnr_order;i++)
         input_matrix(ind++,0) = data.hnr_glot(i,frame_index);

   // lsf_glot
   if (this->input_params.lpc_order_glot > 0)
      for (i=0;i<input_params.lpc_order_glot;i++)
         input_matrix(ind++,0) = data.lsf_glot(i,frame_index);

   // lsf_vt
   if (this->input_params.lpc_order_vt > 0)
      for (i=0;i<input_params.lpc_order_vt;i++)
         input_matrix(ind++,0) = data.lsf_vocal_tract(i,frame_index);

}

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
   std::cout << "Using excitation DNN specified in file " << fname_str  << std::endl;

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
   //for (libconfig::Setting const &layer : layers)
   //   this->layer_sizes.push_back(layer);
   for (size_t i=0;i<num_layers; i++) {
      this->layer_sizes.push_back(layers[i]);
   }



   const libconfig::Setting &activations = cfg.lookup("ACTIVATIONS");
   std::vector<std::string> strs;
   //for (libconfig::Setting const &af : activations) {
   for (size_t i=0;i<num_layers; i++) {
      //strs.push_back(af);
      strs.push_back(activations[i]);
      this->activation_functions.push_back(ActivationParse(strs.back()));
   }

   ConfigLookupInt("LPC_ORDER_VT", cfg, false, &(input_params.lpc_order_vt)) ;
   ConfigLookupInt("LPC_ORDER_GLOT", cfg, false, &(input_params.lpc_order_glot));
   ConfigLookupInt("HNR_ORDER", cfg, false, &(input_params.hnr_order));
   ConfigLookupInt("F0_ORDER", cfg, false, &(input_params.f0_order));
   ConfigLookupInt("GAIN_ORDER", cfg, false, &(input_params.gain_order));
   ConfigLookupInt("SAMPLING_FREQUENCY", cfg, false, &(input_params.fs));
   ConfigLookupDouble("WARPING_LAMBDA_VT", cfg, false, &(input_params.warping_lambda_vt));

   if (input_params.getInputDimension() != (size_t)layer_sizes[0]) {
      std::cerr << "Error: Dnn parameter dimensions (" << input_params.getInputDimension() << ") do not sum up to input dimension (" << layer_sizes[0] <<")"  <<  std::endl;
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

int Dnn::ReadData(const char *basename) {

   /* Filename processing */
   std::string fname_str;
   fname_str += basename;
   fname_str += ".dnnData";

   std::ifstream file(fname_str, std::ios::in | std::ios::binary);
   if (!file)
      return EXIT_FAILURE;

   std::streampos file_size;
   size_t n_values;

   // File size
   file.seekg(0, std::ios::end);
   file_size = file.tellg();
   file.seekg(0, std::ios::beg);

   // Check file length
   size_t expected_length = 0;
   size_t in_size, out_size=0;
   for (size_t layer_index=0; layer_index<this->num_layers-1 ; layer_index++) {
      in_size = this->layer_sizes[layer_index];
      out_size = this->layer_sizes[layer_index+1];
      expected_length += in_size*out_size + out_size;
   }
   //expected_length += 2*(this->layer_sizes[0]);

   n_values = file_size / sizeof(double);
   assert(n_values == expected_length);
   double *file_data = new double[n_values];
   file.read(reinterpret_cast<char*>(file_data), file_size);

   gsl::matrix W;
   gsl::matrix b;

   size_t i,j, ind=0;
   for (size_t layer_index=0; layer_index<this->num_layers-1 ; layer_index++) {
      in_size = this->layer_sizes[layer_index];
      out_size = this->layer_sizes[layer_index+1];
      W = gsl::matrix(out_size, in_size);
      b = gsl::matrix(out_size, 1);

//      for(i=0;i<out_size;i++)
//         for(j=0;j<in_size;j++)
//            W(i,j) = file_data[ind++];


      for(j=0;j<in_size;j++)
         for(i=0;i<out_size;i++)
            W(i,j) = file_data[ind++];

      for(i=0;i<out_size;i++)
         b(i,0) = file_data[ind++];

      this->addLayer(DnnLayer(W,b, this->activation_functions[layer_index]));

   }

   delete[] file_data;

    /* Read input max and min*/

   /* Filename processing */
     fname_str = basename;
     fname_str += ".dnnMinMax";

     std::ifstream file2(fname_str, std::ios::in | std::ios::binary);
     if (!file2)
        return EXIT_FAILURE;

     // File size
     file2.seekg(0, std::ios::end);
     file_size = file2.tellg();
     file2.seekg(0, std::ios::beg);

     // Check file length
     expected_length = 2*(this->layer_sizes[0]);

     ind=0;
     n_values = file_size / sizeof(double);
     assert(n_values == expected_length);
     file_data = new double[n_values];
     file2.read(reinterpret_cast<char*>(file_data), file_size);

   // Read input data min values
   this->input_data_min = gsl::matrix(layer_sizes[0],1);
   for (i=0;i<(size_t)this->layer_sizes[0];i++)
      this->input_data_min(i,0) = file_data[ind++];

   // Read input data max values
   this->input_data_max = gsl::matrix(layer_sizes[0],1);
   for (i=0;i<(size_t)this->layer_sizes[0];i++)
      this->input_data_max(i,0) = file_data[ind++];

   delete[] file_data;

   // Allocate input and output
   input_matrix = gsl::matrix(layer_sizes[0],1, true);
   output_vector = gsl::vector(out_size,true);


   return EXIT_SUCCESS;


}
