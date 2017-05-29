/*
 * DnnClass.h
 *
 *  Created on: 18 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_DNNCLASS_H_
#define SRC_GLOTT_DNNCLASS_H_

#include <list>
#include <libconfig.h++>

enum DnnActivationFunction {SIGMOID, TANH, RELU, LINEAR};

class DnnLayer {
public:
   DnnLayer();
   DnnLayer(const gsl::matrix &W, const gsl::matrix &b, const DnnActivationFunction &af);
   ~DnnLayer();
   void ForwardPass(const gsl::matrix &input);
   const gsl::matrix & getOutput() const {return this->layer_output;};

private:
   // Variables
   gsl::matrix W;
   gsl::matrix b;
   gsl::matrix layer_output;
   DnnActivationFunction activation_function;
   // Functions
   void ApplySigmoid(gsl::matrix *mat);
   //void ApplyRelu(gsl::matrix *vec);
   //void ApplyTanh(gsl::matrix *vec);

};

struct DnnParams {
   DnnParams();
   ~DnnParams() {};
   int lpc_order_vt;
   int lpc_order_glot;
   int f0_order;
   int gain_order;
   int hnr_order;
   double warping_lambda_vt;
   int fs;

   size_t getInputDimension() ;

};

class Dnn {
public:
   Dnn();
   ~Dnn() {};
   void addLayer(const DnnLayer &layer);
   int ReadInfo(const char *basename);
   int ReadData(const char *basename);
   const gsl::vector & getOutput();
   void setInput(const SynthesisData &data, const size_t &frame_index);

private:
   std::list<DnnLayer> layer_list;
   size_t num_layers;
   std::vector<int> layer_sizes;
   std::vector<DnnActivationFunction> activation_functions;
   // scaling variables
   gsl::matrix input_data_min;
   gsl::matrix input_data_max;
   double input_min_value;
   double input_max_value;
   gsl::matrix output_mean;
   gsl::matrix output_std;
   gsl::matrix input_matrix;
   gsl::vector output_vector;
   DnnParams input_params;

   int checkInput(const std::string &type, const size_t &target_size, const size_t &actual_size);

   DnnActivationFunction ActivationParse(std::string &str);

};



#endif /* SRC_GLOTT_DNNCLASS_H_ */
