/*
 * DnnClass.h
 *
 *  Created on: 18 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_DNNCLASS_H_
#define SRC_GLOTT_DNNCLASS_H_

#include <list>

enum DnnActivationFunction {SIGMOID, TANH, RELU, LINEAR};

class DnnLayer {
public:
   DnnLayer();
   DnnLayer(const gsl::matrix &W, const gsl::vector &b, const DnnActivationFunction &af);
   ~DnnLayer();

private:
   gsl::matrix W;
   gsl::vector b;
   DnnActivationFunction activation_function;
};

class Dnn {
public:
   Dnn() {};
   ~Dnn() {};
   void addLayer(const DnnLayer &layer);
   int ReadInfo(const char *basename);
   int ReadData(const char *basename);


private:
   std::list<DnnLayer> layer_list;
   size_t num_layers;
   std::vector<int> layer_sizes;
   std::vector<DnnActivationFunction> activation_functions;

   DnnActivationFunction ActivationParse(std::string &str);

};

#endif /* SRC_GLOTT_DNNCLASS_H_ */
