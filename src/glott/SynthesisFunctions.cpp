/*
 * SynthesisFunctions.cpp
 *
 *  Created on: 13 Oct 2016
 *      Author: ljuvela
 */

#include <gslwrap/random_generator.h>
#include <gslwrap/random_number_distribution.h>

#include <gslwrap/vector_double.h>
#include "definitions.h"
#include "Utils.h"
#include "SpFunctions.h"
#include "InverseFiltering.h"
#include "Filters.h"
#include "SynthesisFunctions.h"



gsl::vector GetSinglePulse(const size_t &pulse_len, const double &energy, const gsl::vector &base_pulse) {

   /* Modify pulse */
   gsl::vector pulse(pulse_len);
   //InterpolateSpline(base_pulse, pulse_len, &pulse);
   Interpolate(base_pulse, &pulse);
   pulse *= energy/getEnergy(pulse);
   ApplyWindowingFunction(HANN, &pulse);

   /* Window length normalization */

   /*
   gsl::vector ones(pulse_len);
   ones.set_all(1.0);
   ApplyWindowingFunction(HANN, &ones);
   pulse *= pulse_len / pow(ones.norm2(),2);
   */

   return pulse;
}

void CreateExcitation(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal) {

   gsl::vector single_pulse_base;
   gsl::random_generator rand_gen;

   gsl::gaussian_random gauss_gen(rand_gen);


   switch (params.excitation_method) {
   case SINGLE_PULSE_EXCITATION:
      single_pulse_base = StdVector2GslVector(kDGLOTPULSE) ;
      break;
   case DNN_GENERATED_EXCITATION:
      // Load DNN
      break;
   case PULSES_AS_FEATURES_EXCITATION:
      // Load pulses as features
      break;
   }


   size_t sample_index = 0;
   size_t frame_index;
   gsl::vector pulse;
   gsl::vector noise(params.frame_shift*2);
   double T0, energy;
   size_t pulse_len;
   while (sample_index < (size_t)params.signal_length) {
      frame_index = rint(params.speed_scale * sample_index / (params.signal_length-1) * (params.number_of_frames-1));
      /** Voiced excitation **/
      if(data.fundf(frame_index) > 0) {

         T0 = params.fs/data.fundf(frame_index);
         pulse_len = rint(2*T0);
         energy = LogEnergy2FrameEnergy(data.frame_energy(frame_index),pulse_len);

         switch (params.excitation_method) {
         case SINGLE_PULSE_EXCITATION:
            pulse = GetSinglePulse(pulse_len, energy, single_pulse_base);
            break;
         case DNN_GENERATED_EXCITATION:
            //pulse = GetDnnPulse();
            std::cout << "DNN" << std::endl;
            break;
         case PULSES_AS_FEATURES_EXCITATION:
            //pulse = GetPafPulse();
            std::cout << "PAF" << std::endl;
            break;
         }


         OverlapAdd(pulse,sample_index,excitation_signal);

         sample_index += rint(T0);

      /** Unvoiced excitation **/
      } else {
         size_t i;
         for(i=0;i<noise.size();i++) {
            noise(i) = gauss_gen.get();
         }
         energy = LogEnergy2FrameEnergy(data.frame_energy(frame_index),noise.size());
         noise *= energy/getEnergy(noise);
         ApplyWindowingFunction(HANN,&noise);
         OverlapAdd(noise,sample_index,excitation_signal);

         sample_index += params.frame_shift;
      }
   }
}




void SpectralMatchExcitation(const Param &params,const SynthesisData &data, gsl::vector *excitation_signal) {
   /* Get analysis filters for generated excitation */
   size_t frame_index;
   gsl::vector frame(params.frame_length);
   gsl::vector a_gen(params.lpc_order_glot+1);
   gsl::vector a_tar(params.lpc_order_glot+1);
   gsl::vector w;
   gsl::matrix poly_glot_generated(params.lpc_order_glot+1, params.number_of_frames);
   for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
      GetFrame(*excitation_signal,frame_index,params.frame_shift,&frame,NULL);
      ApplyWindowingFunction(params.default_windowing_function,&frame);
      //LPC(frame,params.lpc_order_glot,&A);
      ArAnalysis(params.lpc_order_glot, 0.0, NONE, w, frame, &a_gen);
      poly_glot_generated.set_col_vec(frame_index,a_gen);
   }

   /* Spectral match excitation */
   gsl::vector excitation_orig(excitation_signal->size());
   excitation_orig.copy(*excitation_signal);

   int sample_index,i;
   int previous_frame_index = -1;
   double gain = 1.0, sum;
   for(sample_index=0;sample_index<(int)excitation_signal->size();sample_index++) {
      frame_index = rint(params.speed_scale * sample_index / (params.signal_length-1) * (params.number_of_frames-1));
      if((int)frame_index != previous_frame_index) { //TODO: interpolation of parameters between frames according to update_interval
         a_gen = data.poly_glot.get_col_vec(frame_index);
         a_tar = poly_glot_generated.get_col_vec(frame_index);
         gain = GetFilteringGain(a_gen, a_tar, *excitation_signal, sample_index, params.frame_length, 0.0); // Should this be from ecitation_signal or excitation_orig?
         //gain = a_tar.norm2()/a_gen.norm2(); // Experimental gain normalization term
         a_tar(0) = 0.0;
      }
      sum = 0.0;
      for(i=0;i<GSL_MIN(params.lpc_order_glot+1,sample_index);i++) {
         sum += excitation_orig(sample_index-i)*a_gen(i)*gain - (*excitation_signal)(sample_index-i)*a_tar(i);
      }
      (*excitation_signal)(sample_index) = sum;
      previous_frame_index = frame_index;
   }

}










