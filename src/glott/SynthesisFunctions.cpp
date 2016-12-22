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
#include "FileIo.h"
#include "DnnClass.h"
#include "SynthesisFunctions.h"

void PostFilter(const double &postfilter_coefficient, const int &fs, gsl::matrix *lsf) {

   size_t POWER_SPECTRUM_FRAME_LEN = 4096;
	size_t frame_index,i;
	gsl::vector lsf_vec(lsf->get_rows());
   gsl::vector poly_vec(lsf->get_rows()+1);
   gsl::vector r(lsf->get_rows()+1);
   ComplexVector poly_fft(POWER_SPECTRUM_FRAME_LEN/2+1);
   gsl::vector fft_mag;
   gsl::vector_int peak_indices;
   gsl::vector peak_values;
   int POWER_SPECTRUM_WIN = 20;//rint(20*16000/fs); // Originally hard-coded as 20 samples, should be fs adaptive?

   std::cout << "Using LPC postfiltering with a coefficient of " << postfilter_coefficient << std::endl;

	/* Loop for every index of the LSF matrix */
	for(frame_index=0;frame_index<lsf->get_cols();frame_index++) {

		/* Convert LSF to LPC */
		Lsf2Poly(lsf->get_col_vec(frame_index),&poly_vec);

		/* Compute power spectrum */
		FFTRadix2(poly_vec,POWER_SPECTRUM_FRAME_LEN,&poly_fft);
      fft_mag = poly_fft.getAbs();
      for(i=0;i<fft_mag.size();i++)
         fft_mag(i) = 1.0/pow(fft_mag(i),2);

		/* Modification of the power spectrum */
		FindPeaks(fft_mag, 0.1, &peak_indices, &peak_values);
		SharpenPowerSpectrumPeaks(peak_indices, postfilter_coefficient, POWER_SPECTRUM_WIN, &fft_mag);

		/* Construct autocorrelation r */
		poly_fft.setReal(fft_mag);
		poly_fft.setAllImag(0.0);
      IFFTRadix2(poly_fft,&r);

      Levinson(r, &poly_vec);

		/* Convert LPC back to LSF */
      Poly2Lsf(poly_vec,&lsf_vec);
      lsf->set_col_vec(frame_index,lsf_vec);
	}
}


void ParameterSmoothing(const Param &params, SynthesisData *data) {

   if(params.lsf_vt_smooth_len > 2)
      MovingAverageFilter(params.lsf_vt_smooth_len, &(data->lsf_vocal_tract));

   if(params.lsf_glot_smooth_len > 2)
      MovingAverageFilter(params.lsf_glot_smooth_len, &(data->lsf_glot));

   if(params.gain_smooth_len > 2)
      MovingAverageFilter(params.gain_smooth_len, &(data->frame_energy));

   if(params.hnr_smooth_len > 2)
      MovingAverageFilter(params.hnr_smooth_len, &(data->hnr_glot));

}

gsl::vector GetSinglePulse(const size_t &pulse_len, const double &energy, const gsl::vector &base_pulse) {

   /* Modify pulse */
   gsl::vector pulse(pulse_len);
   //InterpolateSpline(base_pulse, pulse_len, &pulse);
   Interpolate(base_pulse, &pulse);
   pulse *= energy/getEnergy(pulse);

   /* Window length normalization */

   /*
   gsl::vector ones(pulse_len);
   ones.set_all(1.0);
   ApplyWindowingFunction(HANN, &ones);
   pulse *= pulse_len / pow(ones.norm2(),2);
   */

   return pulse;
}

gsl::vector GetExternalPulse(const size_t &pulse_len, const bool &use_interpolation, const double &energy, const size_t &frame_index, const gsl::matrix &external_pulses) {

   // Declare and initialize
   gsl::vector pulse(pulse_len,true);

   if (use_interpolation == true){
      // Interpolate pulse
      gsl::vector pulse_orig(external_pulses.size1());
      for(size_t j=0;j<external_pulses.size1();j++) {
         pulse_orig(j) = external_pulses(j , frame_index);
      }
      InterpolateSpline(pulse_orig, pulse_len, &pulse);

   } else{

      // Copy pulse starting from middle of external pulse
      int mid = round(external_pulses.size1()/2.0);
      int ind;
      for (size_t i=0; i<pulse_len;i++) {
         ind = mid-round(pulse_len/2.0)+i;
         if (ind >= 0 && ind < (int)external_pulses.size1())
            pulse(i) = external_pulses(ind , frame_index);
      }
   }

   // Scale with correct energy
    pulse *= energy/getEnergy(pulse);

   /* Window length normalization */
//   gsl::vector ones(pulse_len);
//   ones.set_all(1.0);
//   ApplyWindowingFunction(HANN, &ones);
//   pulse *= pulse_len / pow(ones.norm2(),2);

   return pulse;
}

gsl::vector GetDnnPulse(const size_t &pulse_len, const double &energy, const size_t &frame_index, const SynthesisData &data,  Dnn &excDnn) {

   gsl::vector pulse(pulse_len);
   excDnn.setInput(data, frame_index);
   const gsl::vector &dnn_pulse = excDnn.getOutput();
   //if(frame_index == 5)
    //  VPrint1(dnn_pulse);

   // Copy pulse starting from middle of external pulse
   int mid = round(dnn_pulse.size()/2.0);
   int ind;
   for (size_t i=0; i<pulse_len;i++) {
      ind = mid-round(pulse_len/2.0)+i;
      if (ind >= 0 && ind < (int)dnn_pulse.size())
         pulse(i) = dnn_pulse(ind);
   }

   // Window pulse
   pulse *= energy/getEnergy(pulse);
   // TODO: window type switch

   return pulse;

}

void CreateExcitation(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal) {

   gsl::vector single_pulse_base;
   gsl::random_generator rand_gen;

   gsl::gaussian_random gauss_gen(rand_gen);

   Dnn excDnn;

   // Load excitation pulses
   switch (params.excitation_method) {
   case SINGLE_PULSE_EXCITATION:
      single_pulse_base = StdVector2GslVector(kDGLOTPULSE) ;
      break;
   case DNN_GENERATED_EXCITATION:
      // Load DNN
      excDnn.ReadInfo(params.dnn_path_basename.c_str());
      excDnn.ReadData(params.dnn_path_basename.c_str());
      break;
   case PULSES_AS_FEATURES_EXCITATION:
      // Use the original pulses as features (no-op here)
      break;
   }
   /*  Experimental for accurate PSOLA */
   //size_t frame_index_nx;
   //size_t frame_index_pr = 0;
   //double t0_pr;
   //double t0_nx;

   size_t frame_index, sample_index = 0;
   gsl::vector pulse;
   gsl::vector p2;
   gsl::vector noise(params.frame_shift*2);
   //gsl::vector noise(params.frame_shift);
   double T0, energy;

   size_t pulse_len;
   while (sample_index < (size_t)params.signal_length) {
      frame_index = rint(params.speed_scale * sample_index / (params.signal_length-1) * (params.number_of_frames-1));

      /** Voiced excitation **/
      if(data.fundf(frame_index) > 0) {
         T0 = params.fs/data.fundf(frame_index);

         if(params.excitation_method == DNN_GENERATED_EXCITATION && T0 > params.paf_pulse_length)
            T0 = params.paf_pulse_length;

         /*  Experimental for accurate PSOLA */
         //frame_index_nx = rint(params.speed_scale * (sample_index+T0) / (params.signal_length-1) * (params.number_of_frames-1));;
         //if(data.fundf(frame_index_nx) > 0)
         //   t0_nx = params.fs/data.fundf(frame_index_nx);
         //else
         //   t0_nx = (double)params.frame_shift;
         //if(data.fundf(frame_index_pr) > 0)
         //   t0_pr = params.fs/data.fundf(frame_index_pr);
         //else
         //   t0_pr = (double)params.frame_shift;

         pulse_len = rint(2*T0);
         energy = LogEnergy2FrameEnergy(data.frame_energy(frame_index),pulse_len);

         switch (params.excitation_method) {
         case SINGLE_PULSE_EXCITATION:
            pulse = GetSinglePulse(pulse_len, energy, single_pulse_base);
            ApplyWindowingFunction(HANN, &pulse);
            //p2 = gsl::vector(pulse.size());
            //p2.set_all(1.0);
            //ApplyPsolaWindow(HANN, t0_pr, t0_nx, &p2);
            break;
         case DNN_GENERATED_EXCITATION:
            //pulse = GetDnnPulse();
            pulse = GetDnnPulse(pulse_len, energy, frame_index, data, excDnn);
            ApplyWindowingFunction(params.psola_windowing_function, &pulse);
            break;
         case PULSES_AS_FEATURES_EXCITATION:
            pulse = GetExternalPulse(pulse_len, params.use_pulse_interpolation, energy, frame_index, data.excitation_pulses);
            ApplyWindowingFunction(params.psola_windowing_function, &pulse);
            break;
         }

         OverlapAdd(pulse,sample_index,excitation_signal);
         //OverlapAdd(p2,sample_index,excitation_signal);
         sample_index += rint(T0);

      /** Unvoiced excitation **/
      } else {
         size_t i;
         for(i=0;i<noise.size();i++) {
            noise(i) = gauss_gen.get();
         };
         energy = LogEnergy2FrameEnergy(data.frame_energy(frame_index),noise.size());

         switch (params.excitation_method) {
         case SINGLE_PULSE_EXCITATION:
            //pulse = GetSinglePulse(noise.size(), energy, single_pulse_base);
            pulse = noise;
            pulse *= params.noise_gain_unvoiced*energy/getEnergy(noise);
            pulse /= 0.5*(double)noise.size()/(double)params.frame_shift; // Compensate OLA gain
            ApplyWindowingFunction(HANN, &pulse);

            break;
         case DNN_GENERATED_EXCITATION:
            //pulse = GetDnnPulse();
            //pulse = GetDnnPulse(noise.size(), energy, frame_index, data, excDnn);
            pulse = noise;
            pulse *= params.noise_gain_unvoiced*energy/getEnergy(noise);
            pulse /= 0.5*(double)noise.size()/(double)params.frame_shift; // Compensate OLA gain
            //pulse = GetExternalPulse(noise.size(), energy, frame_index, data.excitation_pulses);
            ApplyWindowingFunction(HANN, &pulse);
            break;
         case PULSES_AS_FEATURES_EXCITATION:
            //pulse = GetExternalPulse(noise.size(), params.use_pulse_interpolation, energy, frame_index, data.excitation_pulses);
            pulse = GetExternalPulse(noise.size(), false, energy, frame_index, data.excitation_pulses); // never interpolate noise
           // pulse = GetExternalPulse(noise.size(), energy, frame_index, data.excitation_pulses);
            ApplyWindowingFunction(params.psola_windowing_function, &pulse);
            break;
         }

//         noise *= params.noise_gain_unvoiced*energy/getEnergy(noise);
//         noise /= 0.5*(double)noise.size()/(double)params.frame_shift; // Compensate OLA gain
         //ApplyWindowingFunction(HANN,&noise);
         //OverlapAdd(noise,sample_index,excitation_signal);
         OverlapAdd(pulse,sample_index,excitation_signal);

         sample_index += params.frame_shift;
      }
      //frame_index_pr = frame_index;
   }
   CheckNanInf(*excitation_signal);
}


void HarmonicModification(const Param &params, const SynthesisData &data, gsl::vector *excitation_signal) {
   std::cout << "HNR modification ...";


   /* Variables */
   gsl::vector frame(params.frame_length_long);
   ComplexVector frame_fft;
   size_t NFFT = 8192; // Long FFT
   double MIN_LOG_POWER = -60.0;
   gsl::vector fft_mag(NFFT/2+1);
   gsl::vector fft_lower_env(NFFT/2+1);
   gsl::vector fft_upper_env(NFFT/2+1);
   gsl::vector fft_lower_env_target(NFFT/2+1);
   gsl::vector fft_noise(NFFT/2+1);
   gsl::vector hnr_interp(fft_mag.size());

   /* Noise generation */
   gsl::random_generator rand_gen;
   gsl::gaussian_random random_gauss_gen(rand_gen);
   ComplexVector noise_vec_fft;
   gsl::vector noise_vec(frame.size());

   /* Generate random Gaussian noise*/
   gsl::vector noise_long(excitation_signal->size());
   for(size_t j=0;j<noise_long.size();j++)
      noise_long(j) = random_gauss_gen.get();


   /* Copy excitation signal and re-initialize for overlap-add*/
   gsl::vector excitation_orig(*excitation_signal);
   excitation_signal->set_all(0.0);

   /* Define analysis and synthesis window */
   double kbd_alpha = 2.3;
   gsl::vector kbd_window = getKaiserBesselDerivedWindow(frame.size(), kbd_alpha);

   int frame_index,i;
   double val;
   for(frame_index=0;frame_index<params.number_of_frames;frame_index++) {

      GetFrame(excitation_orig, frame_index, rint(params.frame_shift/params.speed_scale), &frame, NULL);

      /* FFT with analysis window function */
      frame *= kbd_window;
      FFTRadix2(frame, NFFT, &frame_fft);
      fft_mag = frame_fft.getAbs();

      /* Get log power spectrum */
      for(i=0;i<(int)fft_mag.size();i++) {
         val = 20*log10(fft_mag(i));
         fft_mag(i) = GSL_MAX(val,MIN_LOG_POWER); // Min log-power = -60dB
      }

      /* Upper and lower envelope estimates for synthetic signal */
      if(data.fundf(frame_index) > 0) {
         UpperLowerEnvelope(fft_mag, data.fundf(frame_index), params.fs, &fft_upper_env, &fft_lower_env);
      } else {
         UpperLowerEnvelope(fft_mag, 100.0, params.fs, &fft_upper_env, &fft_lower_env);
      }

      /* Convert HNR from ERB to linear frequency scale */
      Erb2Linear(data.hnr_glot.get_col_vec(frame_index), params.fs, &hnr_interp);

      /* Calculate target noise floor level based on upper envelope and HNR */
      for(i=0;i<(int)fft_lower_env_target.size();i++)
         fft_lower_env_target(i) = fft_upper_env(i) + hnr_interp(i); // Ptar

      /* Calculate additive noise gain for each frequency bin */
      for(i=0;i<(int)fft_lower_env_target.size();i++) {
         fft_noise(i) = pow(10,fft_lower_env_target(i)/20.0) - pow(10,fft_lower_env(i)/20.0);
        // fft_noise(i) = pow(10,fft_lower_env_target(i)/10.0) - pow(10,fft_lower_env(i)/10.0); // power difference
      }


      /* Generate random Gaussian noise*/
      for(i=0;i<(int)noise_vec.size();i++)
         noise_vec(i) = random_gauss_gen.get();

      /* Noise FFT with analysis window */
      noise_vec *= kbd_window;
      FFTRadix2(noise_vec, NFFT, &noise_vec_fft);

      /* Normalize noise s.t. mean(abs(noise_fft)) == 1 */
      noise_vec_fft /= sqrt(noise_vec.size());

      /* Modify noise amplitude at each frequency bin */
      int noise_low_freq_limit_ind = rint(NFFT*params.noise_low_freq_limit_voiced/params.fs);
      for(i=0;i<(int)fft_mag.size();i++) {
         if(i < noise_low_freq_limit_ind) {
            /* Do not add noise below specified frequency */
            noise_vec_fft.setReal(i,0.0);
            noise_vec_fft.setImag(i,0.0);
         } else {
            if(fft_noise(i) > 0) {
               /* Add noise if noise floor is below that indicated by the HNR */
               //noise_vec_fft.setReal(i,noise_vec_fft.getReal(i)/noise_vec_fft.getAbs(i)*sqrt(fft_noise(i))*params.noise_gain_voiced);
               //noise_vec_fft.setImag(i,noise_vec_fft.getImag(i)/noise_vec_fft.getAbs(i)*sqrt(fft_noise(i))*params.noise_gain_voiced);

               /* Don't normalize the noise realization, Gaussian normal distributed noise has unit power at all frequencies (ljuvela) */
               noise_vec_fft.setReal(i,noise_vec_fft.getReal(i)*sqrt(fft_noise(i))*params.noise_gain_voiced);
               noise_vec_fft.setImag(i,noise_vec_fft.getImag(i)*sqrt(fft_noise(i))*params.noise_gain_voiced);
            } else {
               /* Do not add noise if noise floor is above that indicated by the HNR */
               noise_vec_fft.setReal(i,0.0);
               noise_vec_fft.setImag(i,0.0);
            }
         }
      }

      IFFTRadix2(noise_vec_fft,&noise_vec);

      /* Add noise and apply synthesis window */
      frame += noise_vec;
      frame *= kbd_window;
      /* Normalize overlap-add window */
      frame /= 0.5*(double)frame.size()/(double)params.frame_shift;
      OverlapAdd(frame,frame_index*rint(params.frame_shift/params.speed_scale),excitation_signal);

   }

   std::cout << " done." << std::endl;
}




void SpectralMatchExcitation(const Param &params,const SynthesisData &data, gsl::vector *excitation_signal) {
   /* Get analysis filters for synthetic excitation */
   size_t frame_index;
   gsl::vector frame(params.frame_length);
   gsl::vector a_gen(params.lpc_order_glot+1);
   gsl::vector a_tar(params.lpc_order_glot+1);
   gsl::vector lsf_gen(params.lpc_order_glot);
   gsl::vector lsf_tar_interpolated(params.lpc_order_glot);
   gsl::vector lsf_gen_interpolated(params.lpc_order_glot);
   gsl::vector w;
   gsl::matrix lsf_glot_syn(params.lpc_order_glot, params.number_of_frames);
   double gain_target_db;
   for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
      GetFrame(*excitation_signal,frame_index,rint(params.frame_shift/params.speed_scale),&frame,NULL);
      ApplyWindowingFunction(params.default_windowing_function,&frame);
      //LPC(frame,params.lpc_order_glot,&A);
      ArAnalysis(params.lpc_order_glot, 0.0, NONE, w, frame, &a_gen);
      Poly2Lsf(a_gen,&lsf_gen);
      lsf_glot_syn.set_col_vec(frame_index,lsf_gen);
   }
   if(params.use_trajectory_smoothing)
      MovingAverageFilter(params.lsf_glot_smooth_len, &(lsf_glot_syn));

   StabilizeLsf(&(lsf_glot_syn));


   /* Spectral match excitation */
   gsl::vector excitation_orig(excitation_signal->size());
   excitation_orig.copy(*excitation_signal);

   int sample_index,i;
   double gain = 1.0, sum, frame_index_double;
   //int UPDATE_INTERVAL = rint(params.fs*0.005); // Hard-coded 5ms update interval
   int UPDATE_INTERVAL = params.filter_update_interval_specmatch; // Hard-coded 5ms update interval

   for(sample_index=0;sample_index<(int)excitation_signal->size();sample_index++) {

      if(sample_index % UPDATE_INTERVAL == 0) { //TODO: interpolation of parameters between frames according to update_interval
         frame_index_double = params.speed_scale * (double)sample_index / (double)(params.signal_length-1) * (double)(params.number_of_frames-1);
         InterpolateLinear(lsf_glot_syn,frame_index_double,&lsf_gen_interpolated);
         InterpolateLinear(data.lsf_glot,frame_index_double,&lsf_tar_interpolated);
         Lsf2Poly(lsf_gen_interpolated,&a_gen);
         Lsf2Poly(lsf_tar_interpolated,&a_tar);
         //TODO: Add interpolation to frame_energy.
         gain_target_db = InterpolateLinear(data.frame_energy(floor(frame_index_double)),
                                          data.frame_energy(ceil(frame_index_double)),frame_index_double);


         gain = GetFilteringGain(a_gen, a_tar, excitation_orig, gain_target_db,
                                    sample_index, params.frame_length, 0.0); // Should this be from ecitation_signal or excitation_orig?
         if(data.fundf(rint(frame_index_double)) == 0.0)
            gain *= params.noise_gain_unvoiced;
         a_tar(0) = 0.0;
      }
      sum = 0.0;
      for(i=0;i<GSL_MIN(params.lpc_order_glot+1,sample_index);i++) {
         sum += excitation_orig(sample_index-i)*a_gen(i)*gain - (*excitation_signal)(sample_index-i)*a_tar(i);
      }
      (*excitation_signal)(sample_index) = sum;
   }

}


void FilterExcitation(const Param &params, const SynthesisData &data, gsl::vector *signal) {

   int sample_index,i;
   double gain = 1.0, gain_target_db, sum, frame_index_double;
   gsl::vector lsf_interp(params.lpc_order_vt);
   gsl::vector a_interp(params.lpc_order_vt+1);
   gsl::vector B(1);B(0)=1.0;
   /* Initialize warping-specific variables */
   gsl::vector sigma(params.lpc_order_vt+3,true);
   gsl::vector rmem(params.lpc_order_vt+3,true);
   int mlen = params.lpc_order_vt+2;
   int bdim = params.lpc_order_vt+1;
   double tmpr;

   int UPDATE_INTERVAL = params.filter_update_interval_vt;
   signal->copy(data.excitation_signal);
  // std::cout << *signal << std::endl;
   for(sample_index=0;sample_index<(int)signal->size();sample_index++) {

      if(sample_index % UPDATE_INTERVAL == 0) {
         frame_index_double = params.speed_scale * (double)sample_index / (double)(params.signal_length-1) * (double)(params.number_of_frames-1);
         InterpolateLinear(data.lsf_vocal_tract,frame_index_double,&lsf_interp);
         Lsf2Poly(lsf_interp,&a_interp);
         if(params.warping_lambda_vt != 0.0)
            WarpingAlphas2Sigmas(a_interp, params.warping_lambda_vt, &sigma);

         gain_target_db = InterpolateLinear(data.frame_energy(floor(frame_index_double)),
                        data.frame_energy(ceil(frame_index_double)),frame_index_double);

         gain = GetFilteringGain(B, a_interp, data.excitation_signal, gain_target_db,
                                 sample_index, params.frame_length, params.warping_lambda_vt);
         if(data.fundf(rint(frame_index_double)) == 0.0)
            gain *= params.noise_gain_unvoiced;

      }
      /** Normal filtering **/
      if(params.warping_lambda_vt == 0.0) {
         sum = data.excitation_signal(sample_index)*gain;
         for(i=1;i<GSL_MIN(params.lpc_order_vt+1,sample_index);i++) {
            sum -=  (*signal)(sample_index-i)*a_interp(i);
         }
         (*signal)(sample_index) = sum;
      /** Warped filtering **/
      } else {
         sum = data.excitation_signal(sample_index)*gain;
         /* Update feedbackward sum */
	    	for(i=0;i<bdim;i++) {
	    		sum -= sigma(i)*rmem(i);
	    	}
	    	sum /= sigma(bdim);

         /* Set signal */
         (*signal)(sample_index) = sum;

	    	/* Update inner states */
	    	for(i=0;i<mlen;i++) {
	    		tmpr = rmem(i) + params.warping_lambda_vt*(rmem(i+1) - sum);
	    		rmem(i) = sum;
	    		sum = tmpr;
	    	}
      }
   }
   /* Replace nan and inf values with zeros */
   CheckNanInf(*signal);
}







