#include <gslwrap/vector_double.h>
#include <gslwrap/vector_int.h>
#include <cmath>
#include <float.h>

#include "definitions.h"
#include "SpFunctions.h"
#include "FileIo.h"
#include "InverseFiltering.h"
#include "AnalysisFunctions.h"
#include "DebugUtils.h"
#include "Filters.h"
#include "QmfFunctions.h"

int PolarityDetection(const Param &params, gsl::vector *signal, gsl::vector *source_signal_iaif) {
   switch(params.signal_polarity) {
   case POLARITY_DEFAULT :
      return EXIT_SUCCESS;

   case POLARITY_INVERT :
      std::cout << " -- Inverting polarity (SIGNAL_POLARITY = 'INVERT')" << std::endl;
      (*signal) *= (double)-1.0;
      return EXIT_SUCCESS;

   case POLARITY_DETECT :
      std::cout << "Using automatic polarity detection ...";
      if(!source_signal_iaif->is_set()) {
          *source_signal_iaif = gsl::vector(signal->size());
         GetIaifResidual(params, *signal, source_signal_iaif);
      }

      if(Skewness(*source_signal_iaif) > 0) {
         std::cout << "... Detected negative polarity. Inverting signal." << std::endl;
         (*signal) *= (double)-1.0;
         (*source_signal_iaif) *= (double)-1.0;
      } else {
         std::cout << "... Detected positive polarity." << std::endl;
      }

      return EXIT_SUCCESS;
   }

   return EXIT_FAILURE;

}



/**
 * Get the F0 vector if the analyzed signal.
 * input: params, signal
 * output: fundf: Obtained F0 vector.
 *
 */
int GetF0(const Param &params, const gsl::vector &signal, gsl::vector *fundf, gsl::vector *source_signal_iaif) {

	std::cout << "F0 analysis ";
	if(params.use_external_f0) {
		std::cout << "using external F0 file: " << params.external_f0_filename << " ...";
		gsl::vector fundf_ext;
		ReadGslVector(params.external_f0_filename, params.data_type, &fundf_ext);
		if(fundf_ext.size() != (size_t)params.number_of_frames) {
			std::cout << "Warning: External F0 file length differs from number of frames. Interpolating external "
						  "F0 length to match number of frames. F0 length: " \
					<< fundf_ext.size() << ", Number of frames: " << params.number_of_frames << std::endl;
			InterpolateNearest(fundf_ext,params.number_of_frames,fundf);
		} else {
			fundf->copy(fundf_ext);
		}
	}
	std::cout << " done." << std::endl;
	return EXIT_SUCCESS;
}

/**
 * Get the glottal closure instants (GCIs) of the analyzed signal.
 * input: params, signal
 * output: gci_signal: Sparse signal-length representation of gcis as ones and otherwise zeros
 *
 */
int GetGci(const Param &params, const gsl::vector &signal, const gsl::vector &fundf, gsl::vector_int *gci_inds, gsl::vector *source_signal_iaif) {
	if(params.use_external_gci) {
		std::cout << "Reading GCI information from external file: " << params.external_gci_filename << " ...";
		gsl::vector gcis;
		if(ReadGslVector(params.external_gci_filename, params.data_type, &gcis) == EXIT_FAILURE)
         return EXIT_FAILURE;

		*gci_inds = gsl::vector_int(gcis.size());
		size_t i;
		for (i=0; i<gci_inds->size();i++) {
			(*gci_inds)(i) = (int)round(gcis(i) * params.fs);
		}
	} else {
      if(!source_signal_iaif->is_set()) {
         *source_signal_iaif = gsl::vector(signal.size());
         GetIaifResidual(params, signal, source_signal_iaif);
      }
      std::cout << "GCI estimation using the SEDREAMS algorithm ...";
      gsl::vector mean_based_signal(signal.size(),true);
      MeanBasedSignal(signal, params.fs, getMeanF0(fundf),&mean_based_signal);
      SedreamsGciDetection(*source_signal_iaif,mean_based_signal,gci_inds);
	}

   std::cout << " done." << std::endl;
	return EXIT_SUCCESS;
}

int GetGain(const Param &params, const gsl::vector &signal, gsl::vector *gain_ptr) {

	gsl::vector frame = gsl::vector(params.frame_length);
	gsl::vector gain = gsl::vector(params.number_of_frames);
	int frame_index;
	double frame_energy;
	for(frame_index=0;frame_index<params.number_of_frames;frame_index++) {
		GetFrame(params, signal, frame_index, &frame, NULL);

		/* Evaluate gain of frame, normalize energy per sample basis */
	/*	double sum = 0.0;
		double mean = getMean(frame);
		size_t i;
		for(i=0;i<frame.size();i++) {
			sum =+ (frame(i)-mean)*(frame(i)-mean);
		}
		if(sum == 0.0)
			sum =+ DBL_MIN;

		gain(frame_index) = 10.0*log10(sum/((double)(frame.size() * frame.size()))); // energy per sample (not power)*/
		frame_energy = getEnergy(frame);
		if(frame_energy == 0.0)
			frame_energy =+ DBL_MIN;

		gain(frame_index) = (double)20.0*log10(frame_energy/((double)frame.size()));
	}
	*gain_ptr = gain;
	return EXIT_SUCCESS;
}

int GetFrame(const Param &params, const gsl::vector &signal, const int frame_index,gsl::vector *frame, gsl::vector *pre_frame) {
	int i, ind;
	/* Get samples to frame */
	if (frame != NULL) {
		for(i=0; i<(int)frame->size(); i++) {
			ind = frame_index*params.frame_shift - ((int)frame->size())/2 + i; // SPTK compatible, ljuvela
			if (ind >= 0 && ind < (int)signal.size()){
				(*frame)(i) = signal(ind);
			}
		}
	} else {
		return EXIT_FAILURE;
	}

	/* Get pre-frame samples for smooth filtering */
	if (pre_frame){
		for(i=0; i<params.lpc_order_vt; i++) {
			ind = frame_index*params.frame_shift - (int)frame->size()/2+ i - params.lpc_order_vt; // SPTK compatible, ljuvela
			if(ind >= 0 && ind < (int)signal.size())
				(*pre_frame)(i) = signal(ind);

  		}
	}

	return EXIT_SUCCESS;
}


 /**
  *
  *
  */
int SpectralAnalysis(const Param &params, const AnalysisData &data, gsl::matrix *poly_vocal_tract) {

	gsl::vector frame(params.frame_length);
	gsl::vector pre_frame(params.lpc_order_vt,true);
	gsl::vector lp_weight(params.frame_length + params.lpc_order_vt,true);
	gsl::vector A(params.lpc_order_vt+1,true);
	gsl::vector G(params.lpc_order_glot_iaif,true);
	gsl::vector B(1);B(0) = 1.0;
	//gsl::vector lip_radiation(2);lip_radiation(0) = 1.0; lip_radiation(1) = 0.99;
   gsl::vector frame_pre_emph(params.frame_length);
   gsl::vector frame_full; // frame + preframe
   gsl::vector residual_full; // residual with preframe

	std::cout << "Spectral analysis ...";

   size_t frame_index;
	for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
			GetFrame(params, data.signal, frame_index, &frame, &pre_frame);
			/** Voiced analysis **/
			if(data.fundf(frame_index) != 0) {

			   /* Estimate Weighted Linear Prediction weight */
				GetLpWeight(params,params.lp_weighting_function,data.gci_inds, frame, frame_index, &lp_weight);

				/* Pre-emphasis and windowing */
            Filter(std::vector<double>{1.0, -params.gif_pre_emphasis_coefficient},B, frame, &frame_pre_emph);
            ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);

            /* First-loop envelope */
				ArAnalysis(params.lpc_order_vt,params.warping_lambda_vt, params.lp_weighting_function, lp_weight, frame_pre_emph, &A);

				/* Second-loop envelope (if IAIF is used) */
				if(params.use_iterative_gif) {
               ConcatenateFrames(pre_frame, frame, &frame_full);
               Filter(A,B,frame_full,&residual_full);

               ApplyWindowingFunction(params.default_windowing_function, &residual_full);
               ArAnalysis(params.lpc_order_glot_iaif,0.0, NONE, lp_weight, residual_full.subvector(params.lpc_order_vt,params.frame_length), &G);

               Filter(G,B,frame,&frame_pre_emph); // Iterated pre-emphasis
               ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);
               ArAnalysis(params.lpc_order_vt,params.warping_lambda_vt, params.lp_weighting_function, lp_weight, frame_pre_emph, &A);
				}

         /** Unvoiced analysis **/
			} else {
				ApplyWindowingFunction(params.default_windowing_function,&frame);
				LPC(frame, params.lpc_order_vt, &A);
			}
			poly_vocal_tract->set_col_vec(frame_index,A);
	}


	std::cout << " done." << std::endl;
	return EXIT_SUCCESS;
}

int SpectralAnalysisQmf(const Param &params, const AnalysisData &data, gsl::matrix *poly_vocal_tract) {
   gsl::vector frame(params.frame_length);
   gsl::vector frame_pre_emph(params.frame_length);
   gsl::vector pre_frame(params.lpc_order_vt,true);
   gsl::vector frame_qmf1(frame.size()/2); // Downsampled low-band frame
   gsl::vector frame_qmf2(frame.size()/2); // Downsampled high-band frame
   gsl::vector lp_weight_downsampled(frame_qmf1.size() + params.lpc_order_vt_qmf1);
   gsl::vector B(1);B(0) = 1.0;

   gsl::vector H0 = Qmf::LoadFilter(kCUTOFF05PI);
   gsl::vector H1 = Qmf::GetMatchingFilter(H0);

	gsl::vector lp_weight(params.frame_length + params.lpc_order_vt,true);
	gsl::vector A(params.lpc_order_vt+1,true);
	gsl::vector A_qmf1(params.lpc_order_vt_qmf1+1,true);
	gsl::vector A_qmf2(params.lpc_order_vt_qmf2+1,true);
	//gsl::vector lsf_qmf1(params.lpc_order_vt_qmf1,true);
	//gsl::vector lsf_qmf2(params.lpc_order_vt_qmf2,true);
   //gsl::vector gain_qmf(params.number_of_frames);
   double gain_qmf, e1, e2;

   gsl::vector lip_radiation(2);lip_radiation(0) = 1.0;
   lip_radiation(1) = -params.gif_pre_emphasis_coefficient;

   //gsl::vector frame_full; // frame + preframe
   //gsl::vector residual_full; // residual with preframe

   std::cout << "QMF sub-band-based spectral analysis ...";

   size_t frame_index;
	for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
			GetFrame(params, data.signal, frame_index, &frame, &pre_frame);


			/** Voiced analysis (Low-band = QCP, High-band = LPC) **/
         if(data.fundf(frame_index) != 0) {
            /* Pre-emphasis */
            Filter(lip_radiation, B, frame, &frame_pre_emph);
            Qmf::GetSubBands(frame_pre_emph, H0, H1, &frame_qmf1, &frame_qmf2);
            /* Gain differences between frame_qmf1 and frame_qmf2: */

            e1 = getEnergy(frame_qmf1);
            e2 = getEnergy(frame_qmf2);
            if(e1 == 0.0)
               e1 += DBL_MIN;
            if(e2 == 0.0)
               e2 += DBL_MIN;
            gain_qmf = 20*log10(e1/e2);


            /** Low-band analysis **/
            GetLpWeight(params,params.lp_weighting_function,data.gci_inds, frame, frame_index, &lp_weight);
            Qmf::Decimate(lp_weight,2,&lp_weight_downsampled);

            ApplyWindowingFunction(params.default_windowing_function,&frame_qmf1);
            ArAnalysis(params.lpc_order_vt_qmf1,0.0,NONE, lp_weight_downsampled, frame_qmf1, &A_qmf1);

            /** High-band analysis **/
            ApplyWindowingFunction(params.default_windowing_function,&frame_qmf2);
            ArAnalysis(params.lpc_order_vt_qmf2,0.0,NONE, lp_weight_downsampled, frame_qmf2, &A_qmf2);

         /** Unvoiced analysis (Low-band = LPC, High-band = LPC, no pre-emphasis) **/
         } else {
            Qmf::GetSubBands(frame, H0, H1, &frame_qmf1, &frame_qmf2);
            e1 = getEnergy(frame_qmf1);
            e2 = getEnergy(frame_qmf2);
            if(e1 == 0.0)
               e1 += DBL_MIN;
            if(e2 == 0.0)
               e2 += DBL_MIN;
            gain_qmf = 20*log10(e1/e2);


            /** Low-band analysis **/
            ApplyWindowingFunction(params.default_windowing_function,&frame_qmf1);
            ArAnalysis(params.lpc_order_vt_qmf1,0.0,NONE, lp_weight_downsampled, frame_qmf2, &A_qmf1);

            /** High-band analysis **/
            ApplyWindowingFunction(params.default_windowing_function,&frame_qmf2);
            ArAnalysis(params.lpc_order_vt_qmf2,0.0,NONE, lp_weight_downsampled, frame_qmf2, &A_qmf2);
         }
         Qmf::CombinePoly(A_qmf1,A_qmf2,gain_qmf,(int)frame_qmf1.size(),&A);

         poly_vocal_tract->set_col_vec(frame_index,A);
         //Poly2Lsf(A_qmf1,&lsf_qmf1);
         //Poly2Lsf(A_qmf2,&lsf_qmf2);
         //lsf_qmf1->set_col_vec(frame_index,lsf_qmf1);
         //lsf_qmf2->set_col_vec(frame_index,lsf_qmf2);
	}

	return EXIT_SUCCESS;
}



int InverseFilter(const Param &params, const AnalysisData &data, gsl::matrix *poly_glott, gsl::vector *source_signal) {
   size_t frame_index;
	gsl::vector frame(params.frame_length,true);
	gsl::vector pre_frame(params.lpc_order_vt,true);
   gsl::vector frame_full(params.frame_length+params.lpc_order_vt); // Pre-frame + frame
   gsl::vector frame_residual(params.frame_length);
   gsl::vector a_glott(params.lpc_order_glot+1);
   gsl::vector b(1);b(0) = 1.0;

	for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
      GetFrame(params, data.signal, frame_index, &frame, &pre_frame);
      ConcatenateFrames(pre_frame, frame, &frame_full);
      if(params.warping_lambda_vt == 0.0) {
         Filter(data.poly_vocal_tract.get_col_vec(frame_index),b,frame_full,&frame_residual);
      } else {
         WFilter(data.poly_vocal_tract.get_col_vec(frame_index),b,frame_full,params.warping_lambda_vt,&frame_residual);
         if( (data.fundf(frame_index) != 0.0) && (frame_residual.max() > -1.0*frame_residual.min()) ) { // Warped residual seems to be switching the polarity on occasion
            frame_residual *= -1.0;
         }
      }




      double ola_gain = (double)params.frame_length/((double)params.frame_shift*2.0);
      frame_residual *= LogEnergy2FrameEnergy(data.frame_energy(frame_index),frame_residual.size())/getEnergy(frame_residual)/ola_gain;
      ApplyWindowingFunction(HANN, &frame_residual);

      LPC(frame_residual,params.lpc_order_glot,&a_glott);

      poly_glott->set_col_vec(frame_index, a_glott);

      OverlapAdd(frame_residual, frame_index*params.frame_shift, source_signal); // center index = frame_index*params.frame_shift

   }


   return EXIT_SUCCESS;
}

int Find_nearest_pulse_index(const int &sample_index, const gsl::vector &gci_inds, const Param &params, const double &f0){

   int i,j,k;
   int pulse_index = -1; // Return value initialization

   int dist, min_dist, ppos;
   min_dist = INT_MAX;
   /* Find the shortest distance between sample index and gcis */
   for(j=1;j<gci_inds.size()-1;j++){
      ppos = gci_inds(j);
      dist = abs(sample_index-ppos);
      if (dist > min_dist){
         break;
      }
      min_dist = dist;
      pulse_index=j;
   }

   /* Return the closest GCI if unvoiced */
   if (f0 == 0)
      return pulse_index;

   double pulselen, targetlen;
   targetlen = 2.0*params.fs/f0;
   pulselen = round(gci_inds(pulse_index+1)-gci_inds(pulse_index-1))+1;

   int new_pulse_index;
   int prev_index = pulse_index-1;
   int next_index = pulse_index+1;
   int prev_gci, next_gci;

   /* Choose next closest while pulse length deviates too much from f0 */
   while ((fabs(pulselen-targetlen)/targetlen) > params.max_pulse_len_diff){

      /* Prevent illegal reads*/
      if (prev_index < 0)
         prev_index = 0;
      if (next_index > gci_inds.size()-1)
         next_index = gci_inds.size()-1;

      prev_gci = gci_inds(prev_index);
      next_gci = gci_inds(next_index);

      /* choose closest below or above, increment for next iteration */
      if (abs(sample_index - next_gci) < abs(sample_index - prev_gci)) {
         new_pulse_index = next_index;
         next_index++;
      } else {
         new_pulse_index = prev_index;
         prev_index++;
      }

      /* break if out of range */
      if (new_pulse_index-1 < 0 || new_pulse_index+1 > gci_inds.size()-1) {
         break;
      } else {
         pulse_index = new_pulse_index;
      }

      /* calculate new pulse length */
      pulselen = round(gci_inds(pulse_index+1)-gci_inds(pulse_index-1))+1;
   }

   return pulse_index;

}

void GetPulses(const Param &params, const gsl::vector &source_signal, const gsl::vector_int &gci_inds, gsl::vector &fundf, gsl::matrix *pulses_mat) {

   if (params.extract_pulses_as_features == false)
      return;

   size_t frame_index;
   for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
      size_t sample_index = frame_index*params.frame_shift;
      size_t pulse_index = Find_nearest_pulse_index(sample_index, gci_inds, params, fundf(frame_index));

      size_t pulselen;
      pulselen = round(gci_inds(pulse_index+1)-gci_inds(pulse_index-1))+1;
      size_t j;

      gsl::vector paf_pulse(params.paf_pulse_length);
      if (params.use_pulse_interpolation == true){
         gsl::vector pulse_orig(pulselen);
         for(j=0;j<pulselen;j++) {
            pulse_orig(j) = source_signal(gci_inds(pulse_index-1)+j);
         }
         /* Interpolation on windowed signal prevents Gibbs at edges */
         ApplyWindowingFunction(COSINE, &paf_pulse);
         InterpolateSpline(pulse_orig, pulselen, &paf_pulse);
      } else{
         /* No windowing, just center at mid gci */
         for(j=0;j<paf_pulse.size();j++) {
            int i = gci_inds(pulse_index) - round(paf_pulse.size()/2) + j;
            if (i >= 0 && i < (int)source_signal.size())
               paf_pulse(j) = source_signal(i);
         }
      }
      pulses_mat->set_col_vec(frame_index, paf_pulse);

   }
}

void HighPassFiltering(const Param &params, gsl::vector *signal) {
      if(!params.use_highpass_filtering)
         return;

      std::cout << "High-pass filtering input signal with a cutoff frequency of 50Hz." << std::endl;

      gsl::vector signal_cpy(signal->size());
      signal_cpy.copy(*signal);

      if(params.fs < 40000) {
         Filter(k16HPCUTOFF50HZ,std::vector<double>{1},signal_cpy,signal);
         signal_cpy.copy(*signal);
         signal_cpy.reverse();
         Filter(k16HPCUTOFF50HZ,std::vector<double>{1},signal_cpy,signal);
         (*signal).reverse();
      } else {
         Filter(k44HPCUTOFF50HZ,std::vector<double>{1},signal_cpy,signal);
         signal_cpy.copy(*signal);
         signal_cpy.reverse();
         Filter(k16HPCUTOFF50HZ,std::vector<double>{1},signal_cpy,signal);
         (*signal).reverse();
      }

}


void GetIaifResidual(const Param &params, const gsl::vector &signal, gsl::vector *residual) {
   gsl::vector frame(params.frame_length,true);
   gsl::vector frame_residual(params.frame_length,true);
   gsl::vector frame_pre_emph(params.frame_length,true);
   gsl::vector pre_frame(params.lpc_order_vt,true);
   gsl::vector frame_full(params.lpc_order_vt+params.frame_length,true);
   gsl::vector A(params.lpc_order_vt+1,true);
   gsl::vector B(1);B(0) = 1.0;
   gsl::vector G(params.lpc_order_glot_iaif+1,true);
   gsl::vector weight_fn;



   size_t frame_index;
   for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
      GetFrame(params, signal, frame_index, &frame, &pre_frame);


      Filter(std::vector<double>{1.0, -params.gif_pre_emphasis_coefficient},B, frame, &frame_pre_emph);
      ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);

      ArAnalysis(params.lpc_order_vt,0.0, NONE, weight_fn, frame_pre_emph, &A);
      ConcatenateFrames(pre_frame, frame, &frame_full);

      Filter(A,B,frame_full,&frame_residual);

      ApplyWindowingFunction(params.default_windowing_function, &frame_residual);
      ArAnalysis(params.lpc_order_glot_iaif,0.0, NONE, weight_fn, frame_residual, &G);

      Filter(G,B,frame,&frame_pre_emph); // Iterated pre-emphasis
      ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);

      ArAnalysis(params.lpc_order_vt,0.0, NONE, weight_fn, frame_pre_emph, &A);

      Filter(A,B,frame_full,&frame_residual);

      double ola_gain = (double)params.frame_length/((double)params.frame_shift*2.0);
      frame_residual *= getEnergy(frame)/getEnergy(frame_residual)/ola_gain; // Set energy of residual euqual to energy of frame

      ApplyWindowingFunction(HANN, &frame_residual);

      OverlapAdd(frame_residual, frame_index*params.frame_shift, residual);
   }
}



