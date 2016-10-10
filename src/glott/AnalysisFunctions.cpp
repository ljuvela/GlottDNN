#include <gslwrap/vector_double.h>
#include <gslwrap/vector_int.h>
#include <cmath>
#include <float.h>

#include "definitions.h"
#include "SpFunctions.h"
#include "FileIo.h"
#include "InverseFiltering.h"
#include "AnalysisFunctions.h"


/**
 * Get the F0 vector if the analyzed signal.
 * input: params, signal
 * output: fundf: Obtained F0 vector.
 *
 */
int GetF0(const Param &params, const gsl::vector &signal, gsl::vector *fundf) {

	std::cout << "F0 analysis ";
	if(params.use_external_f0) {
		std::cout << "using external F0 file: " << params.external_f0_filename << " ...";
		gsl::vector fundf_ext;
		ReadGslVector(params.external_f0_filename, params.data_type, &fundf_ext);
		if(fundf_ext.size() != (size_t)params.number_of_frames) {
			std::cout << "Warning: External F0 file length differs from number of frames. Interpolating external "
						  "F0 length to match numbder of frames. F0 length: " \
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
int GetGci(const Param &params, const gsl::vector &signal, gsl::vector_int *gci_inds) {

	if(params.use_external_gci) {
		std::cout << "Reading GCI information from external file: " << params.external_gci_filename << " ...";
		gsl::vector gcis;
		ReadGslVector(params.external_gci_filename, params.data_type, &gcis);
		*gci_inds = gsl::vector_int(gcis.size());
		size_t i;
		for (i=0; i<gci_inds->size();i++) {
			(*gci_inds)(i) = (int)round( gcis(i) * params.fs);
		}
	}
	std::cout << " done." << std::endl;
	return EXIT_SUCCESS;
}

int GetGain(const Param &params, const gsl::vector &signal, gsl::vector *gain_ptr) {

	gsl::vector frame = gsl::vector(params.frame_length);
	gsl::vector gain = gsl::vector(params.number_of_frames);
	int frame_index;
	for(frame_index=0;frame_index<params.number_of_frames;frame_index++) {
		GetFrame(params, signal, frame_index, &frame, NULL);

		/* Evaluate gain of frame, normalize energy per sample basis */
		double sum = 0.0;
		size_t i;
		for(i=0;i<frame.size();i++) {
			sum =+ frame(i)*frame(i);
		}
		if(sum == 0.0)
			sum =+ DBL_MIN;

		gain(frame_index) = 10.0*log10(sum/((double)(frame.size() * frame.size()))); // energy per sample (not power)
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
	if (pre_frame != NULL){
		for(i=0; i<params.lpc_order_vt; i++) {
			ind = frame_index*params.frame_shift - (int)frame->size()/2+ i - params.lpc_order_vt; // SPTK compatible, ljuvela
			if(ind >= 0 && ind < (int)signal.size())
				(*pre_frame)(i) = (int)signal(ind);
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
	gsl::vector pre_frame(params.lpc_order_vt);
	gsl::vector lp_weight(params.frame_length + params.lpc_order_vt,true);
	gsl::vector A(params.lpc_order_vt+1,true);
	gsl::vector G(params.lpc_order_glot_iaif,true);
	gsl::vector B(1);B(0) = 1.0;
   gsl::vector frame_pre_emph(params.frame_length);
   gsl::vector frame_full; // frame + preframe
   gsl::vector residual_full; // residual with preframe

	size_t frame_index;

	std::cout << "Spectral analysis ...";


	for(frame_index=0;frame_index<(size_t)params.number_of_frames;frame_index++) {
			GetFrame(params, data.signal, frame_index, &frame, &pre_frame);
			/** Voiced analysis **/
			if(data.fundf(frame_index) != 0) {

				GetLpWeight(params,params.lp_weighting_function,data.gci_inds, frame, frame_index, &lp_weight);

				// Pre-emphasis and windowing
            Filter(std::vector<double>{1.0, params.gif_pre_emphasis_coefficient},B, frame, &frame_pre_emph);
            ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);

            // First-loop envelope
				ArAnalysis(params.lpc_order_vt,params.warping_lambda_vt, params.lp_weighting_function, lp_weight, frame_pre_emph, &A);

				// Second-loop envelope (if IAIF is used)
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


int InverseFilter(const Param &params, const AnalysisData &data, gsl::matrix *poly_glott, gsl::vector *excitation_signal) {

   return EXIT_SUCCESS;
}



