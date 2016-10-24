#include "FileIo.h"

#include <sndfile.hh>
#include <cstring>
#include <string>
#include <cstdio>
//#include <stdio.h>
#include <gslwrap/vector_double.h>
#include <gslwrap/vector_float.h>
#include <gslwrap/matrix_double.h>
#include "definitions.h"


void create_file (const char * fname, int format)
{	static short buffer [1024] ;

	SndfileHandle file ;
	int channels = 2 ;
	int srate = 48000 ;

	printf ("Creating file named '%s'\n", fname) ;

	file = SndfileHandle (fname, SFM_WRITE, format, channels, srate) ;

	memset (buffer, 0, sizeof (buffer)) ;

	file.write (buffer, 1024) ;

	puts ("") ;
	/*
	**	The SndfileHandle object will automatically close the file and
	**	release all allocated memory when the object goes out of scope.
	**	This is the Resource Acquisition Is Initailization idom.
	**	See : http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization
	*/
} /* create_file */

int WriteWavFile(const char *basename, const char *extension, const gsl::vector &signal, const int &fs) {

   /* Filename processing */
   std::string fname_str;
   fname_str += basename;
   fname_str += extension;
   std::cout << "Writing file " << fname_str  << std::endl;

   /* Filename processing */
   SndfileHandle file ;
   int channels = 1;
   file = SndfileHandle(fname_str.c_str(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, channels, fs) ;
   if (file.error()) {
      std::cerr << "Error: Failed to open file: " << fname_str.c_str() << std::endl;
      return EXIT_FAILURE;
   }


   double scale = GSL_MAX(signal.max(),-signal.min());
   if (scale > 1.0)
      std::cout << "Warning: Signal maximum value is: " << scale << ". Re-scaling signal." << std::endl;
   else
      scale = 1.0;

   /* Create buffer and write signal to file */
   double *buffer = new double[signal.size()];
   size_t i;
   for(i=0;i<signal.size();i++)
      buffer[i] = signal(i)/scale;
   file.write(buffer, signal.size());
   delete[] buffer;

   return EXIT_SUCCESS;
}

int ReadWavFile (const char *fname, gsl::vector *signal, Param *params) {

	SndfileHandle file ;

	file = SndfileHandle(fname) ;

	if(file.error()) {
      std::cerr << "Error: Failed to open file: " << fname << std::endl;
      return EXIT_FAILURE;
   }


	printf ("Reading file '%s'\n", fname) ;
	printf ("    Sample rate : %d\n", file.samplerate ()) ;
	printf ("    Channels    : %d\n", file.channels ()) ;
	if (file.samplerate() != params->fs) {
		std::cerr << "Error: Sample rate does not match with config" << std::endl;
		return EXIT_FAILURE;
	}

	/* define buffer for sndfile and read */
	*(signal) = gsl::vector(static_cast <size_t> (file.frames()));
	// double buffer[signal->size()];
	double *buffer = (double*)malloc(signal->size() * sizeof(double));
	file.read(buffer, signal->size()) ;

	/* copy read buffer to signal */
	size_t i;
	for (i=0;i<signal->size();i++)
		(*signal)(i) = buffer[i];

	/* set file parameters */
	params->number_of_frames = (int)ceil((double)signal->size()/(double)params->frame_shift);
	params->signal_length = signal->size();

	/* Get basename (without extension) for saving parameters later*/
   std::string str(fname);
   size_t lastindex = str.find_last_of(".");
   params->basename = new char[lastindex+1]();
   strncpy(params->basename, fname, lastindex);

   free(buffer);

   return EXIT_SUCCESS;
}


/**
 * Function EvalFileLength
 *
 * If file is in ASCII mode, read file and count the number of lines.
 * If file is in BINARY mode, read file size.
 *
 * @param name filename
 * @return number of parameters
 */
int EvalFileLength(const char *filename, DataType data_format) {

	FILE *file;
	char *s = new char[300];
	int fileSize = 0;

	/* Open file */
	file = fopen(filename, "r");
	if(!file) {
		std::cerr << "Error opening file " << filename << std::endl;
		return -1;
	}

	/* Read lines until EOF */
	if(data_format == ASCII) {
		while(fscanf(file,"%s",s) != EOF)
			fileSize++;
	} else if(data_format == BINARY) {
		fseek(file, 0, SEEK_END);
		fileSize = ftell(file)/sizeof(double);
	}
	fclose(file);
	delete[] s;

	return fileSize;
}

int ReadGslVector(const char *filename, const DataType format, gsl::vector *vector_ptr){
	/* Get file length */
	int size;
	size = EvalFileLength(filename, format);
	if (size < 0)
		return EXIT_FAILURE;

	/* Allocate vector */
	if (vector_ptr == NULL)
	   *(vector_ptr) = gsl::vector(size);
	else
	   vector_ptr->resize(size);

	FILE *inputfile = NULL;
	inputfile = fopen(filename, "r");
	if(inputfile==NULL){
		std::cerr << "Error opening file " << filename << std::endl;
		return EXIT_FAILURE;
	}
	if(format == ASCII) {
		vector_ptr->fscanf(inputfile);
	} else if(format == BINARY) {
		vector_ptr->fread(inputfile);
	}
	fclose(inputfile);

	return EXIT_SUCCESS;
}

int ReadGslVector(const char *basename, const char *extension, const DataType format, gsl::vector *vector_ptr) {
   std::string filename;
   filename += basename;
   filename += extension;
   return ReadGslVector(filename.c_str(), format, vector_ptr);
}

int ReadGslMatrix(const char *filename, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr) {
	/* Get file length */
	int size;
	size = EvalFileLength(filename, format);
	if (size < 0)
		return EXIT_FAILURE;

	if (size % n_rows != 0) {
		std::cerr << "ERROR: Invalid matrix dimensions in " << filename << std::endl;
		return EXIT_FAILURE;
	}
	size_t n_cols = size/n_rows;

	/* Read in transposed matrix */
	gsl::matrix matrix_temp(n_cols, n_rows);

	FILE *inputfile = NULL;
	inputfile = fopen(filename, "r");
	if(inputfile==NULL){
		std::cerr << "Error opening file " << filename << std::endl;
		return EXIT_FAILURE;
	}
	if(format == ASCII) {
		matrix_temp.fscanf(inputfile);
	} else if(format == BINARY) {
		matrix_temp.fread(inputfile);
	}
	fclose(inputfile);

	*matrix_ptr = matrix_temp.transpose();

	return EXIT_SUCCESS;
}

int ReadGslMatrix(const char *basename, const char *extension, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr) {
   std::string filename;
   filename += basename;
   filename += extension;
   return ReadGslMatrix(filename.c_str(), format, n_rows, matrix_ptr);
}

int WriteGslVector(const char *basename, const char *extension, const DataType &format, const gsl::vector &vector) {
   std::string filename;
   filename += basename;
   filename += extension;
   FILE *fid = NULL;
   fid = fopen(filename.c_str(), "w");
   if(fid==NULL){
      std::cerr << "Error: could not create file " << filename << std::endl;
      return EXIT_FAILURE;
   }
   switch (format) {
   case ASCII:
      vector.fprintf(fid, "%.7f");
      break;
   case BINARY:
      vector.fwrite(fid);
      break;
   }

   fclose(fid);
   return EXIT_SUCCESS;
}

int WriteGslVectorFloat(const char *basename, const char *extension, const DataType &format, const gsl::vector_float &vector) {
   std::string filename;
   filename += basename;
   filename += extension;
   FILE *fid = NULL;
   fid = fopen(filename.c_str(), "w");
   if(fid==NULL){
      std::cerr << "Error: could not create file " << filename << std::endl;
      return EXIT_FAILURE;
   }
   switch (format) {
   case ASCII:
      vector.fprintf(fid, "%.7f");
      break;
   case BINARY:
      vector.fwrite(fid);
      break;
   }

   fclose(fid);
   return EXIT_SUCCESS;
}

int WriteGslMatrix(const char *basename, const char *extension, const DataType &format, const gsl::matrix &mat) {

   std::string filename;
   filename += basename;
   filename += extension;
   FILE *fid = NULL;
   fid = fopen(filename.c_str(), "w");
   if(fid==NULL){
      std::cerr << "Error: could not create file " << filename << std::endl;
      return EXIT_FAILURE;
   }

   // TODO: make custom write functions to avoid transpose (and related allocation)
   gsl::matrix mat_trans = mat.transpose();
   size_t i,j;
   switch (format) {
   case ASCII:
      //mat_trans.fprintf(fid, "%.7f");
      for(j=0;j<mat.size2();j++)
         for(i=0;i<mat.size1();i++)
            fprintf(fid,"%.7f\n", mat(i,j));
      break;
   case BINARY:
      mat_trans.fwrite(fid);
      //for(i=0;i<mat.size1();i++)
      //for(j=0;j<mat.size2();j++)
      //fwrite( mat(i,j), sizeof(double), 1, fid);
      break;
   }

   fclose(fid);

   return EXIT_SUCCESS;
}

int ReadSynthesisData(const char *basename, Param *params, SynthesisData *data) {

   if (ReadGslVector(basename, ".F0", params->data_type, &(data->fundf)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ReadGslVector(basename, ".Gain", params->data_type, &(data->frame_energy)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ReadGslMatrix(basename, ".LSF", params->data_type, params->lpc_order_vt, &(data->lsf_vocal_tract)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (1) // TODO: add conditional
      if (ReadGslMatrix(basename, ".LSFsource", params->data_type, params->lpc_order_glot, &(data->lsf_glot)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (1) // TODO: add conditional
      if (ReadGslMatrix(basename, ".HNR", params->data_type, params->hnr_order, &(data->hnr_glot)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (1) // TODO: add conditional
      if (ReadGslMatrix(basename, ".PLS", params->data_type, params->paf_pulse_length, &(data->excitation_pulses)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   /* Read number of frames & compute signal length */
   params->number_of_frames = (int)(data->fundf.size());
   if(params->number_of_frames != (int)data->frame_energy.size() ||
      params->number_of_frames != (int)data->lsf_vocal_tract.get_cols() ||
      params->number_of_frames != (int)data->lsf_glot.get_cols() ||
      params->number_of_frames != (int)data->hnr_glot.get_cols() ) {

      std::cerr << "Error: Number of frames in input files do not match." << std::endl;
      return EXIT_FAILURE;
   }

   params->signal_length = rint(params->number_of_frames * params->frame_shift/params->speed_scale);
   data->signal = gsl::vector(params->signal_length,true);
   data->excitation_signal = gsl::vector(params->signal_length,true);

   return EXIT_SUCCESS;

}
