#include "FileIo.h"

#include <sndfile.hh>
#include <cstring>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <libgen.h>
#include <gslwrap/vector_double.h>
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

int WriteWavFile(const std::string &fname_str, const gsl::vector &signal, const int &fs) {

   /* Filename processing */
//   std::string fname_str;
//   fname_str += basename;
//   fname_str += extension;
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

int ReadWavFile(const char *fname, gsl::vector *signal, Param *params) {

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
	/*
	std::string str(fname);
   size_t firstindex;
   firstindex = str.find_last_of("/");
   size_t lastindex = str.find_last_of(".");
   params->file_basename = str.substr(firstindex+1,lastindex-firstindex-1);
   params->file_path = str.substr(0,firstindex);
   */
	FilePathBasename(fname, &(params->file_path), &(params->file_basename));


   free(buffer);

   return EXIT_SUCCESS;
}


/**
 * Function version that doesn't modify params
 */
int ReadWavFile(const char *fname, gsl::vector *signal) {

   SndfileHandle file ;
   file = SndfileHandle(fname) ;
   if(file.error()) {
      std::cerr << "Error: Failed to open file: " << fname << std::endl;
      return EXIT_FAILURE;
   }
   printf ("Reading file '%s'\n", fname) ;
   printf ("    Sample rate : %d\n", file.samplerate ()) ;
   printf ("    Channels    : %d\n", file.channels ()) ;

   /* define buffer for sndfile and read */
   *(signal) = gsl::vector(static_cast <size_t> (file.frames()));
   // double buffer[signal->size()];
   double *buffer = (double*)malloc(signal->size() * sizeof(double));
   file.read(buffer, signal->size()) ;

   /* copy read buffer to signal */
   size_t i;
   for (i=0;i<signal->size();i++)
      (*signal)(i) = buffer[i];

   free(buffer);

   return EXIT_SUCCESS;
}


/**
 * Function EvalFileLength
 *
 * If file is in ASCII mode, read file and count the number of lines.
 * If file is in DOUBLE of FLOAT mode, read file size.
 *
 * @param name filename
 * @return number of parameters
 */
int EvalFileLength(const char *filename, DataType data_format) {
   // TODO: change filename to std::string

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
	switch (data_format) {
	   case ASCII:
	      while(fscanf(file,"%s",s) != EOF)
	         fileSize++;
	      break;
	   case DOUBLE:
	      fseek(file, 0, SEEK_END);
	      fileSize = ftell(file)/sizeof(double);
	      break;
	   case FLOAT:
         fseek(file, 0, SEEK_END);
         fileSize = ftell(file)/sizeof(float);
         break;
	}


	fclose(file);
	delete[] s;

	return fileSize;
}

int ReadGslVector(const std::string &filename, const DataType format, gsl::vector *vector_ptr){

   int size;
   FILE *inputfile = NULL;

   /* Get file length */
   size = EvalFileLength(filename.c_str(), format);
   if (size < 0)
      return EXIT_FAILURE;



   /* Allocate vector */
   if (vector_ptr == NULL)
      *(vector_ptr) = gsl::vector(size);
   else
      vector_ptr->resize(size);


   inputfile = fopen(filename.c_str(), "r");
   if(inputfile==NULL){
      std::cerr << "Error opening file " << filename << std::endl;
      return EXIT_FAILURE;
   }

   /* Read data */
   size_t i;
	switch (format) {
	case ASCII:
	   vector_ptr->fscanf(inputfile);
	   break;
	case DOUBLE:
	   vector_ptr->fread(inputfile);
	   break;
	case FLOAT:
      float fbuffer[1];
      for(i=0;i<(size_t)size;i++) {
         fread(fbuffer, sizeof(float), 1 , inputfile);
         (*vector_ptr)(i) = static_cast<double>(fbuffer[0]);
      }
      break;
	}

	fclose(inputfile);

	return EXIT_SUCCESS;
}


int ReadGslMatrix(const std::string &filename, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr) {
	/* Get file length */
	int size;
	size = EvalFileLength(filename.c_str(), format);
	if (size < 0)
		return EXIT_FAILURE;

	if (size % n_rows != 0) {
		std::cerr << "ERROR: Invalid matrix dimensions in " << filename << std::endl;
		return EXIT_FAILURE;
	}
	size_t n_cols = size/n_rows;

	FILE *inputfile = NULL;
	inputfile = fopen(filename.c_str(), "r");
	if(inputfile==NULL){
		std::cerr << "Error opening file " << filename << std::endl;
		return EXIT_FAILURE;
	}

	*matrix_ptr = gsl::matrix(n_rows,n_cols);
	size_t i,j;
	switch (format) {
	case ASCII:
	   float val;
      for(j=0;j<n_cols;j++)
         for(i=0;i<n_rows;i++) {
            fscanf(inputfile,"%f", &val);
            (*matrix_ptr)(i,j) = static_cast<double>(val);
         }
	   break;
	case DOUBLE:
      float dbuffer[1];
      for(j=0;j<n_cols;j++)
         for(i=0;i<n_rows;i++) {
            fread(dbuffer, sizeof(double), 1 , inputfile);
            (*matrix_ptr)(i,j) = static_cast<double>(dbuffer[0]);
         }
	   break;
	case FLOAT:
	   float fbuffer[1];
	   for(j=0;j<n_cols;j++)
	      for(i=0;i<n_rows;i++) {
	         fread(fbuffer, sizeof(float), 1 , inputfile);
	         (*matrix_ptr)(i,j) = static_cast<double>(fbuffer[0]);
	      }
	   break;
	}

	fclose(inputfile);

	return EXIT_SUCCESS;
}

int WriteGslVector(const std::string &filename, const DataType &format, const gsl::vector &vector) {

   size_t i;

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
   case DOUBLE:
      vector.fwrite(fid);
      break;
   case FLOAT:
      float fbuffer[1];
      for(i=0;i<vector.size();i++) {
         fbuffer[0] = static_cast<float>(vector(i));
         fwrite(fbuffer, sizeof(float), 1 , fid);
      }
      break;
   }

   fclose(fid);
   return EXIT_SUCCESS;
}


int WriteGslMatrix(const std::string &filename, const DataType &format, const gsl::matrix &mat) {

   FILE *fid = NULL;
   fid = fopen(filename.c_str(), "w");
   if(fid==NULL){
      std::cerr << "Error: could not create file " << filename << std::endl;
      return EXIT_FAILURE;
   }

   size_t i,j;
   switch (format) {
   case ASCII:
      for(j=0;j<mat.size2();j++)
         for(i=0;i<mat.size1();i++)
            fprintf(fid,"%.7f\n", mat(i,j));
      break;
   case DOUBLE:
      double dbuffer[1];
      for(j=0;j<mat.size2();j++)
         for(i=0;i<mat.size1();i++) {
            dbuffer[0] = static_cast<double>(mat(i,j));
            fwrite(dbuffer, sizeof(double), 1 , fid);
         }
      break;
   case FLOAT:
      float fbuffer[1];
      for(j=0;j<mat.size2();j++)
         for(i=0;i<mat.size1();i++) {
            fbuffer[0] = static_cast<float>(mat(i,j));
            fwrite(fbuffer, sizeof(float), 1 , fid);
         }
   }

   fclose(fid);

   return EXIT_SUCCESS;
}

int FilePathBasename(const char *filename, std::string *filepath, std::string *basename) {

   std::string str(filename);
   size_t firstindex;
   firstindex = str.find_last_of("/");
   size_t lastindex = str.find_last_of(".");
   //if (lastindex <= firstindex)
   if (lastindex == std::string::npos) // not found
      lastindex = str.size();

   *basename = str.substr(firstindex+1,lastindex-firstindex-1);
   *filepath = str.substr(0, firstindex);

   return EXIT_SUCCESS;
}

int ReadSynthesisData(const char *filename, Param *params, SynthesisData *data) {


   /* Get basename (without extension) for saving parameters later*/
   FilePathBasename(filename, &(params->file_path), &(params->file_basename));

   std::string param_fname;

   /* F0 (expected length for other features is taken from F0) */
   param_fname = GetParamPath("f0", params->extension_f0, params->dir_f0, *params);
   if (ReadGslVector(param_fname, params->data_type, &(data->fundf)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   params->number_of_frames = (int)(data->fundf.size());

   /* Pre-allocate and initialize everything for safety */
   data->frame_energy = gsl::vector(params->number_of_frames, true);
   data->lsf_vocal_tract = gsl::matrix(params->lpc_order_vt, params->number_of_frames, true);
   data->lsf_glot = gsl::matrix(params->lpc_order_glot, params->number_of_frames, true);
   data->hnr_glot = gsl::matrix( params->hnr_order, params->number_of_frames, true);
   data->excitation_pulses = gsl::matrix(params->paf_pulse_length, params->number_of_frames, true);
   // TODO: add generic spectrum

   /* Gain */
   param_fname = GetParamPath("gain", params->extension_gain, params->dir_gain, *params);
   if (ReadGslVector(param_fname, params->data_type, &(data->frame_energy)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if (params->number_of_frames != (int)data->frame_energy.size()) {
      std::cerr << "Error: Number of frames in input files do not match." << std::endl;
      std::cerr << "In file"  << param_fname << std::endl;
      return EXIT_FAILURE;
   }

   /* Vocal tract LSFs */
   if (! params->use_generic_envelope) {
      param_fname = GetParamPath("lsf", params->extension_lsf, params->dir_lsf, *params);
      if (ReadGslMatrix(param_fname, params->data_type, params->lpc_order_vt, &(data->lsf_vocal_tract)) == EXIT_FAILURE)
         return EXIT_FAILURE;
      if (params->number_of_frames != (int)data->lsf_vocal_tract.get_cols()) {
         std::cerr << "Error: Number of frames in input files do not match." << std::endl;
         std::cerr << "In file"  << param_fname << std::endl;
         return EXIT_FAILURE;
      }
   }

   /* Glottal source LSFs */
   if (params->use_spectral_matching || params->excitation_method == DNN_GENERATED_EXCITATION) {
      // TODO: more elaborate check for whether the features are actually used in internal DNN
      param_fname = GetParamPath("slsf", params->extension_lsfg, params->dir_lsfg, *params);
      if (ReadGslMatrix(param_fname, params->data_type, params->lpc_order_glot, &(data->lsf_glot)) == EXIT_FAILURE)
         return EXIT_FAILURE;
      if (params->number_of_frames != (int)data->lsf_glot.get_cols()) {
         std::cerr << "Error: Number of frames in input files do not match." << std::endl;
         std::cerr << "In file"  << param_fname << std::endl;
         return EXIT_FAILURE;
      }
   }

   /* Harmonic-to-Noise Ratio*/
   if (params->noise_gain_voiced > 0.0 || params->excitation_method == DNN_GENERATED_EXCITATION) {
      // TODO: more elaborate check for whether the features are actually used in internal DNN
      param_fname = GetParamPath("hnr", params->extension_hnr, params->dir_hnr, *params);
      if (ReadGslMatrix(param_fname, params->data_type, params->hnr_order, &(data->hnr_glot)) == EXIT_FAILURE)
         return EXIT_FAILURE;
      if (params->number_of_frames != (int)data->hnr_glot.get_cols()) {
         std::cerr << "Error: Number of frames in input files do not match." << std::endl;
         std::cerr << "In file"  << param_fname << std::endl;
         return EXIT_FAILURE;
      }
   }

   /* External excitation pulses as features */
   if (params->excitation_method == PULSES_AS_FEATURES_EXCITATION) {
      param_fname = GetParamPath("pls", params->extension_paf, params->dir_paf, *params);
      if (params->excitation_method == PULSES_AS_FEATURES_EXCITATION) {
         if (ReadGslMatrix(param_fname, params->data_type, params->paf_pulse_length, &(data->excitation_pulses)) == EXIT_FAILURE)
            return EXIT_FAILURE;
      }
      if (params->number_of_frames != (int)data->excitation_pulses.get_cols()) {
         std::cerr << "Error: Number of frames in input files do not match." << std::endl;
         std::cerr << "In file"  << param_fname << std::endl;
         return EXIT_FAILURE;
      }
   }

   /* Generic spectrum for filter envelope */
   if (params->use_generic_envelope) {
      param_fname = GetParamPath("sp", ".sp", params->dir_sp, *params);
      // TODO: add parameter for generic spectrum length
      if (ReadGslMatrix(param_fname, params->data_type, 2049, &(data->spectrum)) == EXIT_FAILURE)
         return EXIT_FAILURE;
      if (params->number_of_frames != (int)data->spectrum.get_cols()) {
         std::cerr << "Error: Number of frames in input files do not match." << std::endl;
         std::cerr << "In file"  << param_fname << std::endl;
         return EXIT_FAILURE;
      }
   }

   params->signal_length = rint(params->number_of_frames * params->frame_shift/params->speed_scale);
   data->signal = gsl::vector(params->signal_length,true);
   data->excitation_signal = gsl::vector(params->signal_length,true);

   return EXIT_SUCCESS;

}



/**
 * Read binary file in float format
 * inputs: fname_str is the filename , file_data is a float pointer where the data is read to
 *
 * This function will allocate the pointer with new, ownership is transferred to the caller
 * who must call delete[] for the pointer
 *
 * Number of values read is saved to the address pointed by n_read
 *
 * author: ljuvela
 *
 */
int ReadFileFloat(const std::string &fname_str, float **file_data, size_t *n_read) {

   std::ifstream file(fname_str, std::ios::in | std::ios::binary);
   if (!file)
      return EXIT_FAILURE;

   std::streampos file_size;
   size_t n_values;

   // File size
   file.seekg(0, std::ios::end);
   file_size = file.tellg();
   file.seekg(0, std::ios::beg);

   // Read file
   n_values = file_size / sizeof(float);
   *file_data = new float[n_values];
   file.read(reinterpret_cast<char*>(*file_data), file_size);

   *n_read = n_values;

   return EXIT_SUCCESS;

}

int WriteFileFloat(const std::string &fname_str, const float *data, const size_t &n_values) {

   // check for null
   if (data == NULL) {
      std::cerr << "Error: attempted to write from NULL pointer" << std::endl;
      return EXIT_FAILURE;
   }

   // open file
   std::ofstream file(fname_str, std::ios::out | std::ios::binary);
   if (!file) {
      std::cerr << "Error: could not open file " << fname_str << std::endl;
      return EXIT_FAILURE;
   }

   size_t bytes_written;
   size_t bytes_float = sizeof(float);
   float buffer[1];
   for (size_t i=0;i<n_values;i++) {
      buffer[0] = data[i];
      file.write(reinterpret_cast<char*>(buffer), bytes_float);
      //bytes_written = file.write(reinterpret_cast<char*>(buffer), bytes_float);
      //if (bytes_written != bytes_float)
      //    return EXIT_FAILURE;
   }
   file.close();

   return EXIT_SUCCESS;

}

/**
 * Get write/read path for a parameter as specified in configuration file
 */
std::string GetParamPath(const std::string &default_dir, const std::string &extension, const std::string &custom_dir, const Param &params)  {
   std::string param_path;

   std::string basedir(params.data_directory);
   if (basedir.back() != '/')
      basedir += "/";

   if (custom_dir.empty()) {
          if (params.save_to_datadir_root) {
             //param_path = basedir + params.file_basename + extension;
             // changed to GlottHMM default behavior
             param_path = params.file_path + "/" + params.file_basename + extension;
             std::cout << param_path << std::endl;
          } else {
             param_path = basedir + default_dir + "/" + params.file_basename + extension;
          }
       } else {
          param_path = custom_dir + "/" + params.file_basename + extension;
       }

   return param_path;
}

