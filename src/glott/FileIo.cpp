#include "FileIo.h"

#include <sndfile.hh>
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

void ReadFile (const char *fname, gsl::vector *signal, Param *params) {
	//static short buffer [BUFFER_LEN] ;

	SndfileHandle file ;
	file = SndfileHandle(fname) ;

	printf ("Opened file '%s'\n", fname) ;
	printf ("    Sample rate : %d\n", file.samplerate ()) ;
	printf ("    Channels    : %d\n", file.channels ()) ;

	if (file.samplerate() != params->fs) {
		std::cerr << "Sample rate does not match with config" << std::endl;
		return;
	}

	*(signal) = gsl::vector(static_cast <size_t> (file.frames()));
	double buffer[signal->size()];
	file.read (buffer, signal->size()) ;

	int i;
	for (i=0;i<signal->size();i++) {
		(*signal)(i) = buffer[i];
	}

	params->number_of_frames = ceil(signal->size()/params->frame_shift);
	params->signal_length = signal->size();

	/* RAII takes care of destroying SndfileHandle object. */
} /* read_file */


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
	char s[300];
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

	return fileSize;
}

int ReadGslVector(const char *filename, const DataType format, gsl::vector *vector_ptr){
	/* Get file length */
	int size;
	size = EvalFileLength(filename, format);
	if (size < 0)
		return EXIT_FAILURE;

	/* Allocate vector */
	*(vector_ptr) = gsl::vector(size);

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

int ReadGslMatrix(const char *filename, const DataType format, const size_t n_rows,  gsl::matrix *matrix_ptr){
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

	/* Allocate vector */
	*(matrix_ptr) = gsl::matrix(n_rows, n_cols);

	FILE *inputfile = NULL;
	inputfile = fopen(filename, "r");
	if(inputfile==NULL){
		std::cerr << "Error opening file " << filename << std::endl;
		return EXIT_FAILURE;
	}
	if(format == ASCII) {
		matrix_ptr->fscanf(inputfile);
	} else if(format == BINARY) {
		matrix_ptr->fread(inputfile);
	}
	fclose(inputfile);

	return EXIT_SUCCESS;
}

