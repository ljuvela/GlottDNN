#include <gslwrap/vector_double.h>

#include "definitions.h"
#include "DebugUtils.h"

/*********************************************************************/
/*                       TEST FUNCTIONS                              */
/*********************************************************************/

// TEST FUNCTION
// PRINT VECTOR TO FILE: p1.dat
void VPrint1(const gsl::vector &vector) {
	FILE *fid = fopen("p1.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint2(const gsl::vector &vector) {
	FILE *fid = fopen("p2.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint3(const gsl::vector &vector) {
	FILE *fid = fopen("p3.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint4(const gsl::vector &vector) {
	FILE *fid = fopen("p4.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}

void VPrint5(const gsl::vector &vector) {
	FILE *fid = fopen("p5.dat", "w");
	vector.fprintf(fid,"%.30f");
	fclose(fid);
}
