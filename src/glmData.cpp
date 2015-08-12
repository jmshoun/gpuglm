#include "glmData.h"

#include <cuda_runtime.h>

glmData::glmData(glmVector<num_t> *_y, glmMatrix<num_t> *_xNumeric,
		glmMatrix<factor_t> *_xFactor,
		glmVector<num_t> *_weights) {
	y = _y;
	xNumeric = _xNumeric;
	xFactor = _xFactor;
	weights = _weights;

	return;
}

glmData::~glmData() {
	delete y;

	if (xNumeric != NULL) 	{ delete xNumeric; }
	if (xFactor != NULL) 	{ delete xFactor; }
	if (weights != NULL) 	{ delete weights; }
}

int glmData::getNObs(void) {
	return y->getLength();
}

int glmData::getNBeta(void) {
	int numXNumeric = xNumeric->getNCols();
	int numXFactor = xFactor->getNCols();
	return numXNumeric + 1;	// +1 is for the intercept
}
