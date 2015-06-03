#include "glmData.h"

#include <cuda_runtime.h>

glmData::glmData(int nRows, num_t *_y, num_t *_xNumeric, int nColsNumeric,
			num_t *_weights) {
	// Handle y first, which is fairly straightforward
	y = new glmVector<num_t>(_y, nRows);
	y->copyHostToDevice();

	// xNumeric is fairly straightforward as well
	xNumeric = new glmMatrix<num_t>(_xNumeric, nRows, nColsNumeric);
	xNumeric->copyHostToDevice();

	// Handle the special case of weights
	if (_weights == NULL) {
		weights = NULL;
	} else {
		weights = new glmVector<num_t>(_weights, nRows);
		weights->copyHostToDevice();
	}

	return;
}

glmData::~glmData() {
	delete y;
	delete xNumeric;
	if (weights != NULL) {
		delete weights;
	}
}

int glmData::getNObs(void) {
	return y->getLength();
}

int glmData::getNBeta(void) {
	int numXNumeric = xNumeric->getNCols();
	return numXNumeric + 1;	// +1 is for the intercept
}
