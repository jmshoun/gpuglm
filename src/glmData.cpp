#include "glmData.h"

#include <cuda_runtime.h>

glmData::glmData(glmVector<num_t> *_y, glmMatrix<num_t> *_xNumeric,
		glmMatrix<factor_t> *_xFactor, glmVector<int> *_factorOffsets,
		glmVector<num_t> *_weights) {
	y = _y;
	xNumeric = _xNumeric;
	xFactor = _xFactor;
	factorOffsets = _factorOffsets;
	weights = _weights;

	return;
}

glmData::~glmData() {
	delete y;

	if (xNumeric != NULL) 		{ delete xNumeric; }
	if (xFactor != NULL) 		{ delete xFactor; }
	if (factorOffsets != NULL)	{ delete factorOffsets; }
	if (weights != NULL) 		{ delete weights; }
}

int glmData::getNObs(void) const {
	return y->getLength();
}

int glmData::getNFactors(void) const {
	return factorOffsets->getLength();
}

factor_t* glmData::getFactorColumn(int index) const {
	return xFactor->getDeviceElement(0, index);
}

int glmData::getFactorOffset(int index) const {
	int* hostData = factorOffsets->getHostData();
	return hostData[index];
}
