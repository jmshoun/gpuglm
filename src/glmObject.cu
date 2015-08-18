#include "glmObject.h"

// Constructors / Destructors /////////////////////////////////////////////////

glmObject::glmObject(glmData *_data, glmFamily *_family,
		glmControl *_control, glmVector<num_t> *_startingBeta) {
	data = _data;
	family = _family;
	control = _control;
	results = new glmResults(_startingBeta);

	nBeta = results->getNBeta();
	nObs = data->getNObs();

	gradient = new glmVector<num_t>(nBeta, true, true);
	betaDelta = new glmVector<num_t>(nBeta, true, true);
	yDelta = new glmVector<num_t>(nObs, true, true);
	predictions = new glmVector<num_t>(nObs, true, true);

	CUBLAS_WRAP(cublasCreate(&handle));

	return;
}

glmObject::~glmObject() {
	delete data;
	delete family;
	delete control;
	delete results;

	delete predictions;
	delete yDelta;
	delete gradient;

	CUBLAS_WRAP(cublasDestroy(handle));
}

// CUDA Kernels Used by Updating Functions ////////////////////////////////////

__global__ void factorProductKernel(int n, factor_t *factor, num_t *numeric,
		num_t *result) {
	int i;
	factor_t factorValue;

	for (i = 0; i < n; i++) {
		factorValue = factor[i];
		if (factorValue > 1) {
			result[factorValue - 2] = numeric[i] + result[factorValue - 2];
		}
	}

	return;
}

__global__ void factorPredictKernel(int n, factor_t *factor, num_t *betas,
		num_t *result) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	factor_t factorValue;

	if (i < n) {
		factorValue = factor[i];
		if (factorValue > 1) {
			result[i] += betas[factorValue];
		}
	}

	return;
}

// Gradient Functions /////////////////////////////////////////////////////////

void glmObject::updateGradient(void) {
	glmVector<num_t> *y = data->getY();
	glmVector<num_t> *weights = data->getWeights();

	this->updatePredictions();

	// Calculate yDelta as y - yHat
	vectorDifference(y, predictions, yDelta);

	// Add weights to yDelta
	if (weights != NULL) {
		vectorMultiply(yDelta, weights, yDelta);
	}

	// Calculate gradient from yDelta * weights...
	this->updateGradientXNumeric();
	this->updateGradientXFactor();
	this->updateGradientIntercept();

	return;
}

void glmObject::updateGradientIntercept(void) {
	vectorSum(yDelta, gradient, nBeta - 1);
	return;
}

void glmObject::updateGradientXNumeric(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();

	if (xNumeric != NULL) {
		xNumeric->columnProduct(handle, yDelta, gradient);
	}

	return;
}

void glmObject::updateGradientXFactor(void) {
	if (data->getXFactor() != NULL) {
		for (int i = 0; i < data->getNFactors(); i++) {
			this->updateGradientSingleFactor(i);
		}
	}

	return;
}

void glmObject::updateGradientSingleFactor(int index) {
	int gradientOffset = data->getFactorOffset(index) + 2;
	int factorLength = data->getFactorLength(index);
	factor_t *factorColumn = data->getFactorColumn(index);

	CUDA_WRAP(cudaMemset(gradient->getDeviceElement(gradientOffset), 0,
			factorLength * sizeof(num_t)));
	factorProductKernel<<<1, 1>>>(nObs, factorColumn, yDelta->getDeviceData(),
			gradient->getDeviceElement(gradientOffset));
	CUDA_WRAP(cudaPeekAtLastError());

	return;
}

// Prediction Functions ///////////////////////////////////////////////////////

void glmObject::updatePredictions(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *beta = results->getBeta();
	linkFunction invLink = family->getInvLink();

	if (xNumeric != NULL) {
		xNumeric->rowProduct(handle, beta, predictions);
	}

	if (data->getXFactor() != NULL) {
		for (int i = 0; i < data->getNFactors(); i++) {
			this->updatePredictionFactor(i);
		}
	}

	// Add in the intercept
	beta->copyDeviceToHost();
	vectorAddScalar(predictions, beta->getHostData()[nBeta - 1], predictions);
	(*invLink)(predictions, predictions, 0.0);

	return;
}

void glmObject::updatePredictionFactor(int index) {
	glmVector<num_t> *beta = results->getBeta();
	linkFunction invLink = family->getInvLink();

	int betaOffset = data->getFactorOffset(index);
	factor_t *factorColumn = data->getFactorColumn(index);
	int numBlocks = nObs / THREADS_PER_BLOCK +
			(nObs % THREADS_PER_BLOCK ? 1 : 0);

	factorPredictKernel<<<numBlocks, THREADS_PER_BLOCK>>>(nObs,
			factorColumn, beta->getDeviceData() + betaOffset,
			predictions->getDeviceData());
	CUDA_WRAP(cudaPeekAtLastError());

	return;
}
