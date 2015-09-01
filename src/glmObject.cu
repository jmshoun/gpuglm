#include "glmObject.h"

// Constructors / Destructors /////////////////////////////////////////////////

glmObject::glmObject(glmData *_data, glmFamily *_family,
		glmControl *_control, glmVector<num_t> *_startingBeta) {
	// Construct member objects
	data = _data;
	family = _family;
	control = _control;
	results = new glmResults(_startingBeta);

	// Set common dimensional parameters
	nBeta = results->getNBeta();
	nObs = data->getNObs();

	// Create common vectors need in the workspace
	gradient = new glmVector<num_t>(nBeta, true, true);
	betaDelta = new glmVector<num_t>(nBeta, true, true);
	yDelta = new glmVector<num_t>(nObs, true, true);
	xScratch = new glmVector<num_t>(nObs, true, true, true);
	linkPredictions = new glmVector<num_t>(nObs, true, true);
	responsePredictions = new glmVector<num_t>(nObs, true, true);

	// Create CUBLAS handle
	CUBLAS_WRAP(cublasCreate(&handle));

	return;
}

glmObject::~glmObject() {
	delete data;
	delete family;
	delete control;
	delete results;

	delete linkPredictions;
	delete responsePredictions;
	delete yDelta;
	delete xScratch;
	delete gradient;

	CUBLAS_WRAP(cublasDestroy(handle));
}

// CUDA Kernels Used by Updating Functions ////////////////////////////////////

__global__ void logLikelihoodKernel(int n, num_t *linkPredictions,
		num_t *responsePredictions, num_t *y, num_t *outputs) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		outputs[i] = y[i] * linkPredictions[i] - responsePredictions[i];
	}

	return;
}

__global__ void weightedLogLikelihoodKernel(int n, num_t *linkPredictions,
		num_t *responsePredictions, num_t *y, num_t *weights, num_t *outputs) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		outputs[i] = (y[i] * linkPredictions[i] - responsePredictions[i]) /
				weights[i];
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

// Log-Likelihood Functions ///////////////////////////////////////////////////

void glmObject::updateLogLikelihood(void) {
	glmVector<num_t> *partialLikelihood = xScratch;
	num_t logLikelihood = 0.0;
	num_t *hostLogLikelihood = &logLikelihood;
	num_t *deviceLogLikelihood;
	int numBlocks = nObs / THREADS_PER_BLOCK +
			(nObs % THREADS_PER_BLOCK ? 1 : 0);
	cumulantFunction cumulant = family->getCumulant();

	(*cumulant)(linkPredictions, responsePredictions, 0.0);

	if (data->getWeights() == NULL) {
		logLikelihoodKernel<<<numBlocks, THREADS_PER_BLOCK>>>(nObs,
				linkPredictions->getDeviceData(),
				responsePredictions->getDeviceData(),
				data->getY()->getDeviceData(),
				partialLikelihood->getDeviceData());
	} else {
		weightedLogLikelihoodKernel<<<numBlocks, THREADS_PER_BLOCK>>>(nObs,
				linkPredictions->getDeviceData(),
				responsePredictions->getDeviceData(),
				data->getY()->getDeviceData(),
				data->getWeights()->getDeviceData(),
				partialLikelihood->getDeviceData());
	}

	CUDA_WRAP(cudaMalloc((void **) &deviceLogLikelihood, sizeof(num_t)));
	vectorSum(partialLikelihood, deviceLogLikelihood);
	CUDA_WRAP(cudaMemcpy(hostLogLikelihood, deviceLogLikelihood, sizeof(num_t),
			cudaMemcpyDeviceToHost));
	CUDA_WRAP(cudaFree(deviceLogLikelihood));
	results->setLogLikelihood(logLikelihood);

	return;
}

// Gradient Functions /////////////////////////////////////////////////////////

void glmObject::updateGradient(void) {
	glmVector<num_t> *y = data->getY();
	glmVector<num_t> *weights = data->getWeights();

	this->updatePredictions();

	// Calculate yDelta as y - yHat
	vectorDifference(y, responsePredictions, yDelta);

	// Add weights to yDelta
	if (weights != NULL) {
		vectorMultiply(yDelta, weights, yDelta);
	}

	// Calculate gradient from yDelta * weights...
	this->updateGradientXNumeric();
	this->updateGradientXFactor();
	this->updateGradientIntercept();

//	gradient->copyDeviceToHost();
//	std::cout << *gradient << std::endl;

	return;
}

void glmObject::updateGradientIntercept(void) {
	num_t *interceptGradientElement = gradient->getDeviceElement(nBeta - 1);
	vectorSum(yDelta, interceptGradientElement);
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
	glmVector<factor_t> *factorColumn = data->getFactorColumn(index);

	factorProduct(factorColumn, factorLength, yDelta,
			gradient->getDeviceElement(gradientOffset));

	return;
}

// Prediction Functions ///////////////////////////////////////////////////////

void glmObject::updatePredictions(void) {
	linkFunction invLink = family->getInvLink();

	this->updateLinkPredictions();
	(*invLink)(linkPredictions, responsePredictions, 0.0);

	return;
}

void glmObject::updateLinkPredictions(void) {
	this->updateLinkPredictionXNumeric();
	this->updateLinkPredictionXFactor();
	this->updateLinkPredictionIntercept();

	return;
}

void glmObject::updateLinkPredictionIntercept(void) {
	glmVector<num_t> *beta = results->getBeta();
	beta->copyDeviceToHost();
	num_t intercept = beta->getHostData()[nBeta - 1];

	vectorAddScalar(linkPredictions, intercept, linkPredictions);

	return;
}

void glmObject::updateLinkPredictionXNumeric(void) {
	glmVector<num_t> *beta = results->getBeta();
	glmMatrix<num_t> *xNumeric = data->getXNumeric();

	if (xNumeric != NULL) {
		xNumeric->rowProduct(handle, beta, linkPredictions);
	}

	return;
}

void glmObject::updateLinkPredictionXFactor(void) {
	if (data->getXFactor() != NULL) {
		for (int i = 0; i < data->getNFactors(); i++) {
			this->updateLinkPredictionSingleFactor(i);
		}
	}

	return;
}

void glmObject::updateLinkPredictionSingleFactor(int index) {
	glmVector<num_t> *beta = results->getBeta();
	int betaOffset = data->getFactorOffset(index);
	factor_t *factorColumn = data->getRawFactorColumn(index);
	int numBlocks = nObs / THREADS_PER_BLOCK +
			(nObs % THREADS_PER_BLOCK ? 1 : 0);

	factorPredictKernel<<<numBlocks, THREADS_PER_BLOCK>>>(nObs,
			factorColumn, beta->getDeviceData() + betaOffset,
			linkPredictions->getDeviceData());
	CUDA_WRAP(cudaPeekAtLastError());

	return;
}
