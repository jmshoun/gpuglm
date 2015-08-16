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
	hessian = new glmMatrix<num_t>(nBeta, nBeta, true, true, true);
	yDelta = new glmVector<num_t>(nObs, true, true);
	yVar = yDelta;
	xScratch = new glmVector<num_t>(nObs, true, true, true);
	predictions = new glmVector<num_t>(nObs, true, true);

	CUBLAS_WRAP(cublasCreate(&handle));
	cusolverDnCreate(&solverHandle);

	CUSOLVER_WRAP(POTRF_B(solverHandle, CUBLAS_FILL_MODE_UPPER, this->nBeta,
			this->hessian->getDeviceData(), this->nBeta, &workspaceSize));
	CUDA_WRAP(cudaMalloc((void **) &workspace, sizeof(num_t) * workspaceSize));
	CUDA_WRAP(cudaMalloc((void **) &devInfo, sizeof(int)));;

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
	delete hessian;
	delete xScratch;

	CUDA_WRAP(cudaFree(workspace));
	CUDA_WRAP(cudaFree(devInfo));
	CUBLAS_WRAP(cublasDestroy(handle));
	CUSOLVER_WRAP(cusolverDnDestroy(solverHandle));
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

// Updating Functions /////////////////////////////////////////////////////////

void glmObject::solve(void) {
	num_t maxDelta = 1000.0;
	num_t thisDelta;
	num_t *hostBetaDelta = betaDelta->getHostData();

	while ((maxDelta > control->getTolerance()) &&
			(results->getNumIterations() < control->getMaxIterations())) {
		results->incrementNumIterations();
		std::cout << "Iteration #" << results->getNumIterations() << std::endl;

		this->updateGradient();
		this->updateHessian();
		this->solveHessian();

		vectorAdd(results->getBeta(), betaDelta, results->getBeta());

		betaDelta->copyDeviceToHost();
		maxDelta = 0.0;
		for (int i = 0; i < nBeta; i++) {
			thisDelta = fabs(hostBetaDelta[i]);
			maxDelta = thisDelta > maxDelta ? thisDelta : maxDelta;
		}
	}

	if (maxDelta < control->getTolerance()) {
		results->setConverged(true);
	}

	return;
}

void glmObject::updateHessian(void) {
	glmVector<num_t> *weights = data->getWeights();
	varianceFunction variance = family->getVariance();

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

	// Calculate the variance of the observations
	(*variance)(predictions, yVar, 0.0);
	// Handle weights of yVar if necessary
	if (weights != NULL) {
		vectorMultiply(yVar, weights, yVar);
	}

	this->updateHessianInterceptIntercept();
	this->updateHessianInterceptNumeric();
	this->updateHessianInterceptFactor();
	this->updateHessianNumericNumeric();
	this->updateHessianNumericFactor();
	this->updateHessianFactorFactor();

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

	return;
}

void glmObject::updateHessianInterceptIntercept(void) {
	vectorSum(yVar, hessian, nBeta * nBeta - 1);
	return;
}

void glmObject::updateHessianInterceptNumeric(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();

	for (int i = 0; i < xNumeric->getNCols(); i++) {
		CUBLAS_WRAP(DOT(handle, nObs,
				yVar->getDeviceData(), 1,
				xNumeric->getDeviceElement(0, i), 1,
				hessian->getDeviceElement(nBeta - 1, i)));
	}

	return;
}

void glmObject::updateHessianInterceptFactor(void) {
	return;
}

void glmObject::updateHessianNumericNumeric(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *xColumn;
	int numericCols = xNumeric->getNCols();

	for (int i = 0; i < numericCols; i++) {
		xColumn = xNumeric->getDeviceColumn(i);
		vectorMultiply(yVar, xColumn, xScratch);
		delete xColumn;

		for (int j = i; j < numericCols; j++) {
			CUBLAS_WRAP(DOT(handle, nObs,
					xScratch->getDeviceData(), 1,
					xNumeric->getDeviceElement(0, j), 1,
					hessian->getDeviceElement(j, i)));
		}
	}

	return;
}

void glmObject::updateHessianNumericFactor(void) {
	return;
}

void glmObject::updateHessianFactorFactor(void) {
	return;
}

void glmObject::solveHessian(void) {
	// Copy the gradient to the update vector
	copyDeviceToDevice(betaDelta, gradient);

	CUSOLVER_WRAP(POTRF(solverHandle, CUBLAS_FILL_MODE_LOWER,
			nBeta, hessian->getDeviceData(), nBeta,
			workspace, workspaceSize, devInfo));
	CUSOLVER_WRAP(POTRS(solverHandle, CUBLAS_FILL_MODE_LOWER,
			nBeta, 1, hessian->getDeviceData(), nBeta,
			betaDelta->getDeviceData(), nBeta, devInfo));

	return;
}

void glmObject::updateGradient(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *y = data->getY();
	glmVector<num_t> *weights = data->getWeights();

	this->updatePredictions();

	// Calculate yDelta as y - yHat
	vectorDifference(y, predictions, yDelta);

	// Add weights to yDelta
	if (weights != NULL) {
		vectorMultiply(yDelta, weights, yDelta);
	}

	// Calculate yDelta %*% xNumeric
	if (data->getXNumeric() != NULL) {
		xNumeric->columnProduct(handle, yDelta, gradient);
	}

	// Calculate yDelta %*% xFactor
	if (data->getXFactor() != NULL) {
		for (int i = 0; i < data->getNFactors(); i++) {
			this->updateGradientFactor(i);
		}
	}

	// Calculate the intercept term of the gradient
	vectorSum(yDelta, gradient, nBeta - 1);

	gradient->copyDeviceToHost();
	std::cout << *(gradient) << std::endl;

	return;
}

void glmObject::updateGradientFactor(int index) {
	int gradientOffset = data->getFactorOffset(index) + 2;
	factor_t *factorColumn = data->getFactorColumn(index);

	CUDA_WRAP(cudaMemset(gradient->getDeviceElement(gradientOffset),
			0, 2 * sizeof(num_t)));
	factorProductKernel<<<1, 1>>>(nObs, factorColumn, yDelta->getDeviceData(),
			gradient->getDeviceElement(gradientOffset));
	CUDA_WRAP(cudaPeekAtLastError());

	return;
}

void glmObject::updatePredictions(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *beta = results->getBeta();
	linkFunction invLink = family->getInvLink();

	if (xNumeric != NULL) {
		xNumeric->rowProduct(handle, beta, predictions);
	}

	if (data->getXFactor() != NULL) {
		for (int i = 0; i < data->getNFactors(); i++) {
			this->predictFactor(i);
		}
	}

	// Add in the intercept
	beta->copyDeviceToHost();
	vectorAddScalar(predictions, beta->getHostData()[nBeta - 1], predictions);
	(*invLink)(predictions, predictions, 0.0);

	return;
}

void glmObject::predictFactor(int index) {
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
