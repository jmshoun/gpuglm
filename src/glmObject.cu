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
		results->getBeta()->copyDeviceToHost();
		std::cout << *(results->getBeta()) << std::endl;

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

	CUDA_WRAP(cudaMemset(hessian->getDeviceElement(0, 0), 0,
			nBeta * nBeta * sizeof(num_t)));

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

	// Calculate the variance of the observations
	(*variance)(predictions, yVar, 0.0);
	// Handle weights of yVar if necessary
	if (weights != NULL) {
		vectorMultiply(yVar, weights, yVar);
	}

	this->updateHessianInterceptIntercept();
	if (data->getXNumeric() != NULL) {
		this->updateHessianInterceptNumeric();
		this->updateHessianNumericNumeric();
	}
	if (data->getXFactor() != NULL) {
		this->updateHessianInterceptFactor();
		this->updateHessianFactorFactor();
	}
	if ((data->getXNumeric() != NULL) && (data->getXFactor() != NULL)) {
		this->updateHessianNumericFactor();
	}

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

	hessian->copyDeviceToHost();
	std::cout << *(hessian) << std::endl;

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
				hessian->getDeviceElement(i, nBeta - 1)));
	}

	return;
}

void glmObject::updateHessianInterceptFactor(void) {
	glmMatrix<factor_t> *xFactor = data->getXFactor();
	int indexOffset;
	factor_t *factorColumn;

	for (int i = 0; i < data->getNFactors(); i++) {
		factorColumn = data->getFactorColumn(i);
		indexOffset = data->getFactorOffset(i) + 2;

		factorProductKernel<<<1, 1>>>(nObs, factorColumn,
				yVar->getDeviceData(),
				hessian->getDeviceElement(indexOffset, nBeta - 1));
		CUDA_WRAP(cudaPeekAtLastError());
	}

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
					hessian->getDeviceElement(i, j)));
		}
	}

	return;
}

void glmObject::updateHessianNumericFactor(void) {
	glmMatrix<factor_t> *xFactor = data->getXFactor();
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *xColumn;
	int numericCols = xNumeric->getNCols();
	int indexOffset;
	factor_t *factorColumn;
	num_t *tempResults;

	for (int numericIndex = 0; numericIndex < numericCols; numericIndex++) {
		xColumn = xNumeric->getDeviceColumn(numericIndex);
		vectorMultiply(yVar, xColumn, xScratch);
		delete xColumn;

		for (int factorIndex = 0; factorIndex < data->getNFactors();
					factorIndex++) {
			factorColumn = data->getFactorColumn(factorIndex);
			indexOffset = data->getFactorOffset(factorIndex) + 2;

			CUDA_WRAP(cudaMalloc((void **) &tempResults, 1 * sizeof(num_t)));
			CUDA_WRAP(cudaMemset(tempResults, 0, 1 * sizeof(num_t)));

			factorProductKernel<<<1, 1>>>(nObs, factorColumn,
					xScratch->getDeviceData(),
					tempResults);
			CUDA_WRAP(cudaPeekAtLastError());

			for (int i = 0; i < 1; i++) {
				cudaMemcpy(hessian->getDeviceElement(numericIndex, indexOffset + i),
						tempResults + i,
						sizeof(num_t),
						cudaMemcpyDeviceToDevice);
			}

			CUDA_WRAP(cudaFree(tempResults));
		}
	}

	return;
}

void glmObject::updateHessianFactorFactor(void) {
	int indexOffset;

	for (int i = 0; i < data->getNFactors(); i++) {
		indexOffset = data->getFactorOffset(i) + 2;
		for (int j = 0; j < data->getNFactors(); j++) {
			if (i == j) {
				for (int k = 0; k < 1; k++) {
					CUDA_WRAP(cudaMemcpy(hessian->getDeviceElement(indexOffset + k, indexOffset + k),
							hessian->getDeviceElement(indexOffset + k, nBeta - 1),
							sizeof(num_t),
							cudaMemcpyDeviceToDevice));
				}
			}
		}
	}

	return;
}

void glmObject::solveHessian(void) {
	// Copy the gradient to the update vector
	copyDeviceToDevice(betaDelta, gradient);

	CUSOLVER_WRAP(POTRF(solverHandle, CUBLAS_FILL_MODE_UPPER,
			nBeta, hessian->getDeviceData(), nBeta,
			workspace, workspaceSize, devInfo));
	CUSOLVER_WRAP(POTRS(solverHandle, CUBLAS_FILL_MODE_UPPER,
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
			0, 1 * sizeof(num_t)));
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
