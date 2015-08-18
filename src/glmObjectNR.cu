#include "glmObjectNR.h"

// Constructors / Destructors /////////////////////////////////////////////////

glmObjectNR::~glmObjectNR() {
	delete hessian;
	delete xScratch;

	CUDA_WRAP(cudaFree(workspace));
	CUDA_WRAP(cudaFree(devInfo));
	CUSOLVER_WRAP(cusolverDnDestroy(solverHandle));
}

__global__ void factorProductAltKernel(int n, factor_t *factor, num_t *numeric,
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

// General Solving Function ///////////////////////////////////////////////////

void glmObjectNR::solve(void) {
	num_t maxDelta = control->getTolerance() * 10;
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

void glmObjectNR::solveHessian(void) {
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

// Hessian Updating ///////////////////////////////////////////////////////////

void glmObjectNR::updateHessian(void) {
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

void glmObjectNR::updateHessianInterceptIntercept(void) {
	vectorSum(yVar, hessian, nBeta * nBeta - 1);
	return;
}

void glmObjectNR::updateHessianInterceptNumeric(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();

	for (int i = 0; i < xNumeric->getNCols(); i++) {
		CUBLAS_WRAP(DOT(handle, nObs,
				yVar->getDeviceData(), 1,
				xNumeric->getDeviceElement(0, i), 1,
				hessian->getDeviceElement(i, nBeta - 1)));
	}

	return;
}

void glmObjectNR::updateHessianInterceptFactor(void) {
	glmMatrix<factor_t> *xFactor = data->getXFactor();
	int indexOffset;
	factor_t *factorColumn;

	for (int i = 0; i < data->getNFactors(); i++) {
		factorColumn = data->getFactorColumn(i);
		indexOffset = data->getFactorOffset(i) + 2;

		factorProductAltKernel<<<1, 1>>>(nObs, factorColumn,
				yVar->getDeviceData(),
				hessian->getDeviceElement(indexOffset, nBeta - 1));
		CUDA_WRAP(cudaPeekAtLastError());
	}

	return;
}

void glmObjectNR::updateHessianNumericNumeric(void) {
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

void glmObjectNR::updateHessianNumericFactor(void) {
	glmMatrix<factor_t> *xFactor = data->getXFactor();
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *xColumn;
	int numericCols = xNumeric->getNCols();
	int indexOffset, factorLength;
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
			factorLength = data->getFactorLength(factorIndex);

			CUDA_WRAP(cudaMalloc((void **) &tempResults,
					factorLength * sizeof(num_t)));
			CUDA_WRAP(cudaMemset(tempResults, 0,
					factorLength * sizeof(num_t)));

			factorProductAltKernel<<<1, 1>>>(nObs, factorColumn,
					xScratch->getDeviceData(),
					tempResults);
			CUDA_WRAP(cudaPeekAtLastError());

			for (int i = 0; i < factorLength; i++) {
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

void glmObjectNR::updateHessianFactorFactor(void) {
	int indexOffset, factorLength;

	for (int i = 0; i < data->getNFactors(); i++) {
		indexOffset = data->getFactorOffset(i) + 2;
		factorLength = data->getFactorLength(i);
		for (int j = 0; j < data->getNFactors(); j++) {
			if (i == j) {
				for (int k = 0; k < factorLength; k++) {
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
