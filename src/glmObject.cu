#include "glmObject.h"

// Constructors / Destructors /////////////////////////////////////////////////

glmObject::glmObject(glmData *_data, glmFamily *_family,
		glmControl *_control) {
	data = _data;
	family = _family;
	control = _control;
	nBeta = data->getNBeta();
	nObs = data->getNObs();
	results = new glmResults(nBeta);

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
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *xColumn;
	glmVector<num_t> *weights = data->getWeights();
	varianceFunction variance = family->getVariance();
	int numericCols = xNumeric->getNCols();

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

	// Calculate the variance of the observations
	(*variance)(predictions, yVar, 0.0);
	// Handle weights of yVar if necessary
	if (weights != NULL) {
		vectorMultiply(yVar, weights, yVar);
	}

	// First, handle the intercept
	vectorSum(yVar, hessian, nBeta * nBeta - 1);

	// Next, take care of the intercept * numeric terms
	for (int i = 0; i < xNumeric->getNCols(); i++) {
		CUBLAS_WRAP(DOT(handle, nObs,
				yVar->getDeviceData(), 1,
				xNumeric->getDeviceElement(0, i), 1,
				hessian->getDeviceElement(nBeta - 1, i)));
	}

	// Finally, handle the numeric * numeric terms
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

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

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
	xNumeric->columnProduct(handle, yDelta, gradient);

	// Calculate the intercept term of the gradient
	vectorSum(yDelta, gradient, nBeta - 1);

	return;
}

void glmObject::updatePredictions(void) {
	glmMatrix<num_t> *xNumeric = data->getXNumeric();
	glmVector<num_t> *beta = results->getBeta();
	linkFunction invLink = family->getInvLink();

	xNumeric->rowProduct(handle, beta, predictions);
	// Add in the intercept
	beta->copyDeviceToHost();
	vectorAddScalar(predictions, beta->getHostData()[nBeta - 1], predictions);
	(*invLink)(predictions, predictions, 0.0);

	return;
}
