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
	xScratch = new glmVector<num_t>(nObs, true, true);
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
	cusolverDnDestroy(solverHandle);
}

// Updating Functions /////////////////////////////////////////////////////////

void glmObject::solve(void) {
	num_t minDelta = 1000.0;
	num_t *hostBetaDelta = this->betaDelta->getHostData();

	while ((minDelta > this->control->getTolerance()) &&
			(this->results->getNumIterations() <
					this->control->getMaxIterations())) {
		this->results->incrementNumIterations();

		this->updateGradient();
		this->updateHessian();
		this->solveHessian();
		vectorAdd(this->results->getBeta(), this->betaDelta,
				this->results->getBeta());

		this->betaDelta->copyDeviceToHost();
		minDelta = hostBetaDelta[0];
		for (int i = 1; i < this->nBeta; i++) {
			minDelta = hostBetaDelta[i] > minDelta ?
					hostBetaDelta[i] : minDelta;
		}
	}

	if (minDelta < this->control->getTolerance()) {
		this->results->setConverged(true);
	}

	return;
}

void glmObject::updateHessian(void) {
	glmMatrix<num_t> *xNumeric = this->data->getXNumeric();
	glmVector<num_t> *xColumn;
	varianceFunction variance = this->family->getVariance();
	int numericCols = xNumeric->getNCols();

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

	// Calculate the variance of the observations
	copyDeviceToDevice(this->yVar, this->predictions);
	(*variance)(this->yVar, 0.0);

	// First, handle the intercept
	vectorSum(this->yVar, this->hessian, this->nBeta * this->nBeta - 1);

	// Next, take care of the intercept * numeric terms
	for (int i = 0; i < xNumeric->getNCols(); i++) {
		CUBLAS_WRAP(DOT(this->handle, this->nObs,
				this->yVar->getDeviceData(), 1,
				xNumeric->getDeviceData() + this->nObs * i, 1,
				this->hessian->getDeviceData() + this->nBeta * (this->nBeta - 1) + i));
	}

	// Finally, handle the numeric * numeric terms
	for (int i = 0; i < numericCols; i++) {
		xColumn = xNumeric->getDeviceColumn(i);
		vectorMultiply(this->yVar, xColumn, this->xScratch);
		delete xColumn;

		for (int j = i; j < numericCols; j++) {
			CUBLAS_WRAP(DOT(this->handle, this->nObs,
					this->xScratch->getDeviceData(), 1,
					xNumeric->getDeviceData() + this->nObs * j, 1,
					this->hessian->getDeviceData() + j * this->nBeta + i));
		}
	}

	CUBLAS_WRAP(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

	return;
}

void glmObject::solveHessian(void) {
	// Copy the gradient to the update vector
	copyDeviceToDevice(this->betaDelta, this->gradient);

	CUSOLVER_WRAP(POTRF(this->solverHandle, CUBLAS_FILL_MODE_UPPER,
			this->nBeta, this->hessian->getDeviceData(), this->nBeta,
			this->workspace, this->workspaceSize, this->devInfo));
	CUSOLVER_WRAP(POTRS(this->solverHandle, CUBLAS_FILL_MODE_UPPER,
			this->nBeta, 1, this->hessian->getDeviceData(), this->nBeta,
			this->betaDelta->getDeviceData(), this->nBeta, this->devInfo));

	return;
}

void glmObject::updateGradient(void) {
	glmMatrix<num_t> *xNumeric = this->data->getXNumeric();
	glmVector<num_t> *y = this->data->getY();

	this->updatePredictions();

	// Calculate yDelta as y - yHat
	vectorDifference(y, this->predictions, this->yDelta);

	// Calculate yDelta %*% xNumeric
	xNumeric->columnProduct(this->handle, this->yDelta, this->gradient);

	// Calculate the intercept term of the gradient
	vectorSum(this->yDelta, this->gradient, this->nBeta - 1);

	return;
}

void glmObject::updatePredictions(void) {
	glmMatrix<num_t> *xNumeric = this->data->getXNumeric();
	glmVector<num_t> *beta = this->results->getBeta();
	linkFunction invLink = this->family->getInvLink();

	xNumeric->rowProduct(this->handle, beta, this->predictions);
	// Add in the intercept
	beta->copyDeviceToHost();
	vectorAddScalar(this->predictions, beta->getHostData()[this->nBeta - 1],
			this->predictions);
	(*invLink)(this->predictions, 0.0);

	return;
}
