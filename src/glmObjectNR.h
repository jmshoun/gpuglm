#ifndef GLMOBJECTNR_H_
#define GLMOBJECTNR_H_

#include <cusolverDn.h>

#include "glmObject.h"

class glmObjectNR : public glmObject {
protected:
	glmMatrix<num_t> *hessian;
	glmVector<num_t> *yVar;

	num_t *workspace;
	int workspaceSize;
	int *devInfo;

	cusolverDnHandle_t solverHandle;

	// Internal Updating Functions ////////////////////////////////////////////
	void updateHessianInterceptIntercept(void);
	void updateHessianInterceptNumeric(void);
	void updateHessianInterceptFactor(void);
	void updateHessianNumericNumeric(void);
	void updateHessianNumericFactor(void);
	void updateHessianFactorFactor(void);

public:
	// Constructors / Destructros /////////////////////////////////////////////
	glmObjectNR(glmData *_data, glmFamily *_family, glmControl *_control,
			glmVector<num_t> *_startingBeta) : glmObject(_data, _family,
					_control, _startingBeta) {
		hessian = new glmMatrix<num_t>(nBeta, nBeta, true, true, true);
		yVar = yDelta;

		cusolverDnCreate(&solverHandle);

		CUSOLVER_WRAP(POTRF_B(solverHandle, CUBLAS_FILL_MODE_UPPER, this->nBeta,
				this->hessian->getDeviceData(), this->nBeta, &workspaceSize));
		CUDA_WRAP(cudaMalloc((void **) &workspace,
				sizeof(num_t) * workspaceSize));
		CUDA_WRAP(cudaMalloc((void **) &devInfo, sizeof(int)));

		return;
	};

	~glmObjectNR();

	// Updating Functions /////////////////////////////////////////////////////
	void updateHessian(void);
	void solveHessian(void);

	void solve(void);

	// Accessors //////////////////////////////////////////////////////////////
	glmMatrix<num_t>* getHessian(void) {
		hessian->copyDeviceToHost();
		return hessian;
	};
};


#endif /* GLMOBJECTNR_H_ */
