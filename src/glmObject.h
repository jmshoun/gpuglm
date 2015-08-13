
#ifndef GLMOBJECT_H_
#define GLMOBJECT_H_

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "glmData.h"
#include "glmFamily.h"
#include "glmControl.h"
#include "glmResults.h"

#include "glmException.h"

class glmObject {
protected:
	int nObs;
	int nBeta;

	glmData *data;
	glmFamily *family;
	glmControl *control;
	glmResults *results;

	glmVector<num_t> *predictions;
	glmVector<num_t> *yDelta;
	glmVector<num_t> *yVar;
	glmVector<num_t> *xScratch;
	glmVector<num_t> *gradient;
	glmVector<num_t> *betaDelta;
	glmMatrix<num_t> *hessian;

	num_t *workspace;
	int workspaceSize;
	int *devInfo;

	cublasHandle_t handle;
	cusolverDnHandle_t solverHandle;

public:
	// Constructors / Destructros /////////////////////////////////////////////
	glmObject(glmData *_data, glmFamily *_family, glmControl *_control,
			glmVector<num_t> *_startingBeta);
	~glmObject();

	// Updating Functions /////////////////////////////////////////////////////
	void updateGradient(void);
	void updateHessian(void);
	void solveHessian(void);
	void updatePredictions(void);

	void solve(void);

	// Accessors //////////////////////////////////////////////////////////////
	glmResults* getResults(void) const { return results; };

	glmVector<num_t>* getGradient(void) {
		gradient->copyDeviceToHost();
		return gradient;
	};

	glmMatrix<num_t>* getHessian(void) {
		hessian->copyDeviceToHost();
		return hessian;
	};

	glmVector<num_t>* getPredictions(void) {
		predictions->copyDeviceToHost();
		return predictions;
	};
};

#endif /* GLMOBJECT_H_ */
