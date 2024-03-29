
#ifndef GLMOBJECT_H_
#define GLMOBJECT_H_

#include <cublas_v2.h>

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

	glmVector<num_t> *linkPredictions;
	glmVector<num_t> *responsePredictions;
	glmVector<num_t> *yDelta;
	glmVector<num_t> *xScratch;
	glmVector<num_t> *gradient;
	glmVector<num_t> *betaDelta;

	cublasHandle_t handle;

	// Internal Updating Functions ////////////////////////////////////////////
	void updateGradientIntercept(void);
	void updateGradientXNumeric(void);
	void updateGradientXFactor(void);
	void updateGradientSingleFactor(int index);


	void updateLinkPredictions(void);
	void updateLinkPredictionIntercept(void);
	void updateLinkPredictionXNumeric(void);
	void updateLinkPredictionXFactor(void);
	void updateLinkPredictionSingleFactor(int index);

public:
	// Constructors / Destructors /////////////////////////////////////////////
	glmObject(glmData *_data, glmFamily *_family, glmControl *_control,
			glmVector<num_t> *_startingBeta);
	~glmObject();

	// Updating Functions /////////////////////////////////////////////////////
	void updateGradient(void);
	void updateHessian(void);
	void updatePredictions(void);
	void updateLogLikelihood(void);

	void solve(void);

	// Accessors //////////////////////////////////////////////////////////////
	glmResults* getResults(void) const { return results; };

	glmVector<num_t>* getGradient(void) {
		gradient->copyDeviceToHost();
		return gradient;
	};

	glmVector<num_t>* getLinkPredictions(void) {
		linkPredictions->copyDeviceToHost();
		return linkPredictions;
	};

	glmVector<num_t>* getResponsePredictions(void) {
			responsePredictions->copyDeviceToHost();
			return responsePredictions;
		};
};

#endif /* GLMOBJECT_H_ */
