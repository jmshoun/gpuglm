#ifndef GLMDATA_H_
#define GLMDATA_H_

#include "glmArray.h"

class glmData {

protected:
	glmVector<num_t> *y;
	glmMatrix<num_t> *xNumeric;
	glmMatrix<factor_t> *xFactor;
	glmVector<num_t> *weights;

public:
	// Constructors / Destructors /////////////////////////////////////////////
	glmData(glmVector<num_t> *_y, glmMatrix<num_t> *_xNumeric,
			glmMatrix<factor_t> *_xFactor = NULL,
			glmVector<num_t> *_weights = NULL);
	~glmData();

	// Accessors //////////////////////////////////////////////////////////////
	glmVector<num_t>* getY(void) const { return y; };
	glmMatrix<num_t>* getXNumeric(void) const { return xNumeric; };
	glmMatrix<factor_t>* getXFactor(void) const { return xFactor; };
	glmVector<num_t>* getWeights(void) const { return weights; };

	// Derived Accessors //////////////////////////////////////////////////////
	int getNObs(void);
	int getNBeta(void);
};

#endif /* GLMDATA_H_ */
