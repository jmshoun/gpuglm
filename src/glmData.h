#ifndef GLMDATA_H_
#define GLMDATA_H_

#include "glmArray.h"

class glmData {

protected:
	glmVector<num_t> *y;
	glmMatrix<num_t> *xNumeric;
	glmVector<num_t> *weights;

public:
	// Constructors / Destructors /////////////////////////////////////////////
	glmData(int nRows, num_t *_y, num_t *_xNumeric, int nColsNumeric,
			num_t *_weights = NULL);
	~glmData();

	// Accessors //////////////////////////////////////////////////////////////
	glmVector<num_t>* getY(void) { return y; };
	glmMatrix<num_t>* getXNumeric(void) { return xNumeric; };
	glmVector<num_t>* getWeights(void) { return weights; };

	// Derived Accessors //////////////////////////////////////////////////////
	int getNObs(void);
	int getNBeta(void);
};

#endif /* GLMDATA_H_ */
