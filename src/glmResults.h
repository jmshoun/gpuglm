
#ifndef GLMRESULTS_H_
#define GLMRESULTS_H_

#include "glmArray.h"

class glmResults {

protected:
	glmVector<num_t> *beta;
	unsigned int numIterations;
	bool converged;

public:
	// Constructors/Destructors ///////////////////////////////////////////////
	glmResults(glmVector<num_t> *_startingBeta) {
		beta = _startingBeta;
		beta->setSharedHost(true);
		numIterations = 0;
		converged = false;
	}

	// Accessors //////////////////////////////////////////////////////////////
	glmVector<num_t>* getBeta(void) const { return beta; };
	unsigned int getNumIterations(void) const { return numIterations; };
	bool getConverged(void) const { return converged; };
	int getNBeta(void) const { return beta->getLength(); };

	void incrementNumIterations(void) { numIterations++; };
	void setConverged(bool _converged) { converged = _converged; };
};

#endif /* GLMRESULTS_H_ */
