
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
	glmResults(int beta_length) {
		beta = new glmVector<num_t>(beta_length, true, true, true);
		beta->setSharedHost(true);
		numIterations = 0;
		converged = false;
	}

	// Accessors //////////////////////////////////////////////////////////////
	glmVector<num_t>* getBeta(void) const { return beta; };
	unsigned int getNumIterations(void) const { return numIterations; };
	bool getConverged(void) const { return converged; };

	void incrementNumIterations(void) { numIterations++; };
	void setConverged(bool _converged) { converged = _converged; };
};

#endif /* GLMRESULTS_H_ */