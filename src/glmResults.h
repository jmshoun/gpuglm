
#ifndef GLMRESULTS_H_
#define GLMRESULTS_H_

#include "glmArray.h"

class glmResults {

protected:
	glmVector<num_t> *beta;
	unsigned int numIterations;
	num_t logLikelihood;
	bool converged;

public:
	// Constructors/Destructors ///////////////////////////////////////////////
	glmResults(glmVector<num_t> *_startingBeta) {
		beta = _startingBeta;
		beta->setSharedHost(true);
		numIterations = 0;
		logLikelihood = 0.0;
		converged = false;
	}

	// Accessors //////////////////////////////////////////////////////////////
	glmVector<num_t>* getBeta(void) const { return beta; };
	unsigned int getNumIterations(void) const { return numIterations; };
	bool getConverged(void) const { return converged; };
	int getNBeta(void) const { return beta->getLength(); };
	num_t getLogLikelihood(void) const { return logLikelihood; };

	void incrementNumIterations(void) { numIterations++; };
	void setConverged(bool _converged) { converged = _converged; };
	void setLogLikelihood(num_t _logLikelihood) {
		logLikelihood = _logLikelihood;
	};
};

#endif /* GLMRESULTS_H_ */
