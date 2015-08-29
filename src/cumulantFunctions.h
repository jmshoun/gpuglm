#ifndef CUMULANTFUNCTIONS_H_
#define CUMULANTFUNCTIONS_H_

#include <string>

#include "gpuglmConfig.h"
#include "glmArray.h"

typedef void (*cumulantFunction)(glmVector<num_t>*, glmVector<num_t>*, num_t);

// Cumulant Functions /////////////////////////////////////////////////////////
void cumulantGaussian(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void cumulantPoisson(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void cumulantBinom(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void cumulantGamma(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void cumulantInvGaussian(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void cumulantNegBin(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void cumulantPower(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);

// Cumulant Function Generator ////////////////////////////////////////////////
cumulantFunction getCumulantFunction(std::string cumulantType);

#endif /* CUMULANTFUNCTIONS_H_ */
