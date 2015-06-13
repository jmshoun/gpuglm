
#ifndef LINKFUNCTIONS_H_
#define LINKFUNCTIONS_H_

#include <string>

#include "gpuglmConfig.h"
#include "glmArray.h"

typedef void (*linkFunction)(glmVector<num_t>*, glmVector<num_t>*, num_t);

// Link Functions /////////////////////////////////////////////////////////////
void linkLogit(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkLog(glmVector<num_t> *input, glmVector<num_t> *output, num_t k = 0.0);
void linkSqRecip(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkNegBin(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k);
void linkIdentity(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkRecip(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkPower(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);

// Inverse Link Functions /////////////////////////////////////////////////////
void linkInvLogit(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkExp(glmVector<num_t> *input, glmVector<num_t> *output, num_t k = 0.0);
void linkSqrtRecip(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkInvNegBin(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k);
void linkInvPower(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
// The identity and reciprocal functions are involutions, so they don't require
// separate inverse functions.

// Derivatives of Inverse Link Functions //////////////////////////////////////
void linkLogitDerivative(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkRecipDerivative(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkSqrtRecipDerivative(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkNegBinDerivative(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);
void linkPowerDerivative(glmVector<num_t> *input,
		glmVector<num_t> *output, num_t k = 0.0);

// Link and Inverse Link Function Generators //////////////////////////////////
linkFunction getLinkFunction(std::string linkType);
linkFunction getInvLinkFunction(std::string linkType);
linkFunction getLinkDerivativeFunction(std::string linkType);

#endif /* LINKFUNCTIONS_H_ */
