
#ifndef VARIANCEFUNCTIONS_H_
#define VARIANCEFUNCTIONS_H_

#include <string>

#include "gpuglmConfig.h"
#include "glmArray.h"

typedef void (*varianceFunction)(glmVector<num_t>*, num_t);

// Variance Functions /////////////////////////////////////////////////////////
void varBinom(glmVector<num_t> *x, num_t k = 0.0);
void varNegBin(glmVector<num_t> *x, num_t k);
void varSq(glmVector<num_t> *x, num_t k = 0.0);
void varCube(glmVector<num_t> *x, num_t k = 0.0);
void varIdentity(glmVector<num_t> *x, num_t k = 0.0);
void varConstant(glmVector<num_t> *x, num_t k = 0.0);

// Link and Inverse Link Function Generators //////////////////////////////////
varianceFunction getVarianceFunction(std::string varianceType);

#endif /* VARIANCEFUNCTIONS_H_ */
