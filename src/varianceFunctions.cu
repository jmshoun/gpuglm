#include <math.h>

#include "varianceFunctions.h"

///////////////////////////////////////////////////////////////////////////////
// Device-side Kernels For Variance Functions                                //
///////////////////////////////////////////////////////////////////////////////

__global__ void cudaBinomVar(int n, num_t* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		x[i] = __fmul_rn(x[i], __fsub_rn(1.0, x[i]));
#else
		x[i] = x[i] * (1.0 - x[i]);
#endif
	}
	return;
}

__global__ void cudaNegBinVar(int n, num_t* x, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		x[i] = __fadd_rn(x[i], __fdividef(__fmul_rn(x[i], x[i]), k));
#else
		x[i] = x[i] + (x[i] * x[i]) / (k);
#endif
	}
	return;
}

__global__ void cudaSqVar(int n, num_t* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		x[i] = __fmul_rn(x[i], x[i]);
#else
		x[i] = x[i] * x[i];
#endif
	}
	return;
}

__global__ void cudaCubeVar(int n, num_t* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		x[i] = __fmul_rn(x[i], __fmul_rn(x[i], x[i]));
#else
		x[i] = x[i] * x[i] * x[i];
#endif
	}
	return;
}

__global__ void cudaConstantVar(int n, num_t* x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		x[i] = 1.0;
	}
	return;
}

///////////////////////////////////////////////////////////////////////////////
// Host-side Variance Functions                                              //
///////////////////////////////////////////////////////////////////////////////

void vapply(glmVector<num_t> *x, void (*cudaKernel)(int, num_t*)) {
	int numBlocks = x->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(x->getLength(),
			x->getDeviceData());

	return;
}

void vapply(glmVector<num_t> *x, void (*cudaKernel)(int, num_t*, num_t),
		num_t k) {
	int numBlocks = x->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(x->getLength(),
			x->getDeviceData(), k);

	return;
}

void varBinom(glmVector<num_t> *x, num_t k) {
	vapply(x, cudaBinomVar);
	return;
}

void varNegBin(glmVector<num_t> *x, num_t k) {
	vapply(x, cudaNegBinVar, k);
	return;
}

void varSq(glmVector<num_t> *x, num_t k) {
	vapply(x, cudaSqVar);
	return;
}

void varCube(glmVector<num_t> *x, num_t k) {
	vapply(x, cudaCubeVar);
	return;
}

void varConstant(glmVector<num_t> *x, num_t k) {
	vapply(x, cudaConstantVar);
	return;
}

// Special cases //////////////////////////////////////////////////////////////

void varIdentity(glmVector<num_t> *x, num_t k) {
	return;
}

varianceFunction getVarianceFunction(std::string varianceType) {
	varianceFunction variance = varIdentity;

	if (varianceType == "binomial") {
		variance = varBinom;
	} else if (varianceType == "negative binomial") {
		variance = varNegBin;
	} else if (varianceType == "squared") {
		variance = varSq;
	} else if (varianceType == "cubed") {
		variance = varCube;
	} else if (varianceType == "constant") {
		variance = varConstant;
	}

	return variance;
}
