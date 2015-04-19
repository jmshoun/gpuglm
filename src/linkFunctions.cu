#include "linkFunctions.h"

#include <iostream>
#include <cstdlib>
#include <cstring>

///////////////////////////////////////////////////////////////////////////////
// Device-side Kernels For Link Functions                                    //
///////////////////////////////////////////////////////////////////////////////

// Link Functions /////////////////////////////////////////////////////////////

__global__ void cudaLogit(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __logf(__fdividef(x[i], __fsub_rn(1.0, x[i])));
#else
		x[i] = log(x[i] / (1.0 - x[i]));
#endif
	}
	return;
}
__global__ void cudaLog(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __logf(x[i]);
#else
		x[i] = log(x[i]);
#endif
	}
	return;
}
__global__ void cudaRecip(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __frcp_rn(x[i]);
#else
		x[i] = 1.0 / x[i];
#endif
	}
	return;
}
__global__ void cudaSqRecip(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __frcp_rn(__fmul_rn(x[i], x[i]));
#else
		x[i] = 1.0 / (x[i] * x[i]);
#endif
	}
	return;
}
__global__ void cudaNegBin(int n, num_t *x, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __logf(__fdividef(x[i], __fadd_rn(x[i], k)));
#else
		x[i] = log(x[i] / (k + x[i]));
#endif
	}
	return;
}

// Inverse Link Functions /////////////////////////////////////////////////////

__global__ void cudaInvLogit(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __fdividef(1.0, __fadd_rn(1.0, __expf(-x[i])));
#else
		x[i] = 1.0 / (1.0 + exp(-x[i]));
#endif
	}
	return;
}
__global__ void cudaExp(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __expf(x[i]);
#else
		x[i] = exp(x[i]);
#endif
	}
	return;
}
__global__ void cudaSqrtRecip(int n, num_t *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		x[i] = __fdividef(1.0, __fsqrt_rn(x[i]));
#else
		x[i] = 1.0  / sqrt(x[i]);
#endif
	}
	return;
}
__global__ void cudaInvNegBin(int n, num_t *x, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float p;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		p = __expf(x[i]);
		x[i] = __fdividef(__fmul_rn(p, k), __fsub_rn(1.0, p));
#else
		p = exp(x[i]);
		x[i] = (p * k) / (1 - p);
#endif
	}
	return;
}

///////////////////////////////////////////////////////////////////////////////
// Host-side Link Functions                                                  //
///////////////////////////////////////////////////////////////////////////////

void sapply(glmVector<num_t> *x, void (*cudaKernel)(int, num_t*)) {
	int numBlocks = x->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(x->getLength(),
			x->getDeviceData());

	return;
}

void sapply(glmVector<num_t> *x, void (*cudaKernel)(int, num_t*, num_t),
		num_t k) {
	int numBlocks = x->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(x->getLength(),
			x->getDeviceData(), k);

	return;
}

void linkLogit(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaLogit);
	return;
}

void linkLog(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaLog);
	return;
}

void linkRecip(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaRecip);
	return;
}

void linkSqRecip(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaSqRecip);
	return;
}

void linkNegBin(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaNegBin, k);
	return;
}

void linkIdentity(glmVector<num_t> *x, num_t k) {
	// The identity link doesn't require any computation, but we include a stub
	// for the link/inverse link function so that the application code doesn't
	// need any additional logic to handle this special case.
	return;
}

// Inverse Link Functions /////////////////////////////////////////////////////

void linkInvLogit(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaInvLogit);
	return;
}

void linkExp(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaExp);
	return;
}
void linkSqrtRecip(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaSqrtRecip);
	return;
}

void linkInvNegBin(glmVector<num_t> *x, num_t k) {
	sapply(x, cudaInvNegBin, k);
	return;
}

///////////////////////////////////////////////////////////////////////////////
// Link and Inverse Link Function Generators                                 //
///////////////////////////////////////////////////////////////////////////////

linkFunction getLinkFunction(std::string linkType) {
	linkFunction link = linkIdentity;

	if (linkType == "log") {
		link = linkLog;
	} else if (linkType == "logit") {
		link = linkLogit;
	} else if (linkType == "reciprocal") {
		link = linkRecip;
	} else if (linkType == "squared reciprocal") {
		link = linkSqRecip;
	} else if (linkType == "negative binomial") {
		link = linkNegBin;
	}

	return link;
}

linkFunction getInvLinkFunction(std::string linkType) {
	linkFunction link = linkIdentity;

	if (linkType == "log") {
		link = linkExp;
	} else if (linkType == "logit") {
		link = linkInvLogit;
	} else if (linkType == "reciprocal") {
		link = linkRecip;
	} else if (linkType == "squared reciprocal") {
		link = linkSqrtRecip;
	} else if (linkType == "negative binomial") {
		link = linkInvNegBin;
	}

	return link;
}
