#include "linkFunctions.h"

#include <iostream>
#include <cstdlib>
#include <cstring>

///////////////////////////////////////////////////////////////////////////////
// Device-side Kernels For Link Functions                                    //
///////////////////////////////////////////////////////////////////////////////

// Link Functions /////////////////////////////////////////////////////////////

__global__ void cudaLogit(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __logf(__fdividef(input[i], __fsub_rn(1.0, input[i])));
#else
		output[i] = log(input[i] / (1.0 - input[i]));
#endif
	}
	return;
}
__global__ void cudaLog(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __logf(input[i]);
#else
		output[i] = log(input[i]);
#endif
	}
	return;
}
__global__ void cudaRecip(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __frcp_rn(input[i]);
#else
		output[i] = 1.0 / input[i];
#endif
	}
	return;
}
__global__ void cudaSqRecip(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __frcp_rn(__fmul_rn(input[i], input[i]));
#else
		output[i] = 1.0 / (input[i] * input[i]);
#endif
	}
	return;
}
__global__ void cudaNegBin(int n, num_t *input, num_t *output, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __logf(__fdividef(input[i], __fadd_rn(input[i], k)));
#else
		output[i] = log(input[i] / (k + input[i]));
#endif
	}
	return;
}

// Inverse Link Functions /////////////////////////////////////////////////////

__global__ void cudaInvLogit(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __fdividef(1.0, __fadd_rn(1.0, __expf(-input[i])));
#else
		output[i] = 1.0 / (1.0 + exp(-input[i]));
#endif
	}
	return;
}
__global__ void cudaExp(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __expf(input[i]);
#else
		output[i] = exp(input[i]);
#endif
	}
	return;
}
__global__ void cudaSqrtRecip(int n, num_t *input, num_t *output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __fdividef(1.0, __fsqrt_rn(input[i]));
#else
		output[i] = 1.0  / sqrt(input[i]);
#endif
	}
	return;
}
__global__ void cudaInvNegBin(int n, num_t *input, num_t *output, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float p;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		p = __expf(input[i]);
		output[i] = __fdividef(__fmul_rn(p, k), __fsub_rn(1.0, p));
#else
		p = exp(input[i]);
		output[i] = (p * k) / (1 - p);
#endif
	}
	return;
}

///////////////////////////////////////////////////////////////////////////////
// Host-side Link Functions                                                  //
///////////////////////////////////////////////////////////////////////////////

void sapply(glmVector<num_t> *input, glmVector<num_t> *output,
		void (*cudaKernel)(int, num_t*, num_t*)) {
	int numBlocks = input->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(input->getLength(),
			input->getDeviceData(), output->getDeviceData());

	return;
}

void sapply(glmVector<num_t> *input, glmVector<num_t> *output,
		void (*cudaKernel)(int, num_t*, num_t*, num_t), num_t k) {
	int numBlocks = input->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(input->getLength(),
			input->getDeviceData(), output->getDeviceData(), k);

	return;
}

void linkLogit(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaLogit);
	return;
}

void linkLog(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaLog);
	return;
}

void linkRecip(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaRecip);
	return;
}

void linkSqRecip(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaSqRecip);
	return;
}

void linkNegBin(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaNegBin, k);
	return;
}

void linkIdentity(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	// The identity link doesn't require any computation, but we include a stub
	// for the link/inverse link function so that the application code doesn't
	// need any additional logic to handle this special case.
	return;
}

// Inverse Link Functions /////////////////////////////////////////////////////

void linkInvLogit(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaInvLogit);
	return;
}

void linkExp(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaExp);
	return;
}
void linkSqrtRecip(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaSqrtRecip);
	return;
}

void linkInvNegBin(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	sapply(input, output, cudaInvNegBin, k);
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
