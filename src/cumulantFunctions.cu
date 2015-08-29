#include <math.h>

#include "cumulantFunctions.h"

///////////////////////////////////////////////////////////////////////////////
// Device-side Kernels For Cumulant Functions                                //
///////////////////////////////////////////////////////////////////////////////

__global__ void cudaGaussianCumulant(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __fmul_rn(0.5, __fmul_rn(input[i], input[i]));
#else
		output[i] = 0.5 * input[i] * input[i];
#endif
	}
	return;
}

__global__ void cudaPoissonCumulant(int n, num_t* input, num_t* output) {
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

__global__ void cudaBinomCumulant(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __flog(__fadd_rn(1, __fexp(input[i])));
#else
		output[i] = log(1 + exp(input[i]));
#endif
	}
	return;
}

__global__ void cudaGammaCumulant(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = -__flog(-input[i]);
#else
		output[i] = -log(-input[i]);
#endif
	}
	return;
}

__global__ void cudaInvGaussianCumulant(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __fmul_rn(2.0, __fsqrt(input[i]));
#else
		output[i] = 2.0 * sqrt(input[i]);
#endif
	}
	return;
}

__global__ void cudaNegBinCumulant(int n, num_t* input, num_t* output,
		num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __fmul(-k, __flog(__fsub_rn(1.0, __fexp(input[i]))));
#else
		output[i] = -k * log(1.0 - exp(input[i]));
#endif
	}
	return;
}

__global__ void cudaPowerCumulant(int n, num_t* input, num_t* output, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __fdiv(__fpow(input[i], k), k);
#else
		output[i] = pow(input[i], k) / k;
#endif
	}
	return;
}

///////////////////////////////////////////////////////////////////////////////
// Host-side Cumulant Functions                                              //
///////////////////////////////////////////////////////////////////////////////

void capply(glmVector<num_t> *input, glmVector<num_t> *output,
		void (*cudaKernel)(int, num_t*, num_t*)) {
	int numBlocks = input->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(input->getLength(),
			input->getDeviceData(), output->getDeviceData());

	return;
}

void capply(glmVector<num_t> *input, glmVector<num_t> *output,
		void (*cudaKernel)(int, num_t*, num_t*, num_t), num_t k) {
	int numBlocks = input->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(input->getLength(),
			input->getDeviceData(), output->getDeviceData(), k);

	return;
}

void cumulantGaussian(glmVector<num_t> *input, glmVector<num_t> *output,
		num_t k) {
	capply(input, output, cudaGaussianCumulant);
	return;
}

void cumulantPoisson(glmVector<num_t> *input, glmVector<num_t> *output,
		num_t k) {
	capply(input, output, cudaPoissonCumulant);
	return;
}

void cumulantBinom(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	capply(input, output, cudaBinomCumulant);
	return;
}

void cumulantGamma(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	capply(input, output, cudaGammaCumulant);
	return;
}

void cumulantInvGaussian(glmVector<num_t> *input, glmVector<num_t> *output,
		num_t k) {
	capply(input, output, cudaInvGaussianCumulant);
	return;
}

void cumulantNegBin(glmVector<num_t> *input, glmVector<num_t> *output,
		num_t k) {
	capply(input, output, cudaNegBinCumulant, k);
	return;
}

void cumulantPower(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	// Pass in a modified version of k to save time inside the kernel
	capply(input, output, cudaPowerCumulant, (1 + 1 / k));
	return;
}


// Cumulant Function Generator ////////////////////////////////////////////////

cumulantFunction getCumulantFunction(std::string linkType) {
	cumulantFunction cumulant = cumulantGaussian;

	if (linkType == "log") {
		cumulant = cumulantPoisson;
	} else if (linkType == "logit") {
		cumulant = cumulantBinom;
	} else if (linkType == "reciprocal") {
		cumulant = cumulantGamma;
	} else if (linkType == "squared reciprocal") {
		cumulant = cumulantInvGaussian;
	} else if (linkType == "negative binomial") {
		cumulant = cumulantNegBin;
	} else if (linkType == "power") {
		cumulant = cumulantPower;
	}

	return cumulant;
}
