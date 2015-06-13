#include <math.h>

#include "varianceFunctions.h"

///////////////////////////////////////////////////////////////////////////////
// Device-side Kernels For Variance Functions                                //
///////////////////////////////////////////////////////////////////////////////

__global__ void cudaBinomVar(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		output[i] = __fmul_rn(input[i], __fsub_rn(1.0, input[i]));
#else
		output[i] = input[i] * (1.0 - input[i]);
#endif
	}
	return;
}

__global__ void cudaNegBinVar(int n, num_t* input, num_t* output, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		output[i] = __fadd_rn(input[i],
				__fdividef(__fmul_rn(input[i], input[i]), k));
#else
		output[i] = input[i] + (input[i] * input[i]) / (k);
#endif
	}
	return;
}

__global__ void cudaSqVar(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		output[i] = __fmul_rn(input[i], input[i]);
#else
		output[i] = input[i] * input[i];
#endif
	}
	return;
}

__global__ void cudaCubeVar(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef CAVE_FASTMATH
		output[i] = __fmul_rn(input[i], __fmul_rn(input[i], input[i]));
#else
		output[i] = input[i] * input[i] * input[i];
#endif
	}
	return;
}

__global__ void cudaConstantVar(int n, num_t* input, num_t* output) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		output[i] = 1.0;
	}
	return;
}

__global__ void cudaPowerVar(int n, num_t* input, num_t* output, num_t k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
#ifdef GPUGLM_FASTMATH
		output[i] = __powf(input[i], k);
#else
		output[i] = pow(input[i], k);
#endif
	}
	return;
}

///////////////////////////////////////////////////////////////////////////////
// Host-side Variance Functions                                              //
///////////////////////////////////////////////////////////////////////////////

void vapply(glmVector<num_t> *input, glmVector<num_t> *output,
		void (*cudaKernel)(int, num_t*, num_t*)) {
	int numBlocks = input->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(input->getLength(),
			input->getDeviceData(), output->getDeviceData());

	return;
}

void vapply(glmVector<num_t> *input, glmVector<num_t> *output,
		void (*cudaKernel)(int, num_t*, num_t*, num_t), num_t k) {
	int numBlocks = input->getNumBlocks();

	(*cudaKernel)<<<numBlocks, THREADS_PER_BLOCK>>>(input->getLength(),
			input->getDeviceData(), output->getDeviceData(), k);

	return;
}

void varBinom(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	vapply(input, output, cudaBinomVar);
	return;
}

void varNegBin(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	vapply(input, output, cudaNegBinVar, k);
	return;
}

void varSq(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	vapply(input, output, cudaSqVar);
	return;
}

void varCube(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	vapply(input, output, cudaCubeVar);
	return;
}

void varConstant(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	vapply(input, output, cudaConstantVar);
	return;
}

void varPower(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
	vapply(input, output, cudaPowerVar, k);
	return;
}

// Special cases //////////////////////////////////////////////////////////////

void varIdentity(glmVector<num_t> *input, glmVector<num_t> *output, num_t k) {
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
