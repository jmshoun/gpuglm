#include "glmArray.h"

void copyDeviceToDevice(glmVector<num_t> *destination,
		glmVector<num_t> *source) {
	CUDA_WRAP(cudaMemcpy(destination->getDeviceData(), source->getDeviceData(),
			destination->getLength() * sizeof(num_t),
			cudaMemcpyDeviceToDevice));
	return;
}

__global__ void vectorAddScalarKernel(int n, num_t *a, num_t b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b;
	}

	return;
}

void vectorAddScalar(glmVector<num_t> *a, num_t b, glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorAddScalarKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b, c->getDeviceData());

	return;
}

__global__ void vectorAddKernel(int n, num_t *a, num_t *b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b[i];
	}

	return;
}

void vectorAdd(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorAddKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b->getDeviceData(), c->getDeviceData());

	return;
}

__global__ void vectorDifferenceKernel(int n, num_t *a, num_t *b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] - b[i];
	}

	return;
}

void vectorDifference(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorDifferenceKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b->getDeviceData(), c->getDeviceData());

	return;
}

__global__ void vectorMultiplyKernel(int n, num_t *a, num_t *b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] * b[i];
	}

	return;
}

void vectorMultiply(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorMultiplyKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b->getDeviceData(), c->getDeviceData());

	return;
}

__global__ void vectorSumKernel(int n, int k, num_t *x, num_t *sum) {
	int i, start;

	start = threadIdx.x * k;
	for (i = start; (i < start + k) && (i < n); i++) {
		sum[threadIdx.x] += x[i];
	}
}

__global__ void vectorSumKernelSlow(int n, num_t *x, num_t *sum) {
	int i;

	for (i = 0; i < n; i++) {
		sum[0] += x[i];
	}

	return;
}

void vectorSum(glmVector<num_t> *vector, glmArray<num_t> *result,
		int resultIndex) {
	num_t *finalResult = result->getDeviceData() + resultIndex;

	CUDA_WRAP(cudaMemset((void *) finalResult, 0, sizeof(num_t)));
	vectorSumKernelSlow<<<1, 1>>>(vector->getLength(),
			vector->getDeviceData(), finalResult);

	return;
}

// Debugging Functions ////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const glmVector<num_t>& glmVec) {
	os << "[" << glmVec.getHostData()[0];
	for (int i = 1; i < glmVec.getLength(); i++) {
		os << ", " << glmVec.getHostData()[i];
	}
	os << "]";

	return os;
};

std::ostream& operator<<(std::ostream& os, const glmMatrix<num_t>& glmMat) {
	for (int i = 0; i < glmMat.getNRows(); i++) {
		if (i == 0) {
			os << "[[";
		} else {
			os << " [";
		}

		os << glmMat.getHostData()[i];
		for (int j = 1; j < glmMat.getNCols(); j++) {
			os << ", " << glmMat.getHostData()[i + j * glmMat.getNRows()];
		}
		os << "]";
		if (i < glmMat.getNRows() - 1) {
			os << std::endl;
		}
	}
	os << "]";

	return os;
}
