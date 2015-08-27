#include "glmArray.h"

// Copying ////////////////////////////////////////////////////////////////////

void copyDeviceToDevice(glmVector<num_t> *destination,
		glmVector<num_t> *source) {
	CUDA_WRAP(cudaMemcpy(destination->getDeviceData(), source->getDeviceData(),
			destination->getLength() * sizeof(num_t),
			cudaMemcpyDeviceToDevice));
	return;
}

// Kernels for vector arithmetic //////////////////////////////////////////////

__global__ void vectorSumKernel(int n, num_t *x, num_t *sum) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ num_t sharedX[];

	// load data into __shared__ memory
	num_t xElement = 0.0;
	if (i < n) {
		xElement = x[i];
	}
	sharedX[threadIdx.x] = xElement;
	__syncthreads();

	// each loop compresses the data by a factor of 2
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (threadIdx.x < offset) {
			sharedX[threadIdx.x] += sharedX[threadIdx.x + offset];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		sum[blockIdx.x] = sharedX[0];
	}

	return;
}

__global__ void vectorAddScalarKernel(int n, num_t *a, num_t b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b;
	}

	return;
}

__global__ void vectorAddKernel(int n, num_t *a, num_t *b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b[i];
	}

	return;
}

__global__ void vectorDifferenceKernel(int n, num_t *a, num_t *b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] - b[i];
	}

	return;
}

__global__ void vectorMultiplyKernel(int n, num_t *a, num_t *b, num_t *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		c[i] = a[i] * b[i];
	}

	return;
}

__global__ void factorProductKernel(int n, int groupSize, factor_t *factor,
		num_t *numeric, num_t *result) {
	unsigned int offset = threadIdx.x;
	unsigned int i;
	factor_t factorValue;

	for (i = offset * groupSize; i < (offset + 1) * groupSize; i++) {
		if (i < n) {
			factorValue = factor[i];
			if (factorValue > 1) {
				result[blockDim.x * (factorValue - 2) + offset] += numeric[i];
			}
		}
	}

	return;
}

// Vector Arithmetic Functions ////////////////////////////////////////////////

void vectorSumSimple(int length, num_t *input, num_t *output) {
	int blockCount = length / THREADS_PER_BLOCK +
			(length % THREADS_PER_BLOCK ? 1 : 0);
	int sharedMemorySize = THREADS_PER_BLOCK * sizeof(num_t);

	vectorSumKernel<<<blockCount, THREADS_PER_BLOCK,
			sharedMemorySize>>>(length, input, output);

	return;
}

void vectorSumRecursive(int length, num_t *input, num_t *output) {
	if (length <= THREADS_PER_BLOCK) {
		// Base case: sum the vector with a single threadblock and save to
		// the (single) output
		vectorSumSimple(length, input, output);
	} else {
		// Recursive case: allocate space for the partial sums...
		int tempLength = length / THREADS_PER_BLOCK +
				(length % THREADS_PER_BLOCK ? 1 : 0);
		num_t *tempOutput = NULL;
		CUDA_WRAP(cudaMalloc((void **) &tempOutput,
				tempLength * sizeof(num_t)));

		// ...calculate the multiple partial sums...
		vectorSumSimple(length, input, tempOutput);

		// ...and then recurse on the partial sums
		vectorSumRecursive(tempLength, tempOutput, output);
		CUDA_WRAP(cudaFree(tempOutput));
	}

	return;
}

void vectorSum(glmVector<num_t> *vector, num_t *result) {
	vectorSumRecursive(vector->getLength(), vector->getDeviceData(), result);

	return;
}

void vectorAddScalar(glmVector<num_t> *a, num_t b, glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorAddScalarKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b, c->getDeviceData());

	return;
}

void vectorAdd(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorAddKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b->getDeviceData(), c->getDeviceData());

	return;
}

void vectorDifference(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorDifferenceKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b->getDeviceData(), c->getDeviceData());

	return;
}

void vectorMultiply(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c) {
	int numBlocks = a->getNumBlocks();

	vectorMultiplyKernel<<<numBlocks, THREADS_PER_BLOCK>>>(a->getLength(),
			a->getDeviceData(), b->getDeviceData(), c->getDeviceData());

	return;
}

// Factor Product Host-Side Function //////////////////////////////////////////

void factorProduct(glmVector<factor_t> *factor, int numFactorLevels,
		glmVector<num_t> *numeric, num_t *result, int stride) {
	int n = factor->getLength();
	int groupSize = n / THREADS_PER_BLOCK + (n % THREADS_PER_BLOCK ? 1 : 0);
	int tempResultSize = sizeof(num_t) * THREADS_PER_BLOCK * numFactorLevels;
	num_t *tempResult = NULL;

	CUDA_WRAP(cudaMalloc((void **) &tempResult, tempResultSize));
	CUDA_WRAP(cudaMemset((void *) tempResult, 0, tempResultSize));

	factorProductKernel<<<1, THREADS_PER_BLOCK>>>(n, groupSize,
			factor->getDeviceData(), numeric->getDeviceData(),
			tempResult);

	for (int i = 0; i < numFactorLevels; i++) {
		vectorSumSimple(THREADS_PER_BLOCK, tempResult + THREADS_PER_BLOCK * i,
				result + i * stride);
	}

	CUDA_WRAP(cudaFree(tempResult));

	return;
}

// Print Functions ////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, const glmVector<factor_t>& glmVec) {
	os << "[" << ((int) glmVec.getHostData()[0]);
	for (int i = 1; i < glmVec.getLength(); i++) {
		os << ", " << ((int) glmVec.getHostData()[i]);
	}
	os << "]";

	return os;
};

std::ostream& operator<<(std::ostream& os, const glmVector<num_t>& glmVec) {
	os << "[" << glmVec.getHostData()[0];
	for (int i = 1; i < glmVec.getLength(); i++) {
		os << ", " << glmVec.getHostData()[i];
	}
	os << "]";

	return os;
};

std::ostream& operator<<(std::ostream& os, const glmMatrix<factor_t>& glmMat) {
	for (int i = 0; i < glmMat.getNRows(); i++) {
		if (i == 0) {
			os << "[[";
		} else {
			os << " [";
		}

		os << ((int) glmMat.getHostData()[i]);
		for (int j = 1; j < glmMat.getNCols(); j++) {
			os << ", " << ((int) glmMat.getHostData()[i + j * glmMat.getNRows()]);
		}

		os << "]";
		if (i < glmMat.getNRows() - 1) {
			os << std::endl;
		}
	}
	os << "]";

	return os;
}

std::ostream& operator<<(std::ostream& os, const glmMatrix<num_t>& glmMat) {
	for (int rowNum = 0; rowNum < glmMat.getNRows(); rowNum++) {
		if (rowNum == 0) {
			os << "[[";
		} else {
			os << " [";
		}

		os << glmMat.getHostData()[rowNum];
		for (int colNum = 1; colNum < glmMat.getNCols(); colNum++) {
			os << ", " <<
					glmMat.getHostData()[rowNum + colNum * glmMat.getNRows()];
		}

		os << "]";
		if (rowNum < glmMat.getNRows() - 1) {
			os << std::endl;
		}
	}
	os << "]";

	return os;
}
