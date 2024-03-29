
#ifndef GLMARRAY_H_
#define GLMARRAY_H_

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpuglmConfig.h"
#include "glmException.h"

///////////////////////////////////////////////////////////////////////////////
// Class definition: glmArray                                                //
///////////////////////////////////////////////////////////////////////////////

// This class should never be used directly; it serves as a parent class for
// glmVector and glmMatrix, and takes care of the memory management and other
// housekeeping functions that both child classes in common.

template <typename T>
class glmArray {

protected:
	T *hostData, *deviceData;
	bool sharedHost, sharedDevice;
	int length, totalSize;

public:
	// Constructors/Destructor ////////////////////////////////////////////////
	glmArray(int _length, bool initHost = false,
			bool initDevice = false, bool initialize = false)  {
		length = _length;
		sharedHost = false;
		sharedDevice = false;
		totalSize = length * sizeof(T);

		if (initHost) {
			this->hostData = (T*) malloc(totalSize);
			if (initialize) {
				memset((void*) this->hostData, 0, totalSize);
			}
		} else {
			hostData = NULL;
		}

		if (initDevice) {
			CUDA_WRAP(cudaMalloc((void **) &deviceData, totalSize));
			if (initialize) {
				CUDA_WRAP(cudaMemset((void*) deviceData, 0, totalSize));
			}
		} else {
			deviceData = NULL;
		}

		return;
	}

	glmArray(T *_data, int _length, bool deepCopy = false,
			location_t dataLocation = LOCATION_HOST)  {
		length = _length;
		totalSize = length * sizeof(T);

		if (dataLocation == LOCATION_HOST) {
			if (deepCopy) {
				this->hostData = (T*) malloc(totalSize);
				memcpy(this->hostData, _data, totalSize);
				sharedHost = false;
			} else {
				hostData = _data;
				sharedHost = true;
			}
			deviceData = NULL;
			sharedDevice = false;
		} else {
			if (deepCopy) {
				CUDA_WRAP(cudaMalloc((void **) &deviceData, totalSize));
				CUDA_WRAP(cudaMemcpy(this->deviceData, _data, totalSize,
						cudaMemcpyDeviceToDevice));
				sharedDevice = false;
			} else {
				deviceData = _data;
				sharedDevice = true;
			}
			hostData = NULL;
			sharedHost = false;
		}

		return;
	}

	~glmArray(void)  {
		if (!sharedHost) {
			free(hostData);
		}

		if (!sharedDevice) {
			cudaFree(deviceData);
		}
	}

	// Data Management Functions //////////////////////////////////////////////
	void copyDeviceToHost(void) {
		if (deviceData == NULL) {
			GLM_ERROR;
		}

		if (hostData == NULL) {
			hostData = (T*) malloc(totalSize);
		}

		CUDA_WRAP(cudaMemcpy(hostData, deviceData, totalSize,
				cudaMemcpyDeviceToHost));
		return;
	}

	void copyHostToDevice(void) {
		if (hostData == NULL) {
			GLM_ERROR;
		}

		if (deviceData == NULL) {
			CUDA_WRAP(cudaMalloc((void **) &deviceData, totalSize));
		}

		CUDA_WRAP(cudaMemcpy(deviceData, hostData, totalSize,
				cudaMemcpyHostToDevice));
		return;
	}

	// Accessor Functions /////////////////////////////////////////////////////
	T* getHostData(void) const { return hostData; };
	T* getDeviceData(void) const { return deviceData; };
	int getLength(void) const { return length; };

	void setSharedHost(bool _sharedHost) { sharedHost = _sharedHost; };
};

///////////////////////////////////////////////////////////////////////////////
// Class Definition: glmVector                                               //
///////////////////////////////////////////////////////////////////////////////

template <typename T>
class glmVector : public glmArray<T> {

public:
	// Constructors ///////////////////////////////////////////////////////////
	glmVector(int _length, bool initHost = false,
			bool initDevice = false, bool initialize = false) :
				glmArray<T>(_length, initHost, initDevice, initialize) { }
	glmVector(T *_data, int _length, bool deepCopy = false,
			location_t dataLocation = LOCATION_HOST) :
		glmArray<T>(_data, _length, deepCopy, dataLocation) { }
	~glmVector() { }

	// Methods that only make sense for vectors ///////////////////////////////
	int getNumBlocks() {
		// The return value is the minimum number of thread blocks that will
		// contain all of the data.
		if (this->length % THREADS_PER_BLOCK == 0) {
			return this->length / THREADS_PER_BLOCK;
		} else {
			return this->length / THREADS_PER_BLOCK + 1;
		}
	}

	T* getHostElement (int index) const {
		return this->hostData + index;
	};

	T* getDeviceElement (int index) const {
		return this->deviceData + index;
	};
};

///////////////////////////////////////////////////////////////////////////////
// Class Definition: glmMatrix                                               //
///////////////////////////////////////////////////////////////////////////////

template <typename T>
class glmMatrix : public glmArray<T> {

protected:
	int nRows, nCols;

public:
	// Constructors ///////////////////////////////////////////////////////////

	// Construct an empty glmMatrix
	glmMatrix(int _nRows, int _nCols, bool initHost = false,
			bool initDevice = false, bool initialize = false) :
				glmArray<T>(_nRows * _nCols, initHost, initDevice,
						initialize) {
		nRows = _nRows;
		nCols = _nCols;
	}

	// Construct a glmMatrix from pre-existing continguous data
	glmMatrix(T *_data, int _nRows, int _nCols, bool deepCopy = false,
			location_t dataLocation = LOCATION_HOST) :
		glmArray<T>(_data, _nRows * _nCols, deepCopy, dataLocation) {
		nRows = _nRows;
		nCols = _nCols;
	}

	~glmMatrix() { }

	// Matrix-specific Functions //////////////////////////////////////////////
	void copyColumnFromHost(T *_data, int colNum) {
		if (colNum < nCols) {
			T* colAddress = this->deviceData + (colNum * nRows);
			CUDA_WRAP(cudaMemcpy(colAddress, _data, sizeof(T) * nRows,
					cudaMemcpyHostToDevice));
		}

		return;
	}

	glmVector<T>* getDeviceColumn(int colNum) {
		return new glmVector<T>(this->deviceData + colNum * this->nRows,
				this->nRows, false, LOCATION_DEVICE);
	}

	void rowProduct(cublasHandle_t handle,
			glmVector<T> *columnVector, glmVector<T> *result) {
		const T ONE = 1.0;
		const T ZERO = 0.0;

		CUBLAS_WRAP(GEMV(handle, CUBLAS_OP_N, this->nRows, this->nCols, &ONE,
				this->deviceData, this->nRows, columnVector->getDeviceData(),
				1, &ZERO, result->getDeviceData(), 1));

		return;
	}

	void columnProduct(cublasHandle_t handle,
			glmVector<T> *columnVector, glmVector<T> *result) {
		const T ONE = 1.0;
		const T ZERO = 0.0;

		CUBLAS_WRAP(GEMV(handle, CUBLAS_OP_T, this->nRows, this->nCols, &ONE,
				this->deviceData, this->nRows, columnVector->getDeviceData(),
				1, &ZERO, result->getDeviceData(), 1));
	}

	// Matrix-specific Accessors //////////////////////////////////////////////
	int getNRows(void) const { return nRows; };
	int getNCols(void) const { return nCols; };

	T* getHostElement(int row, int column) const {
		return this->hostData + column * nRows + row;
	};

	T* getDeviceElement(int row, int column) const {
		return this->deviceData + column * nRows + row;
	};
};

///////////////////////////////////////////////////////////////////////////////
// glmVector Utility Functions ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Copying ////////////////////////////////////////////////////////////////////
void copyDeviceToDevice(glmVector<num_t> *destination,
		glmVector<num_t> *source);

// Vector Arithmetic //////////////////////////////////////////////////////////
void vectorSum(glmVector<num_t> *vector, num_t *result);
void vectorAddScalar(glmVector<num_t> *a, num_t b, glmVector<num_t> *c);
void vectorAdd(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c);
void vectorDifference(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c);
void vectorMultiply(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c);

void factorProduct(glmVector<factor_t> *factor, int numFactorLevels,
		glmVector<num_t> *numeric, num_t *result, int stride=1);
void doubleFactorProduct(glmVector<factor_t> *factor1,
		glmVector<factor_t> *factor2, int numFactor1Levels,
		int numFactor2Levels, glmVector<num_t> *numeric, num_t *result,
		int stride=1);

// Printing ///////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const glmVector<factor_t>& glmVec);
std::ostream& operator<<(std::ostream& os, const glmVector<num_t>& glmVec);
std::ostream& operator<<(std::ostream& os, const glmMatrix<factor_t>& glmMat);
std::ostream& operator<<(std::ostream& os, const glmMatrix<num_t>& glmMat);

#endif /* GLMARRAY_H_ */
