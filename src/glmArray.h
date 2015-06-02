
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
		if (hostData == NULL) {
			hostData = (T*) malloc(totalSize);
		}

		CUDA_WRAP(cudaMemcpy(hostData, deviceData, totalSize,
				cudaMemcpyDeviceToHost));
		return;
	}

	void copyHostToDevice(void) {
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
	glmMatrix(int _nRows, int _nCols, bool initHost = false,
			bool initDevice = false, bool initialize = false) :
				glmArray<T>(_nRows * _nCols, initHost, initDevice,
						initialize) {
		nRows = _nRows;
		nCols = _nCols;
	}

	glmMatrix(T *_data, int _nRows, int _nCols, bool deepCopy = false,
			location_t dataLocation = LOCATION_HOST) :
		glmArray<T>(_data, _nRows * _nCols, deepCopy, dataLocation) {
		nRows = _nRows;
		nCols = _nCols;
	}

	~glmMatrix() { }

	// Matrix-specific Functions //////////////////////////////////////////////
	void copyRowFromHost(T *_data, int colNum) {
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
};

void copyDeviceToDevice(glmVector<num_t> *destination,
		glmVector<num_t> *source);

void vectorSum(glmVector<num_t> *vector, glmArray<num_t> *result,
		int resultIndex = 0);

void vectorAddScalar(glmVector<num_t> *a, num_t b, glmVector<num_t> *c);
void vectorAdd(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c);
void vectorDifference(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c);
void vectorMultiply(glmVector<num_t> *a, glmVector<num_t> *b,
		glmVector<num_t> *c);

std::ostream& operator<<(std::ostream& os, const glmVector<num_t>& glmVec);
std::ostream& operator<<(std::ostream& os, const glmMatrix<num_t>& glmMat);

#endif /* GLMARRAY_H_ */
