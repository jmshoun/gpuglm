
#ifndef GLMARRAY_H_
#define GLMARRAY_H_

#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "gpuglmConfig.h"

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
			cudaMalloc((void **) &deviceData, totalSize);
			if (initialize) {
				cudaMemset((void*) deviceData, 0, totalSize);
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
				cudaMalloc((void **) &deviceData, totalSize);
				cudaMemcpy(this->deviceData, _data, totalSize,
						cudaMemcpyDeviceToDevice);
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

		cudaMemcpy(hostData, deviceData, totalSize, cudaMemcpyDeviceToHost);
		return;
	}

	void copyHostToDevice(void) {
		if (deviceData == NULL) {
			cudaMalloc((void **) &deviceData, totalSize);
		}

		cudaMemcpy(deviceData, hostData, totalSize, cudaMemcpyHostToDevice);
		return;
	}

	// Accessor Functions /////////////////////////////////////////////////////
	T* getHostData(void) { return hostData; };
	T* getDeviceData(void) { return deviceData; };
	int getLength(void) { return length; };
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
};

#endif /* GLMARRAY_H_ */
