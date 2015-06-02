#ifndef GLMEXCEPTION_H_
#define GLMEXCEPTION_H_

#include <exception>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>

class glmException : public std::exception {
	virtual const char* what() const throw() {
		return "Generic gpuglm exception";
	}
};

class glmCudaException : public glmException {
protected:
	cudaError_t cudaErrorCode;

public:
	glmCudaException(cudaError_t _cudaErrorCode) {
		cudaErrorCode = _cudaErrorCode;
	}

	const char* what() const throw() {
		switch (cudaErrorCode) {
		case 1:
			return "Missing Configuration";
		case 2:
			return "Memory Allocation";
		case 3:
			return "Initialization";
		case 4:
			return "Kernel Launch";
		case 7:
			return "Kernel Launch - Device Out Of Resources";
		case 8:
			return "Invalid Device Function";
		case 9:
			return "Invalid Configuration";
		case 11:
			return "Invalid Value";
		case 16:
			return "Invalid Host Pointer";
		case 17:
			return "Invalid Device Pointer";
		default:
			return "Unspecified Error - Consult Documentation";
		}
	};

	cudaError_t getCudaErrorCode(void) const { return cudaErrorCode; };
};

class glmCublasException : public glmException {
protected:
	cublasStatus_t cublasErrorCode;

public:
	glmCublasException(cublasStatus_t _cublasErrorCode) {
		cublasErrorCode = _cublasErrorCode;
	}

	const char* what() const throw() {
		switch (cublasErrorCode) {
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "Not Initialized";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "Resource Allocation Failed";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "Invalid Value Passed To Function";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "Mapping Error";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "Execution Failed";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "Internal Error";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "Functionality Not Supported";
		default:
			return "Unspecified Error - Consult Documentation";
		}
	};
};

#define CUDA_WRAP(value) {													\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		throw glmCudaException(_m_cudaStat);								\
	} }

#define CUBLAS_WRAP(value) {												\
	cublasStatus_t _m_cublasStat = value;									\
	if (_m_cublasStat != CUBLAS_STATUS_SUCCESS) {							\
		throw glmCublasException(_m_cublasStat);							\
	} }

#endif /* GLMEXCEPTION_H_ */
