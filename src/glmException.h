#ifndef GLMEXCEPTION_H_
#define GLMEXCEPTION_H_

#include <exception>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Base class for GLM Exceptions //////////////////////////////////////////////

class glmException : public std::exception {
protected:
	int lineNumber;
	const char *fileName;

public:
	glmException(int _lineNumber, const char *_fileName) {
		lineNumber = _lineNumber;
		fileName = _fileName;
	}

	virtual const char* what() const throw() {
		return "Generic gpuglm exception";
	}

	int getLineNumber(void) const { return lineNumber; };
	const char* getFileName(void) const { return fileName; };
};

// GLM Exceptions thrown by CUDA //////////////////////////////////////////////

class glmCudaException : public glmException {
protected:
	cudaError_t cudaErrorCode;

public:
	glmCudaException(cudaError_t _cudaErrorCode, int _lineNumber,
			const char *_fileName) :
			glmException(_lineNumber, _fileName) {
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

// GLM Exceptions thrown by cuBLAS ////////////////////////////////////////////

class glmCublasException : public glmException {
protected:
	cublasStatus_t cublasErrorCode;

public:
	glmCublasException(cublasStatus_t _cublasErrorCode, int _lineNumber,
			const char *_fileName) :
			glmException(_lineNumber, _fileName) {
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
	}
};

// GLM Exceptions thrown by cusolver //////////////////////////////////////////

class glmCusolverException : public glmException {
protected:
	cusolverStatus_t cusolverErrorCode;

public:
	glmCusolverException(cusolverStatus_t _cusolverErrorCode, int _lineNumber,
			const char *_fileName) :
			glmException(_lineNumber, _fileName) {
		cusolverErrorCode = _cusolverErrorCode;
	};

	const char* what() const throw() {
		switch(cusolverErrorCode) {
		case CUSOLVER_STATUS_NOT_INITIALIZED:
			return "Cusolver Not Initialized";
		case CUSOLVER_STATUS_ALLOC_FAILED:
			return "Resource Allocation Failed";
		case CUSOLVER_STATUS_INVALID_VALUE:
			return "Invalid Value";
		case CUSOLVER_STATUS_ARCH_MISMATCH:
			return "Architecture Mismatch";
		case CUSOLVER_STATUS_EXECUTION_FAILED:
			return "Execution Failed";
		case CUSOLVER_STATUS_INTERNAL_ERROR:
			return "Internal Error";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "Matrix Type Not Supported";
		default:
			return "Unspecified Error - Consult Documentation";
		}
	}
};

// Macros to wrap calls to CUDA/CUBLAS/CUSOLVER and throw exceptions //////////

#define GLM_ERROR throw glmException(__LINE__, __FILE__);

#define CUDA_WRAP(value) {													\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		throw glmCudaException(_m_cudaStat, __LINE__, __FILE__);			\
	} }

#define CUBLAS_WRAP(value) {												\
	cublasStatus_t _m_cublasStat = value;									\
	if (_m_cublasStat != CUBLAS_STATUS_SUCCESS) {							\
		throw glmCublasException(_m_cublasStat, __LINE__, __FILE__);		\
	} }

#define CUSOLVER_WRAP(value) {												\
	cusolverStatus_t _m_cusolverStat = value;								\
	if (_m_cusolverStat != CUSOLVER_STATUS_SUCCESS) {						\
		throw glmCusolverException(_m_cusolverStat, __LINE__, __FILE__);	\
	} }

#endif /* GLMEXCEPTION_H_ */
