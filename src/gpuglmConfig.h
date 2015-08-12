
#ifndef GPUGLMCONFIG_H_
#define GPUGLMCONFIG_H_

typedef double num_t;
typedef unsigned char factor_t;
typedef unsigned short int long_factor_t;

typedef enum {
	LOCATION_HOST,
	LOCATION_DEVICE
} location_t;

#define POTRF_B	cusolverDnDpotrf_bufferSize
#define POTRF 	cusolverDnDpotrf
#define POTRS 	cusolverDnDpotrs
#define AXPY  	cublasDaxpy
#define GEMV  	cublasDgemv
#define DOT   	cublasDdot
#define AMAX	cublasIdamax

const int THREADS_PER_BLOCK = 512;

#endif /* GPUGLMCONFIG_H_ */
