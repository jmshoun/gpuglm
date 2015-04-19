
#ifndef GPUGLMCONFIG_H_
#define GPUGLMCONFIG_H_

typedef double num_t;

typedef enum {
	LOCATION_HOST,
	LOCATION_DEVICE
} location_t;

const int THREADS_PER_BLOCK = 512;

#endif /* GPUGLMCONFIG_H_ */
