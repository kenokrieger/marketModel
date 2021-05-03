#include <iostream>
#include <string>

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#include "cudamacro.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#define THREADS 128

int main() {
		
		return 0;
}
