==18088== NVPROF is profiling process 18088, command: builds\profiling\profiling_as_source.exe
==18088== Profiling application: builds\profiling\profiling_as_source.exe
==18088== Warning: 1 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==18088== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   43.91%  8.43532s     10000  843.53us  841.04us  1.2496ms  void update_agents<bool=1>(char*, char const *, float const *, __int64, __int64)
                   43.90%  8.43444s     10000  843.44us  840.43us  1.2584ms  void update_agents<bool=0>(char*, char const *, float const *, __int64, __int64)
                   12.19%  2.34090s     20002  117.03us  114.08us  319.83us  void gen_sequenced_Philox<curandStatePhilox4_32_10, float4, int, __operator_&__(float4 curand_uniform_noargs_philox<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, int)), rng_config<curandStatePhilox4_32_10, curandOrdering=101>>(curandStatePhilox4_32_10*, float4*, __int64, __int64, int)
                    0.00%  188.41us         2  94.206us  93.886us  94.526us  init_agents(char*, float const *, __int64, __int64)
                    0.00%  44.415us         1  44.415us  44.415us  44.415us  generate_seed_pseudo(__int64, __int64, curandStatePhilox4_32_10*)
      API calls:   87.09%  17.0339s     40005  425.79us  2.4000us  16.162ms  cudaLaunchKernel
                   10.08%  1.97231s         3  657.44ms  471.00us  1.97136s  cudaDeviceSynchronize
                    1.32%  258.75ms     10000  25.875us  4.5000us  331.90us  cudaSetDevice
                    1.09%  213.42ms         2  106.71ms  1.1000us  213.41ms  cudaFree
                    0.28%  55.166ms         5  11.033ms  115.80us  54.551ms  cudaMalloc
                    0.13%  24.448ms     40006     611ns     100ns  304.10us  cudaGetLastError
                    0.00%  41.200us         1  41.200us  41.200us  41.200us  cuModuleUnload
                    0.00%  24.100us       199     121ns       0ns     800ns  cuDeviceGetAttribute
                    0.00%  23.800us         2  11.900us  9.3000us  14.500us  cuDeviceTotalMem
                    0.00%  7.6000us         4  1.9000us     200ns  6.5000us  cuDeviceGetCount
                    0.00%  6.5000us         1  6.5000us  6.5000us  6.5000us  cudaGetDevice
                    0.00%  4.3000us         2  2.1500us  1.5000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuInit
                    0.00%  1.8000us         3     600ns     200ns  1.4000us  cuDeviceGet
                    0.00%  1.0000us         2     500ns     300ns     700ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDriverGetVersion
                    0.00%     600ns         1     600ns     600ns     600ns  cuDevicePrimaryCtxRelease
                    0.00%     500ns         2     250ns     200ns     300ns  cuDeviceGetUuid
                    0.00%     500ns         2     250ns     200ns     300ns  cuDeviceGetLuid
                    0.00%     400ns         1     400ns     400ns     400ns  cudaGetDeviceCount
