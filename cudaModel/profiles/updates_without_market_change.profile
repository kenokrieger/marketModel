==19928== NVPROF is profiling process 19928, command: builds\profiling\profiling_without_market_change.exe
==19928== Profiling application: builds\profiling\profiling_without_market_change.exe
==19928== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   43.91%  8.43903s     10000  843.90us  840.97us  1.5147ms  void update_agents<bool=1>(char*, char const *, float const *, __int64, __int64)
                   43.90%  8.43793s     10000  843.79us  840.87us  1.6281ms  void update_agents<bool=0>(char*, char const *, float const *, __int64, __int64)
                   12.19%  2.34267s     20002  117.12us  114.08us  304.89us  void gen_sequenced_Philox<curandStatePhilox4_32_10, float4, int, __operator_&__(float4 curand_uniform_noargs_philox<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, int)), rng_config<curandStatePhilox4_32_10, curandOrdering=101>>(curandStatePhilox4_32_10*, float4*, __int64, __int64, int)
                    0.00%  187.23us         2  93.613us  93.597us  93.630us  init_agents(char*, float const *, __int64, __int64)
                    0.00%  42.943us         1  42.943us  42.943us  42.943us  generate_seed_pseudo(__int64, __int64, curandStatePhilox4_32_10*)
      API calls:   87.08%  17.0352s     40005  425.83us  2.3000us  16.932ms  cudaLaunchKernel
                   10.10%  1.97563s         3  658.54ms  458.50us  1.97469s  cudaDeviceSynchronize
                    1.35%  263.18ms     10000  26.317us  4.6000us  240.80us  cudaSetDevice
                    1.05%  205.82ms         2  102.91ms  1.3000us  205.82ms  cudaFree
                    0.29%  57.277ms         5  11.455ms  124.10us  56.595ms  cudaMalloc
                    0.13%  24.953ms     40006     623ns     100ns  265.30us  cudaGetLastError
                    0.00%  44.800us         1  44.800us  44.800us  44.800us  cuModuleUnload
                    0.00%  26.300us         2  13.150us  10.100us  16.200us  cuDeviceTotalMem
                    0.00%  25.000us       199     125ns     100ns     700ns  cuDeviceGetAttribute
                    0.00%  6.7000us         1  6.7000us  6.7000us  6.7000us  cudaGetDevice
                    0.00%  6.0000us         4  1.5000us     200ns  5.0000us  cuDeviceGetCount
                    0.00%  4.2000us         2  2.1000us  1.2000us  3.0000us  cudaGetDeviceProperties
                    0.00%  3.0000us         1  3.0000us  3.0000us  3.0000us  cuInit
                    0.00%  2.1000us         3     700ns     200ns  1.7000us  cuDeviceGet
                    0.00%  1.1000us         2     550ns     300ns     800ns  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDevicePrimaryCtxRelease
                    0.00%     600ns         1     600ns     600ns     600ns  cuDriverGetVersion
                    0.00%     500ns         2     250ns     200ns     300ns  cuDeviceGetLuid
                    0.00%     400ns         1     400ns     400ns     400ns  cudaGetDeviceCount
                    0.00%     300ns         2     150ns     100ns     200ns  cuDeviceGetUuid
