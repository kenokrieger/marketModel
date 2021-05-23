==14784== NVPROF is profiling process 14784, command: builds\profiling\profiling_without_market_coupling.exe
==14784== Profiling application: builds\profiling\profiling_without_market_coupling.exe
==14784== Warning: 2 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14784== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.00%  8.56944s     10000  856.94us  850.83us  1.5166ms  void update_agents<bool=0>(char*, int*, char const *, float const *, __int64, __int64)
                   43.93%  8.55689s     10000  855.69us  850.96us  1.5484ms  void update_agents<bool=1>(char*, int*, char const *, float const *, __int64, __int64)
                   12.07%  2.34983s     20002  117.48us  113.95us  416.98us  void gen_sequenced_Philox<curandStatePhilox4_32_10, float4, int, __operator_&__(float4 curand_uniform_noargs_philox<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, int)), rng_config<curandStatePhilox4_32_10, curandOrdering=101>>(curandStatePhilox4_32_10*, float4*, __int64, __int64, int)
                    0.00%  187.55us         2  93.773us  93.693us  93.854us  init_agents(char*, float const *, __int64, __int64)
                    0.00%  42.783us         1  42.783us  42.783us  42.783us  generate_seed_pseudo(__int64, __int64, curandStatePhilox4_32_10*)
      API calls:   87.16%  17.2460s     40005  431.10us  2.3000us  16.202ms  cudaLaunchKernel
                   10.14%  2.00655s         3  668.85ms  466.30us  2.00561s  cudaDeviceSynchronize
                    1.25%  247.88ms     10000  24.787us  3.2000us  370.20us  cudaSetDevice
                    1.04%  205.90ms         2  102.95ms     900ns  205.90ms  cudaFree
                    0.28%  55.838ms         5  11.168ms  109.90us  55.292ms  cudaMalloc
                    0.12%  23.226ms     40006     580ns       0ns  292.40us  cudaGetLastError
                    0.00%  42.500us         1  42.500us  42.500us  42.500us  cuModuleUnload
                    0.00%  34.600us         2  17.300us  16.100us  18.500us  cuDeviceTotalMem
                    0.00%  26.600us       199     133ns     100ns     700ns  cuDeviceGetAttribute
                    0.00%  11.600us         2  5.8000us  1.9000us  9.7000us  cudaGetDeviceProperties
                    0.00%  9.0000us         1  9.0000us  9.0000us  9.0000us  cudaGetDevice
                    0.00%  6.3000us         4  1.5750us     200ns  5.1000us  cuDeviceGetCount
                    0.00%  3.5000us         1  3.5000us  3.5000us  3.5000us  cuInit
                    0.00%  1.9000us         3     633ns     200ns  1.4000us  cuDeviceGet
                    0.00%  1.6000us         2     800ns     600ns  1.0000us  cuDeviceGetName
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDriverGetVersion
                    0.00%     600ns         2     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     500ns         1     500ns     500ns     500ns  cuDevicePrimaryCtxRelease
                    0.00%     400ns         2     200ns     200ns     200ns  cuDeviceGetUuid
                    0.00%     400ns         1     400ns     400ns     400ns  cudaGetDeviceCount
