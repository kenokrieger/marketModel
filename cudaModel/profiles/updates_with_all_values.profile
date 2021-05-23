==1732== NVPROF is profiling process 1732, command: builds\profiling\profiling_with_all_values.exe
==1732== Profiling application: builds\profiling\profiling_with_all_values.exe
==1732== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.12%  10.7284s     10000  1.0728ms  1.0574ms  2.6993ms  void update_agents<bool=0>(char*, int*, char const *, float const *, float, float, float, __int64, __int64)
                   45.00%  10.6999s     10000  1.0700ms  1.0579ms  1.9991ms  void update_agents<bool=1>(char*, int*, char const *, float const *, float, float, float, __int64, __int64)
                    9.89%  2.35136s     20002  117.56us  114.27us  338.36us  void gen_sequenced_Philox<curandStatePhilox4_32_10, float4, int, __operator_&__(float4 curand_uniform_noargs_philox<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, int)), rng_config<curandStatePhilox4_32_10, curandOrdering=101>>(curandStatePhilox4_32_10*, float4*, __int64, __int64, int)
                    0.00%  187.80us         2  93.902us  93.790us  94.014us  init_agents(char*, float const *, __int64, __int64)
                    0.00%  42.815us         1  42.815us  42.815us  42.815us  generate_seed_pseudo(__int64, __int64, curandStatePhilox4_32_10*)
      API calls:   87.86%  21.1260s     40005  528.08us  2.3000us  16.260ms  cudaLaunchKernel
                   10.23%  2.46052s         3  820.17ms  407.10us  2.45962s  cudaDeviceSynchronize
                    0.95%  227.65ms     10000  22.764us  4.6000us  309.40us  cudaSetDevice
                    0.86%  206.38ms         2  103.19ms     800ns  206.38ms  cudaFree
                    0.10%  23.244ms     40006     581ns     100ns  299.40us  cudaGetLastError
                    0.01%  1.2056ms         5  241.12us  112.00us  666.10us  cudaMalloc
                    0.00%  44.500us         1  44.500us  44.500us  44.500us  cuModuleUnload
                    0.00%  24.200us       199     121ns     100ns     700ns  cuDeviceGetAttribute
                    0.00%  22.800us         2  11.400us  9.0000us  13.800us  cuDeviceTotalMem
                    0.00%  6.8000us         1  6.8000us  6.8000us  6.8000us  cudaGetDevice
                    0.00%  6.1000us         4  1.5250us     200ns  5.2000us  cuDeviceGetCount
                    0.00%  4.2000us         2  2.1000us  1.3000us  2.9000us  cudaGetDeviceProperties
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuInit
                    0.00%  1.2000us         3     400ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         2     500ns     300ns     700ns  cuDeviceGetName
                    0.00%     600ns         2     300ns     200ns     400ns  cuDeviceGetLuid
                    0.00%     600ns         1     600ns     600ns     600ns  cuDriverGetVersion
                    0.00%     500ns         1     500ns     500ns     500ns  cuDevicePrimaryCtxRelease
                    0.00%     500ns         1     500ns     500ns     500ns  cudaGetDeviceCount
                    0.00%     400ns         2     200ns     200ns     200ns  cuDeviceGetUuid
